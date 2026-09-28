"""POST /v1/audio/transcriptions: OpenAI-compatible transcription via faster-whisper."""

import logging
import time
from typing import Annotated, Literal

import faster_whisper.transcribe
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from faster_whisper import BatchedInferencePipeline
from faster_whisper.vad import VadOptions, get_speech_timestamps

from meetscribe_server.audio import SAMPLE_RATE, decode_upload
from meetscribe_server.models import ClientGone, ModelStore, acquire_slot
from meetscribe_server.schemas import (
    Transcription,
    TranscriptionSegment,
    TranscriptionVerbose,
    TranscriptionWord,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcription"])

ResponseFormat = Literal["json", "text", "verbose_json"]


def segments_to_text(segments: list[faster_whisper.transcribe.Segment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def merge_clips(
    speech_chunks: list[dict[str, int]], max_span_s: float, sample_rate: int = SAMPLE_RATE
) -> list[dict[str, float]]:
    """Group consecutive VAD speech chunks into clips whose *time span* stays within max_span_s.

    This is the faster-whisper 1.1 / speaches behaviour: a clip is a contiguous stretch of the
    original audio (silence included) and becomes exactly one output segment, so segment
    boundaries fall on pauses. faster-whisper 1.2's own VAD path instead concatenates up to 30 s
    of *speech* and reports one segment spanning all of it, hiding long pauses inside a segment.
    """
    if not speech_chunks:
        return []
    max_span = int(max_span_s * sample_rate)
    clips: list[dict[str, float]] = []
    start = speech_chunks[0]["start"]
    end = speech_chunks[0]["end"]
    for chunk in speech_chunks[1:]:
        if chunk["end"] - start > max_span:
            clips.append({"start": start / sample_rate, "end": end / sample_rate})
            start = chunk["start"]
        end = chunk["end"]
    clips.append({"start": start / sample_rate, "end": end / sample_rate})
    return clips


def fixed_clips(duration_s: float, span_s: float = 30.0) -> list[dict[str, float]]:
    """Consecutive fixed windows for the no-VAD path (the batched pipeline needs explicit clips)."""
    clips = []
    start = 0.0
    while start < duration_s:
        clips.append({"start": start, "end": min(start + span_s, duration_s)})
        start += span_s
    return clips


def build_verbose_response(
    segments: list[faster_whisper.transcribe.Segment],
    language: str,
    duration: float,
    word_timestamps: bool,
) -> TranscriptionVerbose:
    return TranscriptionVerbose(
        language=language,
        duration=duration,
        text=segments_to_text(segments),
        segments=[
            TranscriptionSegment(
                id=s.id,
                seek=s.seek,
                start=s.start,
                end=s.end,
                text=s.text,
                tokens=list(s.tokens),
                temperature=s.temperature if s.temperature is not None else 0.0,
                avg_logprob=s.avg_logprob,
                compression_ratio=s.compression_ratio,
                no_speech_prob=s.no_speech_prob,
            )
            for s in segments
        ],
        words=[TranscriptionWord(start=w.start, end=w.end, word=w.word) for s in segments for w in (s.words or [])]
        if word_timestamps
        else None,
    )


async def timestamp_granularities(request: Request) -> list[str]:
    """The repeated form field timestamp_granularities[] cannot be declared with Form(alias=...)."""
    form = await request.form()
    values = [str(v) for v in form.getlist("timestamp_granularities[]")]
    return values or ["segment"]


@router.post("/v1/audio/transcriptions", response_model=None)
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    without_timestamps: Annotated[bool, Form()] = True,
) -> Response:
    if stream:
        raise HTTPException(status_code=400, detail="Streaming transcription is not supported")
    store: ModelStore = request.app.state.models
    settings = store.settings
    model_id = model or settings.stt_model
    granularities = await timestamp_granularities(request)
    word_timestamps = "word" in granularities

    audio = decode_upload(file)
    duration = audio.size / SAMPLE_RATE

    def run() -> tuple[list[faster_whisper.transcribe.Segment], str]:
        whisper = store.whisper(model_id)
        if settings.stt_vad_filter:
            vad_options = VadOptions(
                min_silence_duration_ms=settings.stt_vad_min_silence_ms,
                max_speech_duration_s=settings.stt_vad_max_speech_s,
            )
            clips = merge_clips(get_speech_timestamps(audio, vad_options), settings.stt_vad_max_speech_s)
        else:
            clips = fixed_clips(duration, settings.stt_vad_max_speech_s)
        if not clips:
            logger.info("VAD found no speech in %s", file.filename)
            return [], language or ""
        pipeline = BatchedInferencePipeline(model=whisper)
        segments, info = pipeline.transcribe(
            audio,
            task="transcribe",
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            beam_size=settings.stt_beam_size,
            batch_size=settings.stt_batch_size,
            vad_filter=False,
            clip_timestamps=clips,
            word_timestamps=word_timestamps,
            hotwords=hotwords,
            without_timestamps=without_timestamps,
        )
        return list(segments), info.language

    t0 = time.perf_counter()
    try:
        await acquire_slot(store.stt_semaphore, request)
    except ClientGone:
        logger.warning("Client disconnected while %s waited for a transcription slot; skipping", file.filename)
        return Response(status_code=499)
    try:
        segments, detected_language = await run_in_threadpool(run)
    except (ValueError, FileNotFoundError, OSError) as e:
        # faster-whisper raises these for unknown / not-downloadable model ids
        logger.warning("Transcription with model %s failed: %s", model_id, e)
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not available: {e}") from e
    finally:
        store.stt_semaphore.release()
    elapsed = time.perf_counter() - t0
    logger.info(
        "Transcribed %s: %.1fs of audio in %.2fs (rtf %.3f, %d segments, model=%s)",
        file.filename,
        duration,
        elapsed,
        elapsed / duration if duration else 0.0,
        len(segments),
        model_id,
    )

    if response_format == "text":
        return Response(content=segments_to_text(segments), media_type="text/plain")
    if response_format == "json":
        body = Transcription(text=segments_to_text(segments)).model_dump_json()
        return Response(content=body, media_type="application/json")
    verbose = build_verbose_response(segments, detected_language, duration, word_timestamps)
    return Response(content=verbose.model_dump_json(), media_type="application/json")
