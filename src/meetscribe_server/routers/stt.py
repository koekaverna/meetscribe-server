"""Transcription via faster-whisper.

* POST /v1/audio/transcriptions        OpenAI-compatible, one upload = one request.
* POST /v1/audio/transcriptions/clips  One track plus a list of clips, decoded in GPU batches.
"""

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import faster_whisper.transcribe
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from faster_whisper import BatchedInferencePipeline
from faster_whisper.vad import VadOptions, get_speech_timestamps

from meetscribe_server.audio import SAMPLE_RATE, decode_upload
from meetscribe_server.config import Settings
from meetscribe_server.hallucinations import is_known_hallucination
from meetscribe_server.models import ClientGone, ModelStore, acquire_slot
from meetscribe_server.schemas import (
    ClipSegment,
    ClipsTranscription,
    FailedClip,
    Transcription,
    TranscriptionSegment,
    TranscriptionVerbose,
    TranscriptionWord,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcription"])

ResponseFormat = Literal["json", "text", "verbose_json"]
Clip = dict[str, float]
Span = tuple[float, float]
FindSpeech = Callable[[float, float], list[Span]]

WHISPER_WINDOW_S = 30.0
FRAMES_PER_SECOND = 100  # Whisper feature frames; faster-whisper reports Segment.seek in these
MIN_CLIP_S = 0.05


def segments_to_text(segments: list[faster_whisper.transcribe.Segment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def merge_clips(speech_chunks: list[dict[str, int]], max_span_s: float, sample_rate: int = SAMPLE_RATE) -> list[Clip]:
    """Group consecutive VAD speech chunks into clips whose *time span* stays within max_span_s.

    This is the faster-whisper 1.1 / speaches behaviour: a clip is a contiguous stretch of the
    original audio (silence included) and becomes exactly one output segment, so segment
    boundaries fall on pauses. faster-whisper 1.2's own VAD path instead concatenates up to 30 s
    of *speech* and reports one segment spanning all of it, hiding long pauses inside a segment.
    """
    if not speech_chunks:
        return []
    max_span = int(max_span_s * sample_rate)
    clips: list[Clip] = []
    start = speech_chunks[0]["start"]
    end = speech_chunks[0]["end"]
    for chunk in speech_chunks[1:]:
        if chunk["end"] - start > max_span:
            clips.append({"start": start / sample_rate, "end": end / sample_rate})
            start = chunk["start"]
        end = chunk["end"]
    clips.append({"start": start / sample_rate, "end": end / sample_rate})
    return clips


def fixed_clips(duration_s: float, span_s: float = 30.0) -> list[Clip]:
    """Consecutive fixed windows for the no-VAD path (the batched pipeline needs explicit clips)."""
    clips = []
    start = 0.0
    while start < duration_s:
        clips.append({"start": start, "end": min(start + span_s, duration_s)})
        start += span_s
    return clips


def upload_clips(
    speech_chunks: list[dict[str, int]], duration_s: float, max_span_s: float, empty_fallback: bool
) -> tuple[list[Clip], bool]:
    """Clips for a plain upload. Returns (clips, used_fallback).

    A short upload in which VAD hears nothing is transcribed whole when empty_fallback is set:
    the client cut it out of a region its diarization called speech.
    """
    clips = merge_clips(speech_chunks, max_span_s)
    if clips:
        return clips, False
    if empty_fallback and MIN_CLIP_S <= duration_s <= max_span_s:
        return [{"start": 0.0, "end": duration_s}], True
    return [], False


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
        segments=[TranscriptionSegment(**segment_fields(s)) for s in segments],
        words=[TranscriptionWord(start=w.start, end=w.end, word=w.word) for s in segments for w in (s.words or [])]
        if word_timestamps
        else None,
    )


def segment_fields(s: faster_whisper.transcribe.Segment) -> dict[str, Any]:
    return {
        "id": s.id,
        "seek": s.seek,
        "start": s.start,
        "end": s.end,
        "text": s.text,
        "tokens": list(s.tokens),
        "temperature": s.temperature if s.temperature is not None else 0.0,
        "avg_logprob": s.avg_logprob,
        "compression_ratio": s.compression_ratio,
        "no_speech_prob": s.no_speech_prob,
    }


async def timestamp_granularities(request: Request) -> list[str]:
    """The repeated form field timestamp_granularities[] cannot be declared with Form(alias=...)."""
    form = await request.form()
    values = [str(v) for v in form.getlist("timestamp_granularities[]")]
    return values or ["segment"]


def require_allowed_model(settings: Settings, model_id: str) -> None:
    allowed = settings.allowed_stt_models()
    if model_id not in allowed:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' is not enabled on this server. Enabled: {sorted(allowed)}",
        )


def vad_options(settings: Settings) -> VadOptions:
    return VadOptions(
        min_silence_duration_ms=settings.stt_vad_min_silence_ms,
        max_speech_duration_s=settings.stt_vad_max_speech_s,
    )


async def run_cancellable[T](request: Request, fn: Callable[[], T], cancel: threading.Event) -> T:
    """Run fn in the thread pool; set `cancel` when the client disconnects so fn can stop early."""
    task = asyncio.ensure_future(run_in_threadpool(fn))
    while True:
        done, _ = await asyncio.wait({task}, timeout=1.0)
        if done:
            return task.result()
        if not cancel.is_set() and await request.is_disconnected():
            cancel.set()


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
    vad: Annotated[bool | None, Form()] = None,
) -> Response:
    if stream:
        raise HTTPException(status_code=400, detail="Streaming transcription is not supported")
    store: ModelStore = request.app.state.models
    settings = store.settings
    model_id = model or settings.stt_model
    require_allowed_model(settings, model_id)
    use_vad = settings.stt_vad_filter if vad is None else vad
    granularities = await timestamp_granularities(request)
    word_timestamps = "word" in granularities

    audio = decode_upload(file)
    duration = audio.size / SAMPLE_RATE
    stats = {"clips": 0, "vad_fallback": False, "hallucinations": 0}

    def run() -> tuple[list[faster_whisper.transcribe.Segment], str]:
        whisper = store.whisper(model_id)
        if use_vad:
            clips, stats["vad_fallback"] = upload_clips(
                get_speech_timestamps(audio, vad_options(settings)),
                duration,
                settings.stt_vad_max_speech_s,
                settings.stt_vad_empty_fallback,
            )
        else:
            clips = fixed_clips(duration, settings.stt_vad_max_speech_s)
        stats["clips"] = len(clips)
        if not clips:
            logger.info("VAD found no speech in %s (%.1fs)", file.filename, duration)
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
        kept = list(segments)
        if settings.stt_drop_known_hallucinations:
            total = len(kept)
            kept = [seg for seg in kept if not is_known_hallucination(seg.text)]
            stats["hallucinations"] = total - len(kept)
        return kept, info.language

    t0 = time.perf_counter()
    try:
        await acquire_slot(store.stt_semaphore, request)
    except ClientGone:
        logger.warning("Client disconnected while %s waited for a transcription slot; skipping", file.filename)
        return Response(status_code=499)
    waited = time.perf_counter() - t0
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
        "Transcribed %s: %.1fs of audio in %.2fs (rtf %.3f, waited %.2fs, %d clips%s, %d segments, "
        "%d known hallucinations dropped, model=%s)",
        file.filename,
        duration,
        elapsed,
        elapsed / duration if duration else 0.0,
        waited,
        stats["clips"],
        ", vad-empty fallback" if stats["vad_fallback"] else "",
        len(segments),
        stats["hallucinations"],
        model_id,
    )

    if response_format == "text":
        return Response(content=segments_to_text(segments), media_type="text/plain")
    if response_format == "json":
        body = Transcription(text=segments_to_text(segments)).model_dump_json()
        return Response(content=body, media_type="application/json")
    verbose = build_verbose_response(segments, detected_language, duration, word_timestamps)
    return Response(content=verbose.model_dump_json(), media_type="application/json")


# --- /v1/audio/transcriptions/clips ---


@dataclass
class ClipRequest:
    index: int
    start: float
    end: float
    speaker: str | None = None


def parse_clips(raw: str) -> list[ClipRequest]:
    """Parse the `clips` form field: a JSON list of {start, end, speaker?} in seconds."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"clips is not valid JSON: {e}") from e
    if not isinstance(data, list) or not data:
        raise HTTPException(status_code=422, detail="clips must be a non-empty JSON list")
    clips: list[ClipRequest] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail=f"clips[{i}] must be an object")
        start, end, speaker = item.get("start"), item.get("end"), item.get("speaker")
        if isinstance(start, bool) or isinstance(end, bool):
            raise HTTPException(status_code=422, detail=f"clips[{i}] needs numeric start and end")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise HTTPException(status_code=422, detail=f"clips[{i}] needs numeric start and end")
        if speaker is not None and not isinstance(speaker, str):
            raise HTTPException(status_code=422, detail=f"clips[{i}].speaker must be a string")
        clips.append(ClipRequest(index=i, start=float(start), end=float(end), speaker=speaker))
    return clips


def seek_of(start_s: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Segment.seek that faster-whisper's batched pipeline reports for a clip starting at start_s."""
    return int(int(start_s * sample_rate) / sample_rate * FRAMES_PER_SECOND)


def speech_lookup(speech_chunks: list[dict[str, int]], sample_rate: int = SAMPLE_RATE) -> FindSpeech:
    """Finder over VAD chunks computed once for the whole track."""
    spans = [(c["start"] / sample_rate, c["end"] / sample_rate) for c in speech_chunks]
    return lambda start, end: [(max(a, start), min(b, end)) for a, b in spans if b > start and a < end]


def clip_vad(audio: Any, options: VadOptions, sample_rate: int = SAMPLE_RATE) -> FindSpeech:
    """Finder that runs VAD on the clip alone, exactly as for a separately uploaded chunk."""

    def find(start: float, end: float) -> list[Span]:
        first = int(start * sample_rate)
        chunks = get_speech_timestamps(audio[first : int(end * sample_rate)], options)
        return [((first + c["start"]) / sample_rate, (first + c["end"]) / sample_rate) for c in chunks]

    return find


def split_long_clip(start: float, end: float, find_speech: FindSpeech | None) -> list[Clip]:
    """Spans of at most 30 s covering a clip that is longer than the Whisper window."""
    if find_speech is None:
        return [{"start": start + c["start"], "end": start + c["end"]} for c in fixed_clips(end - start)]
    inside = [{"start": int(a * SAMPLE_RATE), "end": int(b * SAMPLE_RATE)} for a, b in find_speech(start, end)]
    spans: list[Clip] = []
    for span in merge_clips(inside, WHISPER_WINDOW_S):
        # a single VAD chunk is at most 30 s, but guard the window anyway
        spans.extend(
            {"start": span["start"] + c["start"], "end": span["start"] + c["end"]}
            for c in fixed_clips(span["end"] - span["start"])
        )
    return spans


def plan_clips(
    requests: list[ClipRequest],
    duration_s: float,
    pad_s: float,
    find_speech: FindSpeech | None,
    empty_fallback: bool,
) -> tuple[list[tuple[ClipRequest, Clip]], list[FailedClip]]:
    """Turn requested clips into the spans that are actually decoded.

    * Padding is added on both sides, shrunk so a clip never exceeds the 30 s Whisper window.
    * With VAD, a clip is trimmed to the speech found inside it, which is what happens to a
      separately uploaded chunk.
    * A clip longer than 30 s is split on pauses into several spans (several segments, same
      clip_index), again like a separate upload.
    * Start times are made unique on the 10 ms grid: results are matched back by Segment.seek.
    """
    planned: list[tuple[ClipRequest, Clip]] = []
    failed: list[FailedClip] = []
    used_seeks: set[int] = set()
    for req in requests:
        start, end = max(req.start, 0.0), min(req.end, duration_s)
        if end - start < MIN_CLIP_S:
            reason = "out_of_range" if req.start >= duration_s or req.end <= 0 else "empty"
            failed.append(FailedClip(clip_index=req.index, reason=reason))
            continue
        if end - start > WHISPER_WINDOW_S:
            spans = split_long_clip(start, end, find_speech)
            if not spans and empty_fallback:
                spans = split_long_clip(start, end, None)
        else:
            pad = max(0.0, min(pad_s, (WHISPER_WINDOW_S - (end - start)) / 2))
            start, end = max(start - pad, 0.0), min(end + pad, duration_s)
            inside = [] if find_speech is None else find_speech(start, end)
            if inside:
                start, end = max(start, inside[0][0]), min(end, inside[-1][1])
            spans = [{"start": start, "end": end}] if inside or find_speech is None or empty_fallback else []
        if not spans:
            failed.append(FailedClip(clip_index=req.index, reason="no_speech"))
            continue
        usable = 0
        for span in spans:
            start, end = span["start"], span["end"]
            while seek_of(start) in used_seeks and end - start > MIN_CLIP_S:
                start += 1 / FRAMES_PER_SECOND
            if end - start < MIN_CLIP_S or seek_of(start) in used_seeks:
                continue
            used_seeks.add(seek_of(start))
            planned.append((req, {"start": start, "end": end}))
            usable += 1
        if not usable:
            failed.append(FailedClip(clip_index=req.index, reason="empty"))
    return planned, failed


@router.post("/v1/audio/transcriptions/clips", response_model=None)
async def transcribe_clips(
    request: Request,
    file: Annotated[UploadFile, File()],
    clips: Annotated[str, Form()],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    hotwords: Annotated[str | None, Form()] = None,
    pad_ms: Annotated[int | None, Form(ge=0, le=2000)] = None,
    vad: Annotated[bool | None, Form()] = None,
) -> Response:
    """Transcribe many clips of one track in GPU batches.

    Segments are tagged with `clip_index` (position in the request) and the clip's `speaker`.
    A clip of up to 30 s yields at most one segment; a longer one is split on pauses and yields
    several. A clip listed in neither `segments` nor `failed_clips` was decoded and produced
    no text.
    """
    store: ModelStore = request.app.state.models
    settings = store.settings
    model_id = model or settings.stt_model
    require_allowed_model(settings, model_id)
    requests = parse_clips(clips)
    use_vad = settings.stt_vad_filter if vad is None else vad
    pad_s = (settings.stt_clip_pad_ms if pad_ms is None else pad_ms) / 1000

    audio = decode_upload(file)
    duration = audio.size / SAMPLE_RATE
    cancel = threading.Event()
    dropped = [0]

    def run() -> tuple[list[ClipSegment], list[FailedClip], str]:
        whisper = store.whisper(model_id)
        speech = clip_vad(audio, vad_options(settings)) if use_vad else None
        planned, failed = plan_clips(requests, duration, pad_s, speech, settings.stt_vad_empty_fallback)
        if not planned:
            return [], failed, language or ""
        by_seek = {seek_of(clip["start"]): req for req, clip in planned}
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
            clip_timestamps=[clip for _, clip in planned],
            hotwords=hotwords,
            without_timestamps=True,
        )
        out: list[ClipSegment] = []
        decoded: set[int] = set()
        try:
            for seg in segments:
                if cancel.is_set():
                    raise ClientGone
                req = by_seek.get(seg.seek)
                if req is None:
                    logger.warning("Segment at seek %d matches no requested clip; dropped", seg.seek)
                    continue
                decoded.add(seg.seek)
                if settings.stt_drop_known_hallucinations and is_known_hallucination(seg.text):
                    dropped[0] += 1
                    continue
                if seg.text.strip():
                    out.append(ClipSegment(**segment_fields(seg), clip_index=req.index, speaker=req.speaker))
        except ClientGone:
            raise
        except Exception as e:  # keep what was decoded; the client retries the rest
            logger.exception("Decoding stopped after %d of %d clips", len(decoded), len(planned))
            reason = f"error: {type(e).__name__}"
            lost = {req.index for req, clip in planned if seek_of(clip["start"]) not in decoded}
            failed.extend(FailedClip(clip_index=i, reason=reason) for i in sorted(lost))
        out.sort(key=lambda s: (s.clip_index, s.start))
        failed.sort(key=lambda f: f.clip_index)
        return out, failed, info.language

    t0 = time.perf_counter()
    try:
        await acquire_slot(store.stt_semaphore, request)
    except ClientGone:
        logger.warning("Client disconnected while %s waited for a transcription slot; skipping", file.filename)
        return Response(status_code=499)
    waited = time.perf_counter() - t0
    try:
        segments, failed, detected_language = await run_cancellable(request, run, cancel)
    except ClientGone:
        logger.warning("Client disconnected during clip transcription of %s; stopped", file.filename)
        return Response(status_code=499)
    except (ValueError, FileNotFoundError, OSError) as e:
        logger.warning("Transcription with model %s failed: %s", model_id, e)
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not available: {e}") from e
    finally:
        store.stt_semaphore.release()
    elapsed = time.perf_counter() - t0
    clip_audio = sum(r.end - r.start for r in requests)
    logger.info(
        "Transcribed %s: %d clips (%.1fs of %.1fs) in %.2fs (rtf %.3f, waited %.2fs, %d segments, %d failed, "
        "%d known hallucinations dropped, model=%s)",
        file.filename,
        len(requests),
        clip_audio,
        duration,
        elapsed,
        elapsed / clip_audio if clip_audio else 0.0,
        waited,
        len(segments),
        len(failed),
        dropped[0],
        model_id,
    )
    body = ClipsTranscription(
        language=detected_language,
        duration=duration,
        clips=len(requests),
        segments=segments,
        failed_clips=failed,
    )
    return Response(content=body.model_dump_json(), media_type="application/json")
