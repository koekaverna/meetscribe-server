"""POST /v1/audio/diarization: speaker diarization (pyannote segmentation + WeSpeaker + VBx, all ONNX).

The audio is processed as a whole; no VAD is applied before segmentation. Output labels are
SPEAKER_00, SPEAKER_01, ... exactly as the speaches endpoint produced them.
"""

import logging
import time
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pyannote.core import Annotation

from meetscribe_server.audio import decode_upload, duration_seconds
from meetscribe_server.models import ClientGone, ModelStore, acquire_slot
from meetscribe_server.schemas import DiarizationResponse, DiarizationSegment

logger = logging.getLogger(__name__)
router = APIRouter(tags=["diarization"])


def annotation_to_segments(annotation: Annotation) -> list[DiarizationSegment]:
    """Flatten an Annotation the same way speaches did: one entry per segment, first label wins."""
    segments: list[DiarizationSegment] = []
    for segment in annotation.itersegments():
        labels = annotation.get_labels(segment)
        if not labels:
            logger.warning("No label for segment %s, assigning UNKNOWN", segment)
            speaker = "UNKNOWN"
        else:
            if len(labels) > 1:
                logger.debug("Overlapping labels for segment %s: %s, using the first", segment, labels)
            speaker = str(next(iter(labels)))
        segments.append(DiarizationSegment(start=float(segment.start), end=float(segment.end), speaker=speaker))
    return segments


@router.post("/v1/audio/diarization", response_model=DiarizationResponse)
async def diarize(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    response_format: Annotated[Literal["json", "rttm"], Form()] = "json",
    min_speakers: Annotated[int | None, Form()] = None,
    max_speakers: Annotated[int | None, Form()] = None,
) -> Response:
    store: ModelStore = request.app.state.models
    if model is not None and model != store.settings.seg_model:
        raise HTTPException(
            status_code=404,
            detail=f"Diarization model '{model}' is not available; this server serves '{store.settings.seg_model}'",
        )
    audio = decode_upload(file)
    duration = duration_seconds(audio)

    def run() -> Annotation:
        pipeline = store.diarization_pipeline()
        return pipeline(audio, file_id=file.filename, min_speakers=min_speakers, max_speakers=max_speakers)

    t0 = time.perf_counter()
    try:
        await acquire_slot(store.diarization_semaphore, request)
    except ClientGone:
        logger.warning("Client disconnected while %s waited for a diarization slot; skipping", file.filename)
        return Response(status_code=499)
    try:
        annotation = await run_in_threadpool(run)
    except ValueError as e:  # invalid min/max speakers
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        store.diarization_semaphore.release()
    elapsed = time.perf_counter() - t0

    if response_format == "rttm":
        return Response(content=annotation.to_rttm(), media_type="text/plain")

    segments = annotation_to_segments(annotation)
    speakers = {s.speaker for s in segments}
    logger.info(
        "Diarized %s: %.1fs of audio in %.2fs (rtf %.3f, %d segments, %d speakers)",
        file.filename,
        duration,
        elapsed,
        elapsed / duration if duration else 0.0,
        len(segments),
        len(speakers),
    )
    response = DiarizationResponse(duration=duration, segments=segments)
    return Response(content=response.model_dump_json(), media_type="application/json")
