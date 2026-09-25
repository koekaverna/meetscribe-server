import logging

import numpy as np
import numpy.typing as npt
from fastapi import HTTPException, UploadFile
from faster_whisper.audio import decode_audio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def decode_upload(file: UploadFile) -> npt.NDArray[np.float32]:
    """Decode an uploaded audio file into mono float32 PCM at 16 kHz."""
    try:
        data = decode_audio(file.file, sampling_rate=SAMPLE_RATE)
    except Exception as e:  # PyAV raises a zoo of error types
        logger.warning("Failed to decode upload %r: %s", file.filename, e)
        raise HTTPException(status_code=400, detail=f"Could not decode audio file: {e}") from e
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 1:
        data = data.reshape(-1)
    if data.size == 0:
        raise HTTPException(status_code=400, detail="Audio file contains no samples")
    logger.debug("Decoded %r: %.2fs", file.filename, data.size / SAMPLE_RATE)
    return data


def duration_seconds(data: npt.NDArray[np.float32]) -> float:
    return float(data.size) / SAMPLE_RATE
