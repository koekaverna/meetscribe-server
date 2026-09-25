"""POST /v1/audio/speech/embedding: 256-d WeSpeaker speaker embedding of a whole clip."""

import logging
import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from meetscribe_server.audio import decode_upload, duration_seconds
from meetscribe_server.models import ModelStore
from meetscribe_server.schemas import CreateEmbeddingResponse, EmbeddingObject

logger = logging.getLogger(__name__)
router = APIRouter(tags=["speaker-embedding"])


@router.post("/v1/audio/speech/embedding", response_model=CreateEmbeddingResponse)
async def speaker_embedding(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
) -> CreateEmbeddingResponse:
    store: ModelStore = request.app.state.models
    model_id = store.settings.emb_model
    if model is not None and model != model_id:
        raise HTTPException(
            status_code=404,
            detail=f"Embedding model '{model}' is not available; this server serves '{model_id}'",
        )
    audio = decode_upload(file)

    def run() -> np.ndarray:
        with store.embedding_semaphore:
            emb_model = store.embedding_model()
            fbank_data = emb_model.preprocess(audio)
            embedding = np.asarray(emb_model.extract(fbank_data))
            if embedding.ndim == 2:
                embedding = embedding.squeeze(0) if embedding.shape[0] == 1 else embedding.mean(axis=0)
            return embedding.astype(np.float32)

    t0 = time.perf_counter()
    embedding = await run_in_threadpool(run)
    logger.info(
        "Embedded %s: %.1fs of audio in %.2fs (dim %d)",
        file.filename,
        duration_seconds(audio),
        time.perf_counter() - t0,
        embedding.size,
    )
    return CreateEmbeddingResponse(
        data=[EmbeddingObject(embedding=[float(x) for x in embedding], index=0)],
        model=model_id,
    )
