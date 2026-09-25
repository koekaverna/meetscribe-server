"""ASGI app factory. Run with: uvicorn --factory meetscribe_server.main:create_app"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from meetscribe_server import __version__
from meetscribe_server.config import Settings
from meetscribe_server.models import ModelStore
from meetscribe_server.routers import diarization, embedding, stt
from meetscribe_server.schemas import ModelInfo, ModelList

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def create_app(settings: Settings | None = None, store: ModelStore | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)
    store = store or ModelStore(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.preload:
            logger.info("Preloading models")
            await run_in_threadpool(store.load_all)
            logger.info("Models ready: %s", store.loaded())
        else:
            logger.info("PRELOAD=false: models load on first request")
        yield

    app = FastAPI(title="meetscribe-server", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.models = store

    app.include_router(stt.router)
    app.include_router(diarization.router)
    app.include_router(embedding.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"message": "OK"}

    @app.get("/v1/models", response_model=ModelList)
    def list_models() -> ModelList:
        loaded = store.loaded()
        entries = (
            (settings.stt_model, "automatic-speech-recognition"),
            (settings.seg_model, "speaker-segmentation"),
            (settings.emb_model, "speaker-embedding"),
        )
        return ModelList(
            data=[
                ModelInfo(id=model_id, owned_by=model_id.split("/")[0], task=task, loaded=loaded.get(model_id, False))
                for model_id, task in entries
            ]
        )

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    return app
