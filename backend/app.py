"""FastAPI app factory."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.routers import (
    auth,
    copilot,
    data_migration,
    history,
    interview,
    knowledge,
    profile,
    public_audio,
    recording,
    resume,
    settings as settings_router,
    topics,
    voiceprint,
)
from backend.graphs.resume_interview import init_resume_checkpointer
from backend.startup import preload_models

logger = logging.getLogger("uvicorn")

_CLEANUP_INTERVAL_SECONDS = 3600


async def _periodic_public_audio_cleanup():
    from backend.public_url import cleanup_expired

    while True:
        try:
            await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
            cleanup_expired(settings.public_audio_retain_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"public audio cleanup failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_models()
    await init_resume_checkpointer()
    cleanup_task = None
    if settings.public_base_url:
        cleanup_task = asyncio.create_task(_periodic_public_audio_cleanup())
    try:
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass


def create_app() -> FastAPI:
    app = FastAPI(title="TechSpar", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(resume.router)
    app.include_router(recording.router)
    app.include_router(topics.router)
    app.include_router(profile.router)
    app.include_router(settings_router.router)
    app.include_router(voiceprint.router)
    app.include_router(interview.router)
    app.include_router(knowledge.router)
    app.include_router(history.router)
    app.include_router(data_migration.router)
    app.include_router(public_audio.router)
    app.include_router(copilot.rest_router)
    app.include_router(copilot.ws_router)
    return app
