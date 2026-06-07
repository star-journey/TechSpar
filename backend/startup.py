"""Application startup hooks."""

import logging

from backend.auth import ensure_default_user, init_users_table
from backend.models import STTSettings, SystemSettings
from backend.storage import copilot_preps as prep_store
from backend.storage.sessions import reset_stale_reviewing
from backend.storage.user_settings import apply_global_settings, load_global_settings
from backend.vector_memory import init_memory_table

logger = logging.getLogger("uvicorn")


def preload_models():
    """Initialize shared tables on startup. Provider configs are per-user and built
    lazily at request time, so no LLM/embedding client is constructed here."""
    init_memory_table()
    init_users_table()
    ensure_default_user()
    prep_store.reset_stale_running()
    recovered = reset_stale_reviewing()
    if recovered:
        logger.info("Recovered %s stuck reviewing sessions to review_failed.", recovered)
    data = load_global_settings()
    if data:
        stt = STTSettings(**data["stt"])
        apply_global_settings(stt, SystemSettings(**data["system"]))
        logger.info("Loaded persisted global settings (STT provider=%s).", stt.provider)
    logger.info("Database tables initialized.")
