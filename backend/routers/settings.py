"""Settings routes."""

from fastapi import APIRouter, Depends

from backend.auth import get_current_user
from backend.config import settings
from backend.models import LLMSettings, SettingsResponse
from backend.storage.user_settings import load_user_settings, save_user_settings

router = APIRouter(prefix="/api")


@router.get("/settings")
def get_user_settings(user_id: str = Depends(get_current_user)):
    """Get combined LLM (global) and training (per-user) settings."""
    llm = LLMSettings(
        api_base=settings.api_base,
        api_key=settings.api_key,
        model=settings.model,
        temperature=settings.temperature,
    )
    training = load_user_settings(user_id)
    return SettingsResponse(llm=llm, training=training)


@router.put("/settings")
def put_user_settings(payload: SettingsResponse, user_id: str = Depends(get_current_user)):
    """Update LLM (global, hot-reload) and training (per-user) settings."""
    from backend.llm_provider import _reset_llama_singleton

    llm = payload.llm
    settings.api_base = llm.api_base
    settings.api_key = llm.api_key
    settings.model = llm.model
    settings.temperature = llm.temperature
    _reset_llama_singleton()

    save_user_settings(payload.training, user_id)
    return {"ok": True}
