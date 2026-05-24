"""Settings routes."""

from fastapi import APIRouter, Depends

from backend.auth import get_current_user
from backend.config import settings
from backend.models import LLMSettings, STTSettings, SettingsResponse
from backend.storage.user_settings import load_user_settings, save_user_settings

router = APIRouter(prefix="/api")


def _current_stt_settings() -> STTSettings:
    return STTSettings(
        provider=settings.stt_provider or "dashscope",
        dashscope_api_key=settings.dashscope_api_key,
        azure_speech_key=settings.azure_speech_key,
        azure_speech_region=settings.azure_speech_region,
        azure_speech_locales=settings.azure_speech_locales,
        soniox_api_key=settings.soniox_api_key,
        soniox_model=settings.soniox_model,
        elevenlabs_api_key=settings.elevenlabs_api_key,
        elevenlabs_model=settings.elevenlabs_model,
        qwencloud_api_key=settings.qwencloud_api_key,
    )


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
    return SettingsResponse(llm=llm, training=training, stt=_current_stt_settings())


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

    if payload.stt is not None:
        stt = payload.stt
        settings.stt_provider = stt.provider or "dashscope"
        settings.dashscope_api_key = stt.dashscope_api_key
        settings.azure_speech_key = stt.azure_speech_key
        settings.azure_speech_region = stt.azure_speech_region
        settings.azure_speech_locales = stt.azure_speech_locales
        settings.soniox_api_key = stt.soniox_api_key
        settings.soniox_model = stt.soniox_model
        settings.elevenlabs_api_key = stt.elevenlabs_api_key
        settings.elevenlabs_model = stt.elevenlabs_model
        settings.qwencloud_api_key = stt.qwencloud_api_key

    save_user_settings(payload.training, user_id)
    return {"ok": True}
