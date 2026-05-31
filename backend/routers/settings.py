"""Settings routes — per-user LLM/Embedding overrides + global system flags."""

import logging

from fastapi import APIRouter, Depends

from backend.auth import get_current_user, is_admin_user
from backend.config import settings
from backend.llm_provider import embedding_signature, provider_status, reset_embedding_cache
from backend.models import EmbeddingSettings, LLMSettings, STTSettings, SettingsResponse, SystemSettings
from backend.storage.user_settings import (
    load_user_provider,
    load_user_settings,
    save_user_provider,
    save_user_settings,
)

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/api")


def _stt_settings(include_secrets: bool) -> STTSettings:
    return STTSettings(
        provider=settings.stt_provider or "dashscope",
        dashscope_api_key=settings.dashscope_api_key if include_secrets else "",
        azure_speech_key=settings.azure_speech_key if include_secrets else "",
        azure_speech_region=settings.azure_speech_region,
        azure_speech_locales=settings.azure_speech_locales,
        soniox_api_key=settings.soniox_api_key if include_secrets else "",
        soniox_model=settings.soniox_model,
        elevenlabs_api_key=settings.elevenlabs_api_key if include_secrets else "",
        elevenlabs_model=settings.elevenlabs_model,
        qwencloud_api_key=settings.qwencloud_api_key if include_secrets else "",
    )


@router.get("/settings")
def get_user_settings(user_id: str = Depends(get_current_user)):
    llm_override, emb_override = load_user_provider(user_id)
    llm = llm_override or LLMSettings()
    embedding = emb_override or EmbeddingSettings()
    system = SystemSettings(allow_registration=settings.allow_registration)
    training = load_user_settings(user_id)
    return SettingsResponse(
        llm=llm,
        embedding=embedding,
        system=system,
        training=training,
        stt=_stt_settings(include_secrets=is_admin_user(user_id)),
        is_admin=is_admin_user(user_id),
        configured=provider_status(user_id),
    )


@router.put("/settings")
def put_user_settings(payload: SettingsResponse, user_id: str = Depends(get_current_user)):
    old_emb_sig = embedding_signature(user_id)
    llm = payload.llm
    embedding = payload.embedding

    save_user_provider(user_id, llm, embedding)
    reset_embedding_cache(user_id)

    embedding_changed = embedding_signature(user_id) != old_emb_sig
    if embedding_changed:
        from backend.indexer import invalidate_user_embeddings

        logger.info("Embedding model changed for user %s — vectors invalidated.", user_id)
        invalidate_user_embeddings(user_id)

    if is_admin_user(user_id):
        settings.allow_registration = payload.system.allow_registration
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
    return {"ok": True, "embedding_changed": embedding_changed}


@router.post("/settings/rebuild-index")
def rebuild_index(user_id: str = Depends(get_current_user)):
    from backend.indexer import (
        build_resume_index,
        build_topic_index,
        get_topic_map,
        invalidate_user_embeddings,
    )
    from backend.vector_memory import rebuild_index_from_profile

    invalidate_user_embeddings(user_id)
    rebuild_index_from_profile(user_id)

    result = {"weak_points": True, "resume": False, "topics": []}

    resume_dir = settings.user_resume_path(user_id)
    if resume_dir.exists() and any(p.is_file() for p in resume_dir.rglob("*")):
        try:
            build_resume_index(user_id, force_rebuild=True)
            result["resume"] = True
        except Exception as exc:
            logger.warning("Resume reindex failed for user %s: %s", user_id, exc)

    for topic in get_topic_map(user_id):
        try:
            build_topic_index(topic, user_id, force_rebuild=True)
            result["topics"].append(topic)
        except Exception as exc:
            logger.info("Topic '%s' reindex skipped for user %s: %s", topic, user_id, exc)

    return {"ok": True, "rebuilt": result}
