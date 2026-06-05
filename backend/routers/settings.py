"""Settings routes — per-user LLM/Embedding overrides + global system flags."""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.auth import get_current_user, is_admin_user
from backend.config import settings
from backend.llm_provider import embedding_signature, provider_status, reset_embedding_cache
from backend.models import EmbeddingSettings, LLMSettings, STTSettings, SettingsResponse, SystemSettings
from backend.storage.user_settings import (
    load_index_meta,
    load_user_provider,
    load_user_settings,
    save_index_meta,
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
        last_reindex_at=load_index_meta(user_id).get("last_rebuild_at", ""),
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
    """Re-embed the user's resume / knowledge bases / weak-point memory with their
    current embedding model. Streams SSE progress so the UI can show a determinate bar.

    Idempotent: clears stale vectors first. Best-effort per source — a missing/empty
    corpus is skipped (status='error' for that step), not a fatal failure. Runs as a
    sync generator so blocking embed calls execute in Starlette's threadpool.

    Events: {completed, total, label, status: running|done|error[, error]} per step,
    a final {done, rebuilt, last_rebuild_at}, or {fatal, error} on setup failure."""

    def event_stream():
        from backend.indexer import (
            ingest_resume,
            ingest_topic,
            invalidate_user_embeddings,
            load_topics,
        )
        from backend.vector_memory import rebuild_index_from_profile

        try:
            topics = load_topics(user_id)  # {key: {name, dir, ...}}
            resume_dir = settings.user_resume_path(user_id)
            has_resume = resume_dir.exists() and any(p.is_file() for p in resume_dir.rglob("*"))

            plan = [("cleanup", "清理旧向量"), ("weak_points", "记忆 / 薄弱点")]
            if has_resume:
                plan.append(("resume", "简历"))
            for key, meta in topics.items():
                plan.append((f"topic:{key}", f"知识库 · {meta.get('name', key)}"))
            total = len(plan)
        except Exception as exc:  # noqa: BLE001 - setup failed, nothing rebuilt
            logger.exception("rebuild-index setup failed for user %s", user_id)
            yield f"data: {json.dumps({'fatal': True, 'error': str(exc)})}\n\n"
            return

        result = {"weak_points": False, "resume": False, "topics": []}
        done = 0

        for key, label in plan:
            yield f"data: {json.dumps({'completed': done, 'total': total, 'label': label, 'status': 'running'})}\n\n"
            try:
                if key == "cleanup":
                    invalidate_user_embeddings(user_id)
                elif key == "weak_points":
                    rebuild_index_from_profile(user_id)
                    result["weak_points"] = True
                elif key == "resume":
                    ingest_resume(user_id)
                    result["resume"] = True
                elif key.startswith("topic:"):
                    topic = key.split(":", 1)[1]
                    ingest_topic(topic, user_id)
                    result["topics"].append(topic)
            except Exception as exc:  # noqa: BLE001 - best-effort per source; skip and continue
                logger.warning("Reindex step '%s' failed for user %s: %s", key, user_id, exc)
                done += 1
                yield f"data: {json.dumps({'completed': done, 'total': total, 'label': label, 'status': 'error', 'error': str(exc)})}\n\n"
                continue
            done += 1
            yield f"data: {json.dumps({'completed': done, 'total': total, 'label': label, 'status': 'done'})}\n\n"

        now = datetime.now().isoformat(timespec="seconds")
        save_index_meta(user_id, {"last_rebuild_at": now})
        yield f"data: {json.dumps({'done': True, 'rebuilt': result, 'last_rebuild_at': now})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
