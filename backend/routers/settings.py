"""Settings routes — per-user LLM/Embedding overrides + global system flags."""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.auth import get_current_user, is_admin_user
from backend.config import settings
from backend.llm_provider import (
    ProviderNotConfigured,
    embedding_signature,
    probe_embedding,
    probe_llm,
    provider_status,
    reset_embedding_cache,
)
from backend.models import EmbeddingSettings, LLMSettings, STTSettings, SettingsResponse, SystemSettings
from backend.storage.user_settings import (
    apply_global_settings,
    load_index_meta,
    load_user_provider,
    load_user_settings,
    save_global_settings,
    save_index_meta,
    save_user_provider,
    save_user_settings,
)
from backend.storage.system_settings import save_system_settings

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
        # 管理员保存:先落盘系统标志(注册开关跨重启保留),再回灌 STT 全局配置。
        save_system_settings(payload.system)
        if payload.stt is not None:
            apply_global_settings(payload.stt, payload.system)
        else:
            settings.allow_registration = payload.system.allow_registration
        # 始终落盘当前完整全局状态(从 settings 重建),跨进程重启/应用更新保留。
        save_global_settings(
            _stt_settings(include_secrets=True),
            SystemSettings(allow_registration=settings.allow_registration),
        )

    save_user_settings(payload.training, user_id)
    return {"ok": True, "embedding_changed": embedding_changed}


def _conn_error_message(exc: Exception) -> str:
    """Map a probe exception to a concise Chinese hint for the test UI."""
    import openai

    if isinstance(exc, ProviderNotConfigured):
        return "请先填写必填字段"
    if isinstance(exc, openai.AuthenticationError):
        return "API Key 无效（认证失败）"
    if isinstance(exc, openai.PermissionDeniedError):
        return "Key 无该模型权限或被拒绝访问"
    if isinstance(exc, openai.NotFoundError):
        return "模型不存在，或 Base URL 路径不正确"
    if isinstance(exc, openai.APIConnectionError):
        return "无法连接到 Base URL，请检查地址与网络"
    msg = str(exc).strip().replace("\n", " ")
    return msg[:300] or exc.__class__.__name__


@router.post("/settings/test-llm")
def test_llm_connection(payload: LLMSettings, user_id: str = Depends(get_current_user)):
    """Probe the provided LLM config with a 1-token request. Returns {ok[, error]}
    so the UI can show inline status and the onboarding gate can block on failure.
    Tests the form values, not the saved config — works before first save."""
    try:
        probe_llm(payload.api_base, payload.api_key, payload.model)
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001 - any failure means 'not reachable'
        return {"ok": False, "error": _conn_error_message(exc)}


@router.post("/settings/test-embedding")
def test_embedding_connection(payload: EmbeddingSettings, user_id: str = Depends(get_current_user)):
    """Probe the provided embedding config by embedding a tiny string."""
    try:
        probe_embedding(payload.model_dump())
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001 - any failure means 'not reachable'
        return {"ok": False, "error": _conn_error_message(exc)}


@router.post("/settings/rebuild-index")
def rebuild_index(user_id: str = Depends(get_current_user)):
    """Re-embed the user's personal documents / knowledge bases / weak-point memory with their
    current embedding model. Streams SSE progress so the UI can show a determinate bar.

    Idempotent: clears stale vectors first. Best-effort per source — a missing/empty
    corpus is skipped (status='error' for that step), not a fatal failure. Runs as a
    sync generator so blocking embed calls execute in Starlette's threadpool.

    Events: {completed, total, label, status: running|done|error[, error]} per step,
    a final {done, rebuilt, last_rebuild_at}, or {fatal, error} on setup failure."""

    def event_stream():
        from backend.indexer import (
            ingest_topic,
            invalidate_user_embeddings,
            load_topics,
        )
        from backend.vector_memory import rebuild_index_from_profile
        from backend.personal_agent import list_documents, reindex_all_documents

        try:
            topics = load_topics(user_id)  # {key: {name, dir, ...}}

            plan = [("cleanup", "清理旧向量"), ("weak_points", "记忆 / 薄弱点")]
            if list_documents(user_id):
                plan.append(("personal_documents", "个人资料库"))
            for key, meta in topics.items():
                plan.append((f"topic:{key}", f"知识库 · {meta.get('name', key)}"))
            total = len(plan)
        except Exception as exc:  # noqa: BLE001 - setup failed, nothing rebuilt
            logger.exception("rebuild-index setup failed for user %s", user_id)
            yield f"data: {json.dumps({'fatal': True, 'error': str(exc)})}\n\n"
            return

        result = {"weak_points": False, "personal_documents": False, "topics": []}
        done = 0

        for key, label in plan:
            yield f"data: {json.dumps({'completed': done, 'total': total, 'label': label, 'status': 'running'})}\n\n"
            try:
                if key == "cleanup":
                    invalidate_user_embeddings(user_id)
                elif key == "weak_points":
                    rebuild_index_from_profile(user_id)
                    result["weak_points"] = True
                elif key == "personal_documents":
                    reindex_all_documents(user_id)
                    result["personal_documents"] = True
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
