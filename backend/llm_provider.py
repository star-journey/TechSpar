"""Per-user LLM and embedding providers."""

import logging
import time

import httpx
import openai
from langchain_openai import ChatOpenAI

from backend.config import (
    DEFAULT_API_EMBED_BATCH_SIZE,
    embedding_api_model_of,
    embedding_local_model_of,
    embedding_local_path_of,
    embedding_mode_of,
    embedding_target_of,
    settings,
)
from backend.storage.user_settings import load_user_provider
from backend.user_context import get_current_user_id

logger = logging.getLogger("uvicorn")

_embedding_cache: dict[str, tuple[str, object]] = {}

_DEFAULT_TEMPERATURE = 0.7
_COPILOT_TEMPERATURE = 0.3


class ProviderNotConfigured(RuntimeError):
    """Raised when a user tries to use an unconfigured LLM/Embedding provider."""

    def __init__(self, what: str):
        self.what = what
        super().__init__(f"{what} provider not configured for this user")


def _effective_uid(user_id: str | None) -> str | None:
    return user_id if user_id is not None else get_current_user_id()


def resolve_llm_config(user_id: str | None = None) -> dict:
    uid = _effective_uid(user_id)
    override = load_user_provider(uid)[0] if uid else None
    if override is None:
        return {
            "api_base": settings.api_base,
            "api_key": settings.api_key,
            "model": settings.model,
            "temperature": settings.temperature,
        }
    return {
        "api_base": override.api_base,
        "api_key": override.api_key,
        "model": override.model,
        "temperature": override.temperature,
    }


def resolve_embedding_config(user_id: str | None = None) -> dict:
    uid = _effective_uid(user_id)
    override = load_user_provider(uid)[1] if uid else None
    if override is None:
        return {
            "backend": settings.embedding_backend,
            "api_base": settings.embedding_api_base,
            "api_key": settings.embedding_api_key,
            "api_model": settings.embedding_api_model,
            "local_model": settings.local_embedding_model,
            "local_path": settings.local_embedding_path,
            "api_batch_size": DEFAULT_API_EMBED_BATCH_SIZE,
        }
    return {
        "backend": override.backend,
        "api_base": override.api_base,
        "api_key": override.api_key,
        "api_model": override.api_model,
        "local_model": override.local_model,
        "local_path": override.local_path,
        "api_batch_size": override.api_batch_size,
    }


def embedding_signature(user_id: str | None = None) -> str:
    c = resolve_embedding_config(user_id)
    return embedding_target_of(
        c["backend"], c["api_base"], c["api_key"], c["api_model"],
        c["local_model"], c["local_path"], settings.base_dir, "",
    )


def _embedding_cache_sig(c: dict) -> str:
    return "|".join(
        (c["backend"], c["api_base"], c["api_key"], c["api_model"],
         c["local_model"], c["local_path"], str(c["api_batch_size"]))
    )


def _require_llm(c: dict):
    if not c["api_key"] or not c["model"]:
        raise ProviderNotConfigured("LLM")


def get_langchain_llm(user_id: str | None = None, *, streaming: bool = True,
                      timeout: float | None = None, max_retries: int | None = None):
    c = resolve_llm_config(user_id)
    _require_llm(c)
    extra: dict = {}
    if timeout is not None:
        extra["timeout"] = timeout
    if max_retries is not None:
        extra["max_retries"] = max_retries
    return ChatOpenAI(
        model=c["model"],
        api_key=c["api_key"],
        base_url=c["api_base"],
        temperature=c["temperature"],
        streaming=streaming,
        **extra,
    )


def get_copilot_llm(user_id: str | None = None, streaming: bool = False):
    c = resolve_llm_config(user_id)
    if settings.copilot_api_key and settings.copilot_model:
        c = {
            "api_base": settings.copilot_api_base,
            "api_key": settings.copilot_api_key,
            "model": settings.copilot_model,
            "temperature": c["temperature"],
        }
    _require_llm(c)
    return ChatOpenAI(
        model=c["model"],
        api_key=c["api_key"],
        base_url=c["api_base"],
        temperature=_COPILOT_TEMPERATURE,
        streaming=streaming,
    )


# Transient transport failures worth retrying. Mid-response HTTP/2 resets
# ("INTERNAL_ERROR; received from peer") and dropped connections surface as these;
# a fresh request usually succeeds. The OpenAI SDK sometimes wraps the underlying
# httpx error, so we also match on the message as a fallback.
_RETRYABLE_LLM_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ProtocolError,
)
_RETRYABLE_MESSAGE_FRAGMENTS = ("INTERNAL_ERROR", "stream error", "received from peer")


def _is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_LLM_ERRORS):
        return True
    message = str(exc)
    return any(fragment in message for fragment in _RETRYABLE_MESSAGE_FRAGMENTS)


def invoke_with_retry(llm, messages, *, attempts: int = 3, base_delay: float = 2.0):
    """Invoke an LLM, retrying transient stream/connection failures with backoff.

    Batch-JSON calls (e.g. recording analysis) can be reset mid-response by the
    provider/gateway (HTTP/2 INTERNAL_ERROR). Such failures are usually transient,
    so retry the whole call rather than failing the task. Non-transient errors
    (bad request, auth, parse) propagate immediately.
    """
    for attempt in range(1, attempts + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if not _is_retryable_llm_error(exc) or attempt == attempts:
                raise
            logger.warning(
                "LLM invoke transient failure (attempt %d/%d): %s", attempt, attempts, exc
            )
            time.sleep(base_delay * attempt)


# ── Embedding ──

class _APIEmbedding:
    """OpenAI-compatible embedding client. Exposes the minimal interface the rest of
    the codebase relies on (get_text_embedding / get_text_embedding_batch). Batches to
    `batch_size` to respect per-request limits, which vary by provider."""

    def __init__(self, model: str, api_key: str, api_base: str, batch_size: int):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=api_base or None)
        self._model = model
        self._batch = max(1, batch_size)

    def get_text_embedding(self, text: str) -> list[float]:
        return self.get_text_embedding_batch([text])[0]

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            resp = self._client.embeddings.create(model=self._model, input=texts[i:i + self._batch])
            out.extend(d.embedding for d in resp.data)
        return out


class _LocalEmbedding:
    """Local sentence-transformers embedding with the same minimal interface."""

    def __init__(self, model_name_or_path: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name_or_path)

    def get_text_embedding(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()


def _build_embedding(c: dict):
    deprecated = ""
    if embedding_mode_of(c["backend"], c["api_base"], c["api_key"]) == "api":
        if not c["api_key"]:
            raise ProviderNotConfigured("Embedding")
        model_name = embedding_api_model_of(c["api_model"], deprecated)
        if not model_name:
            raise RuntimeError("EMBEDDING_API_MODEL is required when EMBEDDING_BACKEND=api")
        return _APIEmbedding(
            model=model_name,
            api_key=c["api_key"],
            api_base=c["api_base"],
            batch_size=c["api_batch_size"],
        )

    # mode == "local". An empty backend means nothing was configured (the implicit
    # fallback), which in this bring-your-own-key deployment is a misconfiguration
    # rather than a real local setup — surface it as a handled "not configured" error.
    if not c["backend"]:
        raise ProviderNotConfigured("Embedding")

    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require optional dependencies. "
            "Install `pip install -r requirements.local-embedding.txt` "
            "and a torch build that matches your environment."
        ) from exc

    model_path = embedding_local_path_of(c["local_path"], c["local_model"], settings.base_dir, deprecated)
    if model_path is not None:
        return _LocalEmbedding(str(model_path))
    model_name = embedding_local_model_of(c["local_model"], deprecated)
    if model_name:
        return _LocalEmbedding(model_name)
    raise RuntimeError(
        "LOCAL_EMBEDDING_MODEL or LOCAL_EMBEDDING_PATH is required when EMBEDDING_BACKEND=local"
    )


def get_embedding(user_id: str | None = None):
    c = resolve_embedding_config(user_id)
    sig = _embedding_cache_sig(c)
    key = _effective_uid(user_id) or "__global__"
    cached = _embedding_cache.get(key)
    if cached and cached[0] == sig:
        return cached[1]
    inst = _build_embedding(c)
    _embedding_cache[key] = (sig, inst)
    return inst


def batched_embed(texts: list[str], user_id: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    embed_model = get_embedding(user_id)
    batch_size = max(1, settings.openai_embedding_max_batch_size)
    if len(texts) <= batch_size:
        return embed_model.get_text_embedding_batch(texts)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embed_model.get_text_embedding_batch(texts[start : start + batch_size]))
    return vectors


def reset_embedding_cache(user_id: str | None = None):
    if user_id is None:
        _embedding_cache.clear()
    else:
        _embedding_cache.pop(user_id, None)


# ── Connectivity probes (test the *provided* config, not the saved one) ──

def probe_llm(api_base: str, api_key: str, model: str) -> None:
    """Verify an LLM config is reachable & valid by issuing a 1-token chat
    completion. Returns None on success; raises (openai errors / ProviderNotConfigured)
    on any failure. Drives the settings 'test connection' button and the onboarding
    gate, so it tests the form values rather than what's persisted."""
    from openai import OpenAI

    if not api_key or not model:
        raise ProviderNotConfigured("LLM")
    client = OpenAI(api_key=api_key, base_url=api_base or None, timeout=20.0, max_retries=0)
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
        temperature=0,
    )


def probe_embedding(config: dict) -> None:
    """Verify an embedding config by embedding a probe string. `config` matches
    EmbeddingSettings (backend/api_base/api_key/api_model/local_model/local_path/...).
    Returns None on success; raises on failure. For local mode, loading the model is
    itself the test."""
    if embedding_mode_of(config["backend"], config["api_base"], config["api_key"]) == "api":
        if not config["api_key"]:
            raise ProviderNotConfigured("Embedding")
        model_name = embedding_api_model_of(config["api_model"], "")
        if not model_name:
            raise RuntimeError("Embedding Model 必填")
        from openai import OpenAI

        client = OpenAI(api_key=config["api_key"], base_url=config["api_base"] or None,
                        timeout=20.0, max_retries=0)
        client.embeddings.create(model=model_name, input=["ping"])
        return
    _build_embedding(config).get_text_embedding("ping")


# ── Optional service credentials (per-user, no global fallback) ──

def resolve_dashscope_key(user_id: str | None = None) -> str:
    """DashScope key for ASR (语音输入 / 录音转写 / Copilot 实时)。未配置返回空串。"""
    uid = _effective_uid(user_id)
    return load_user_services(uid).dashscope_api_key if uid else ""


def resolve_tavily_key(user_id: str | None = None) -> str:
    """Tavily key for Copilot 联网搜索。未配置返回空串。"""
    uid = _effective_uid(user_id)
    return load_user_services(uid).tavily_api_key if uid else ""


def resolve_oss_config(user_id: str | None = None) -> dict:
    """阿里云 OSS 配置(录音复盘长音频上传)。未配置字段为空串。"""
    uid = _effective_uid(user_id)
    s = load_user_services(uid) if uid else None
    return {
        "access_key_id": s.oss_access_key_id if s else "",
        "access_key_secret": s.oss_access_key_secret if s else "",
        "bucket": s.oss_bucket if s else "",
        "endpoint": s.oss_endpoint if s else "",
    }


def provider_status(user_id: str | None = None) -> dict:
    llm = resolve_llm_config(user_id)
    emb = resolve_embedding_config(user_id)
    if embedding_mode_of(emb["backend"], emb["api_base"], emb["api_key"]) == "api":
        emb_ok = bool(emb["api_key"])
    else:
        emb_ok = bool(
            emb["local_model"]
            or emb["local_path"]
            or embedding_local_path_of(emb["local_path"], emb["local_model"], settings.base_dir, "")
        )
    return {"llm": bool(llm["api_key"] and llm["model"]), "embedding": emb_ok}
