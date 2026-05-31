"""Per-user LLM and embedding providers."""

from langchain_openai import ChatOpenAI
from llama_index.llms.openai_like import OpenAILike

from backend.config import (
    embedding_api_model_of,
    embedding_local_model_of,
    embedding_local_path_of,
    embedding_mode_of,
    embedding_target_of,
    settings,
)
from backend.storage.user_settings import load_user_provider
from backend.user_context import get_current_user_id

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
        }
    return {
        "backend": override.backend,
        "api_base": override.api_base,
        "api_key": override.api_key,
        "api_model": override.api_model,
        "local_model": override.local_model,
        "local_path": override.local_path,
    }


def embedding_signature(user_id: str | None = None) -> str:
    c = resolve_embedding_config(user_id)
    return embedding_target_of(
        c["backend"], c["api_base"], c["api_key"], c["api_model"],
        c["local_model"], c["local_path"], settings.base_dir, "",
    )


def _embedding_cache_sig(c: dict) -> str:
    return "|".join(
        (c["backend"], c["api_base"], c["api_key"], c["api_model"], c["local_model"], c["local_path"])
    )


def _require_llm(c: dict):
    if not c["api_key"] or not c["model"]:
        raise ProviderNotConfigured("LLM")


def get_langchain_llm(user_id: str | None = None):
    c = resolve_llm_config(user_id)
    _require_llm(c)
    return ChatOpenAI(
        model=c["model"],
        api_key=c["api_key"],
        base_url=c["api_base"],
        temperature=c["temperature"],
        streaming=True,
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


def get_llama_llm(user_id: str | None = None):
    c = resolve_llm_config(user_id)
    _require_llm(c)
    return OpenAILike(
        model=c["model"],
        api_key=c["api_key"],
        api_base=c["api_base"],
        temperature=c["temperature"],
        is_chat_model=True,
    )


def _build_embedding(c: dict):
    deprecated = ""
    if embedding_mode_of(c["backend"], c["api_base"], c["api_key"]) == "api":
        from llama_index.embeddings.openai import OpenAIEmbedding

        if not c["api_key"]:
            raise ProviderNotConfigured("Embedding")
        model_name = embedding_api_model_of(c["api_model"], deprecated)
        kwargs = {"model_name": model_name, "api_key": c["api_key"]}
        if c["api_base"]:
            kwargs["api_base"] = c["api_base"]
        return OpenAIEmbedding(**kwargs)

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require optional dependencies. "
            "Install `pip install -r requirements.local-embedding.txt` "
            "and a torch build that matches your environment."
        ) from exc

    model_path = embedding_local_path_of(c["local_path"], c["local_model"], settings.base_dir, deprecated)
    if model_path is not None:
        return HuggingFaceEmbedding(model_name=str(model_path))
    model_name = embedding_local_model_of(c["local_model"], deprecated)
    if model_name:
        return HuggingFaceEmbedding(model_name=model_name)
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
            or embedding_local_model_of(emb["local_model"], "")
        )
    return {"llm": bool(llm["api_key"] and llm["model"]), "embedding": emb_ok}
