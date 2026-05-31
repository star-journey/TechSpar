"""Persistence helpers for per-user training settings and provider overrides."""

import json
import os

from backend.config import settings
from backend.models import EmbeddingSettings, LLMSettings, UserSettings


def load_user_provider(user_id: str) -> tuple[LLMSettings | None, EmbeddingSettings | None]:
    path = settings.user_provider_path(user_id)
    if not path.exists():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    llm = LLMSettings(**data["llm"]) if data.get("llm") else None
    embedding = EmbeddingSettings(**data["embedding"]) if data.get("embedding") else None
    return llm, embedding


def save_user_provider(user_id: str, llm: LLMSettings, embedding: EmbeddingSettings):
    path = settings.user_provider_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"llm": llm.model_dump(), "embedding": embedding.model_dump()}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    tmp_path.replace(path)
    os.chmod(path, 0o600)


def load_user_settings(user_id: str) -> UserSettings:
    path = settings.user_settings_path(user_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return UserSettings(**data)
    return UserSettings()


def save_user_settings(user_settings: UserSettings, user_id: str):
    path = settings.user_settings_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(user_settings.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
