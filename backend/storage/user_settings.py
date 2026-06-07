"""Persistence helpers for per-user training settings / provider overrides and
global (deployment-level) STT + system settings."""

import json
import os

from backend.config import settings
from backend.models import EmbeddingSettings, LLMSettings, STTSettings, SystemSettings, UserSettings


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


def load_index_meta(user_id: str) -> dict:
    """向量索引元数据,目前仅 {last_rebuild_at}。未重建过时为空 dict。"""
    path = settings.user_index_meta_path(user_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_index_meta(user_id: str, meta: dict):
    path = settings.user_index_meta_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


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


def load_global_settings() -> dict | None:
    """已落盘的全局 STT + 系统设置;从未保存过则返回 None(由 .env 提供默认值)。"""
    path = settings.global_settings_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_global_settings(stt: STTSettings, system: SystemSettings):
    """落盘全局设置。含 API key,故原子写 + chmod 0o600(对齐 save_user_provider)。"""
    path = settings.global_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"stt": stt.model_dump(), "system": system.model_dump()}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    tmp_path.replace(path)
    os.chmod(path, 0o600)


def apply_global_settings(stt: STTSettings, system: SystemSettings):
    """把全局设置回灌到 settings 单例。启动回灌与 PUT 保存共用,避免字段映射重复。"""
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
    settings.allow_registration = system.allow_registration
