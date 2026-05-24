from pathlib import Path
from pydantic_settings import BaseSettings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


class Settings(BaseSettings):
    # LLM (OpenAI-compatible proxy)
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.7

    # Embedding — explicit backend + separate config for API/local modes
    embedding_backend: str = ""  # api | local; empty keeps legacy inference
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_api_model: str = ""
    local_embedding_model: str = ""
    local_embedding_path: str = ""
    embedding_model: str = ""  # deprecated fallback for EMBEDDING_MODEL
    openai_embedding_max_batch_size: int = 64  # max texts per API call

    # STT (speech-to-text) provider selection
    # 可选值: dashscope | azure | soniox | elevenlabs | qwencloud
    stt_provider: str = "dashscope"

    # DashScope ASR (speech-to-text, batch transcription)
    dashscope_api_key: str = ""

    # Azure Speech (Fast Transcription)
    azure_speech_key: str = ""
    azure_speech_region: str = ""  # 例 "eastus"，或直接填资源域名
    azure_speech_locales: str = "zh-CN,en-US"  # 逗号分隔

    # Soniox (async transcription)
    soniox_api_key: str = ""
    soniox_model: str = "stt-async-v4"

    # ElevenLabs Speech-to-Text
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "scribe_v2"

    # QwenCloud (DashScope International) — host = dashscope-intl.aliyuncs.com
    qwencloud_api_key: str = ""

    # Copilot — 独立 LLM 配置（可选，不填则 fallback 到主 LLM）
    copilot_api_base: str = ""
    copilot_api_key: str = ""
    copilot_model: str = ""
    copilot_temperature: float = 0.3  # Copilot 场景偏确定性

    # Copilot — 腾讯云 VPR 声纹识别（可选，未配置时自动回退手动按钮模式）
    # 允许在用户 settings.json 中覆盖，此处为全局兜底
    tencent_secret_id: str = ""
    tencent_secret_key: str = ""
    tencent_vpr_app_id: str = ""

    # Copilot — Tavily Web Search
    tavily_api_key: str = ""

    # Alibaba Cloud OSS (only long-audio filetrans needs a public URL;
    # short audio goes through base64 sync chat/completions, no OSS required).
    aliyun_oss_access_key_id: str = ""
    aliyun_oss_access_key_secret: str = ""
    aliyun_oss_bucket: str = ""
    aliyun_oss_endpoint: str = ""  # e.g. "oss-cn-shanghai.aliyuncs.com"

    # Self-hosted public URL — 配置后替代 OSS 为 DashScope filetrans 提供拉取 URL。
    # 形如 "https://your.domain.com"，不带尾斜杠。
    public_base_url: str = ""
    public_url_secret: str = ""  # HMAC 密钥；为空则复用 jwt_secret
    public_audio_ttl_seconds: int = 3600  # 签名 URL 有效期
    public_audio_retain_seconds: int = 86400  # 磁盘文件保留时长

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    resume_path: Path = Path(__file__).resolve().parent.parent / "data" / "resume"
    knowledge_path: Path = Path(__file__).resolve().parent.parent / "data" / "knowledge"
    high_freq_path: Path = Path(__file__).resolve().parent.parent / "data" / "high_freq"
    db_path: Path = Path(__file__).resolve().parent.parent / "data" / "interviews.db"

    # Auth
    jwt_secret: str = "change-me-in-production"
    default_email: str = "admin@techspar.local"
    default_password: str = "admin123"
    default_name: str = "Admin"
    allow_registration: bool = False

    # Interview settings
    max_questions_per_phase: int = 5
    max_drill_questions: int = 15

    def user_data_dir(self, user_id: str) -> Path:
        return self.base_dir / "data" / "users" / user_id

    def user_profile_dir(self, user_id: str) -> Path:
        return self.user_data_dir(user_id) / "profile"

    def user_resume_path(self, user_id: str) -> Path:
        return self.user_data_dir(user_id) / "resume"

    def user_knowledge_path(self, user_id: str) -> Path:
        return self.user_data_dir(user_id) / "knowledge"

    def user_high_freq_path(self, user_id: str) -> Path:
        return self.user_data_dir(user_id) / "high_freq"

    def user_topics_path(self, user_id: str) -> Path:
        return self.user_data_dir(user_id) / "topics.json"

    def user_index_cache_path(self, user_id: str) -> Path:
        return self.user_data_dir(user_id) / ".index_cache"

    def user_settings_path(self, user_id: str) -> Path:
        return self.user_data_dir(user_id) / "settings.json"

    def public_audio_dir(self) -> Path:
        path = self.base_dir / "data" / "public_audio"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def effective_public_url_secret(self) -> str:
        return self.public_url_secret or self.jwt_secret

    @property
    def effective_dashscope_api_key(self) -> str:
        """DashScope API key, with fallback to COPILOT_API_KEY when the Copilot
        LLM is already pointed at DashScope's OpenAI-compatible endpoint.

        Lets users reuse a single DashScope account key across LLM + ASR
        without forcing them to duplicate it into two env vars.
        """
        if self.dashscope_api_key:
            return self.dashscope_api_key
        if self.copilot_api_key and "dashscope.aliyuncs.com" in (self.copilot_api_base or ""):
            return self.copilot_api_key
        return ""

    def embedding_backend_mode(self) -> str:
        if self.embedding_backend:
            backend = self.embedding_backend.strip().lower()
            if backend in {"api", "local"}:
                return backend
            raise ValueError("EMBEDDING_BACKEND must be 'api' or 'local'")
        if self.embedding_api_base or self.embedding_api_key:
            return "api"
        return "local"

    def embedding_api_model_name(self) -> str:
        return self.embedding_api_model or self.embedding_model or DEFAULT_EMBEDDING_MODEL

    def local_embedding_model_name(self) -> str:
        return self.local_embedding_model or self.embedding_model or DEFAULT_EMBEDDING_MODEL

    def local_embedding_model_path(self) -> Path | None:
        if self.local_embedding_path:
            return Path(self.local_embedding_path).expanduser()

        bundled_path = self.base_dir / "data" / "models" / "bge-m3"
        if self.local_embedding_model_name() == DEFAULT_EMBEDDING_MODEL and bundled_path.exists():
            return bundled_path
        return None

    def active_embedding_target(self) -> str:
        if self.embedding_backend_mode() == "api":
            return self.embedding_api_model_name()

        model_path = self.local_embedding_model_path()
        if model_path is not None:
            return str(model_path)
        return self.local_embedding_model_name()

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
