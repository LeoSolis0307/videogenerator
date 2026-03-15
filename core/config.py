import os
import json
from typing import Any, Set
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Ollama Settings
    ollama_url: str = Field(default="http://localhost:11434/api/generate", validation_alias="OLLAMA_URL")
    ollama_text_model: str = Field(default="", validation_alias="OLLAMA_TEXT_MODEL")
    ollama_text_model_short: str = Field(default="qwen2.5:7b", validation_alias="OLLAMA_TEXT_MODEL_SHORT")
    ollama_text_model_long: str = Field(default="qwen2.5:32b", validation_alias="OLLAMA_TEXT_MODEL_LONG")
    ollama_timeout: int = Field(default=90, validation_alias="OLLAMA_TIMEOUT")
    ollama_keep_alive: int = Field(default=900, validation_alias="OLLAMA_KEEP_ALIVE")
    ollama_options_json: str = Field(default="", validation_alias="OLLAMA_OPTIONS_JSON")
    ollama_text_num_ctx: int = Field(default=2048, validation_alias="OLLAMA_TEXT_NUM_CTX")
    unload_text_model: bool = Field(default=True, validation_alias="UNLOAD_TEXT_MODEL")

    # Vision Settings
    vision_model: str = Field(default="llama3.2-vision", validation_alias="VISION_MODEL")
    vision_timeout_sec: int = Field(default=90, validation_alias="VISION_TIMEOUT_SEC")
    vision_retries: int = Field(default=3, validation_alias="VISION_RETRIES")

    # Video Generation Settings
    video_quality: str | None = Field(default=None, validation_alias="VIDEO_QUALITY")
    custom_min_video_sec: int = Field(default=40, validation_alias="CUSTOM_MIN_VIDEO_SEC")
    long_video_min_seconds: int = Field(default=300, validation_alias="LONG_VIDEO_MIN_SECONDS_FOR_VIDEOS_DIR")

    long_videos_dir: str = Field(default="videos", validation_alias="LONG_VIDEOS_DIR")

    # Video Encoding / Rendering
    video_fit_mode: str = Field(default="pad", validation_alias="VIDEO_FIT_MODE")
    video_pad_style: str = Field(default="blur", validation_alias="VIDEO_PAD_STYLE")
    video_kenburns: bool = Field(default=False, validation_alias="VIDEO_KENBURNS")
    video_kb_rate: float = Field(default=0.0009, validation_alias="VIDEO_KB_RATE")
    video_kb_max: float = Field(default=1.08, validation_alias="VIDEO_KB_MAX")
    video_crossfade_ms: int = Field(default=250, validation_alias="VIDEO_CROSSFADE_MS")
    enable_loudnorm: bool = Field(default=True, validation_alias="ENABLE_LOUDNORM")
    video_blur: str = Field(default="10:1", validation_alias="VIDEO_BLUR")
    
    video_preset: str | None = Field(default=None, validation_alias="VIDEO_PRESET")
    video_crf: str | None = Field(default=None, validation_alias="VIDEO_CRF")
    video_codec: str = Field(default="h264_amf", validation_alias="VIDEO_CODEC")
    video_codec_args: str = Field(default="", validation_alias="VIDEO_CODEC_ARGS")
    require_gpu: bool = Field(default=True, validation_alias="REQUIRE_GPU")
    video_fps: str | None = Field(default=None, validation_alias="VIDEO_FPS")
    audio_bitrate: str | None = Field(default=None, validation_alias="AUDIO_BITRATE")
    video_scale_flags: str | None = Field(default=None, validation_alias="VIDEO_SCALE_FLAGS")
    video_tune: str | None = Field(default=None, validation_alias="VIDEO_TUNE")

    # Image Settings
    comfyui_enabled: bool = Field(default=False, validation_alias="COMFYUI_ENABLED")
    custom_img_quality: str = Field(default="high", validation_alias="CUSTOM_IMG_QUALITY")
    custom_min_img_score: int = Field(default=1, validation_alias="CUSTOM_MIN_IMG_SCORE")
    custom_img_max_per_query: int = Field(default=8, validation_alias="CUSTOM_IMG_MAX_PER_QUERY")
    custom_hook_segments: int = Field(default=2, validation_alias="CUSTOM_HOOK_SEGMENTS")
    custom_hook_extra_candidates: int = Field(default=10, validation_alias="CUSTOM_HOOK_EXTRA_CANDIDATES")
    custom_hook_min_img_score: int = Field(default=3, validation_alias="CUSTOM_HOOK_MIN_IMG_SCORE")
    custom_allow_placeholder_images: bool = Field(default=False, validation_alias="CUSTOM_ALLOW_PLACEHOLDER_IMAGES")
    custom_comfy_fallback_on_missing_images: bool = Field(default=True, validation_alias="CUSTOM_COMFY_FALLBACK_ON_MISSING_IMAGES")
    custom_voice: str = Field(default="es-MX-JorgeNeural", validation_alias="CUSTOM_VOICE")
    
    # Image Sources & Network
    img_sources: str = Field(default="bing,flickr,wikimedia,openverse", validation_alias="IMG_SOURCES")
    ddg_images_backend: str = Field(default="lite", validation_alias="DDG_IMAGES_BACKEND")
    ddg_search_timeout_sec: float = Field(default=25.0, validation_alias="DDG_SEARCH_TIMEOUT_SEC")
    openverse_timeout_sec: float = Field(default=20.0, validation_alias="OPENVERSE_TIMEOUT_SEC")
    img_antibot_proxy: bool = Field(default=True, validation_alias="IMG_ANTIBOT_PROXY")
    img_antibot_browser: bool = Field(default=False, validation_alias="IMG_ANTIBOT_BROWSER")
    allow_avif: bool = Field(default=False, validation_alias="ALLOW_AVIF")
    wikimedia_user_agent: str = Field(default="", validation_alias="WIKIMEDIA_USER_AGENT")
    
    # Text Processing
    enable_text_rank: bool = Field(default=True, validation_alias="CUSTOM_IMG_TEXT_RANK")
    
    # Safety
    img_blocked_hosts: str = Field(default="", validation_alias="IMG_BLOCKED_HOSTS")

    # Reddit Importer
    reddit_429_soft_limit: int = Field(default=6, validation_alias="REDDIT_429_SOFT_LIMIT")
    reddit_429_hard_limit: int = Field(default=22, validation_alias="REDDIT_429_HARD_LIMIT")
    reddit_request_pause_s: float = Field(default=0.35, validation_alias="REDDIT_REQUEST_PAUSE_S")
    reddit_timeout_sec: float = Field(default=20.0, validation_alias="REDDIT_TIMEOUT_SEC")
    reddit_request_retries: int = Field(default=2, validation_alias="REDDIT_REQUEST_RETRIES")
    reddit_retry_backoff_s: float = Field(default=0.8, validation_alias="REDDIT_RETRY_BACKOFF_S")
    reddit_max_cooldown_wait_s: float = Field(default=12.0, validation_alias="REDDIT_MAX_COOLDOWN_WAIT_S")
    reddit_import_subreddits: str = Field(default="", validation_alias="REDDIT_IMPORT_SUBREDDITS")
    reddit_feed_profile: str = Field(default="light", validation_alias="REDDIT_FEED_PROFILE")
    reddit_fallback_wait_s: float = Field(default=6.0, validation_alias="REDDIT_FALLBACK_WAIT_S")
    reddit_client_id: str = Field(default="", validation_alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", validation_alias="REDDIT_CLIENT_SECRET")
    reddit_state_retention_days: int = Field(default=10, validation_alias="REDDIT_STATE_RETENTION_DAYS")
    reddit_state_max_post_ids: int = Field(default=12000, validation_alias="REDDIT_STATE_MAX_POST_IDS")
    reddit_state_max_comment_ids: int = Field(default=60000, validation_alias="REDDIT_STATE_MAX_COMMENT_IDS")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @staticmethod
    def _split_csv(raw: str) -> list[str]:
        parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
        out: list[str] = []
        seen: set[str] = set()
        for part in parts:
            p = part.lower()
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    @property
    def ollama_options(self) -> dict[str, Any]:
        raw = (self.ollama_options_json or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    @property
    def img_sources_list(self) -> list[str]:
        return self._split_csv(self.img_sources)

    @property
    def blocked_hosts_set(self) -> Set[str]:
        default_blocked = {
            "freepik.com", "img.freepik.com", "pinterest.com", "i.pinimg.com",
            "researchgate.net", "rgstatic.net"
        }
        if not self.img_blocked_hosts.strip():
            return default_blocked
        
        custom = {h.strip().lower() for h in self.img_blocked_hosts.split(",") if h.strip()}
        return custom.union(default_blocked)

settings = Settings()
