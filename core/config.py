import os
from typing import Set
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Ollama Settings
    ollama_url: str = Field(default="http://localhost:11434/api/generate", validation_alias="OLLAMA_URL")
    ollama_text_model: str = Field(default="", validation_alias="OLLAMA_TEXT_MODEL")
    ollama_text_model_short: str = Field(default="qwen2.5:32b", validation_alias="OLLAMA_TEXT_MODEL_SHORT")
    ollama_text_model_long: str = Field(default="qwen2.5:32b", validation_alias="OLLAMA_TEXT_MODEL_LONG")
    ollama_timeout: int = Field(default=90, validation_alias="OLLAMA_TIMEOUT")
    ollama_options_json: str = Field(default="", validation_alias="OLLAMA_OPTIONS_JSON")
    ollama_text_num_ctx: int = Field(default=2048, validation_alias="OLLAMA_TEXT_NUM_CTX")
    unload_text_model: bool = Field(default=True, validation_alias="UNLOAD_TEXT_MODEL")

    # Vision Settings
    vision_model: str = Field(default="llama3.2-vision", validation_alias="VISION_MODEL")
    vision_timeout_sec: int = Field(default=90, validation_alias="VISION_TIMEOUT_SEC")
    vision_retries: int = Field(default=3, validation_alias="VISION_RETRIES")

    # Video Generation Settings
    video_quality: str | None = Field(default=None, validation_alias="VIDEO_QUALITY")
    custom_min_video_sec: int = Field(default=60, validation_alias="CUSTOM_MIN_VIDEO_SEC")
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
    video_fps: str | None = Field(default=None, validation_alias="VIDEO_FPS")
    audio_bitrate: str | None = Field(default=None, validation_alias="AUDIO_BITRATE")
    video_scale_flags: str | None = Field(default=None, validation_alias="VIDEO_SCALE_FLAGS")
    video_tune: str | None = Field(default=None, validation_alias="VIDEO_TUNE")

    # Image Settings
    custom_img_quality: str = Field(default="", validation_alias="CUSTOM_IMG_QUALITY")
    custom_min_img_score: int = Field(default=1, validation_alias="CUSTOM_MIN_IMG_SCORE")
    custom_img_max_per_query: int = Field(default=8, validation_alias="CUSTOM_IMG_MAX_PER_QUERY")
    custom_hook_segments: int = Field(default=2, validation_alias="CUSTOM_HOOK_SEGMENTS")
    custom_hook_extra_candidates: int = Field(default=10, validation_alias="CUSTOM_HOOK_EXTRA_CANDIDATES")
    custom_hook_min_img_score: int = Field(default=3, validation_alias="CUSTOM_HOOK_MIN_IMG_SCORE")
    
    # Image Sources & Network
    img_sources: str = Field(default="ddg,openverse,wikimedia", validation_alias="IMG_SOURCES")
    ddg_images_backend: str = Field(default="lite", validation_alias="DDG_IMAGES_BACKEND")
    ddg_search_timeout_sec: float = Field(default=25.0, validation_alias="DDG_SEARCH_TIMEOUT_SEC")
    openverse_timeout_sec: float = Field(default=20.0, validation_alias="OPENVERSE_TIMEOUT_SEC")
    allow_avif: bool = Field(default=False, validation_alias="ALLOW_AVIF")
    wikimedia_user_agent: str = Field(default="", validation_alias="WIKIMEDIA_USER_AGENT")
    
    # Text Processing
    enable_text_rank: bool = Field(default=True, validation_alias="CUSTOM_IMG_TEXT_RANK")
    
    # Safety
    img_blocked_hosts: str = Field(default="", validation_alias="IMG_BLOCKED_HOSTS")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
