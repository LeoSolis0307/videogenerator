from importlib import import_module


_LAZY_MODULES = {
    "reddit_scraper": ".reddit_scraper",
    "text_processor": ".text_processor",
    "tts": ".tts_engine",
    "voice_clone": ".voice_clone",
    "image_downloader": ".image_fetcher",
    "video_renderer": ".video_renderer",
    "story_generator": ".story_generator",
    "custom_video": ".custom_video",
    "reddit_story_importer": ".reddit_story_importer",
}

__all__ = [
    "reddit_scraper",
    "text_processor",
    "tts",
    "voice_clone",
    "image_downloader",
    "video_renderer",
    "story_generator",
    "custom_video",
    "reddit_story_importer",
]


def __getattr__(name: str):
    module_path = _LAZY_MODULES.get(name)
    if not module_path:
        raise AttributeError(f"module 'core' has no attribute '{name}'")
    mod = import_module(module_path, __name__)
    globals()[name] = mod
    return mod


def __dir__():
    return sorted(set(globals()) | set(__all__))
