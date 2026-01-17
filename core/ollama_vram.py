import os
from urllib.parse import urlparse

import requests


def _ollama_host() -> str:
    host = (os.environ.get("OLLAMA_HOST") or "").strip()
    if host:
        return host.rstrip("/")

    raw = (os.environ.get("OLLAMA_URL") or "http://localhost:11434/api/generate").strip()
    if not raw:
        return "http://localhost:11434"

    try:
        p = urlparse(raw)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}".rstrip("/")
    except Exception:
        pass

    return raw.split("/api/")[0].rstrip("/")


def _api(path: str) -> str:
    return _ollama_host().rstrip("/") + "/" + path.lstrip("/")


def try_unload_model(model: str, *, timeout_sec: int = 20) -> bool:
    name = (model or "").strip()
    if not name:
        return False

                                             
    url = _api("/api/generate")
    payload = {
        "model": name,
        "prompt": " ",
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0},
    }
    try:
        r = requests.post(url, json=payload, timeout=max(5, int(timeout_sec)))
        r.raise_for_status()
        return True
    except Exception:
        return False
