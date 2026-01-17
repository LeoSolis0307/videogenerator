import base64
import os
import re
import time
from urllib.parse import urlparse

import requests

from core.ollama_metrics import maybe_print_ollama_speed

                                        
                                                                                  
                                                                                                  
                   
                                          
                                      
                
VISION_MODEL = (os.environ.get("VISION_MODEL") or "minicpm-v:latest").strip() or "minicpm-v:latest"
VISION_TIMEOUT_SEC = int(os.environ.get("VISION_TIMEOUT_SEC", "90") or "90")
VISION_RETRIES = int(os.environ.get("VISION_RETRIES", "3") or "3")

_DEBUG_IMG_VALIDATION = (os.environ.get("DEBUG_IMG_VALIDATION") or "").strip() in {"1", "true", "True", "YES", "yes"}
_VISION_PRINT_SPEED = (os.environ.get("VISION_PRINT_SPEED") or "").strip().lower() in {"1", "true", "yes", "si", "sí", "on"}

                                                                           
_AVAILABLE: bool | None = None


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


def _ollama_api_url(path: str) -> str:
    return _ollama_host().rstrip("/") + "/" + path.lstrip("/")


def _is_transient_error(err: Exception) -> bool:
    try:
        if isinstance(err, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(err, requests.HTTPError):
            resp = getattr(err, "response", None)
            code = getattr(resp, "status_code", None)
            return code in {429, 500, 502, 503, 504}
    except Exception:
        pass

    msg = str(err).lower()
    return any(k in msg for k in ("timeout", "timed out", "connection", "reset", "broken pipe"))


def score_image(
    image_path: str,
    requirement: str,
    *,
    note: str = "",
    model: str | None = None,
    timeout_sec: int | None = None,
    retries: int | None = None,
) -> int:
    global _AVAILABLE

    if _AVAILABLE is False:
        return 1

    model_name = (model or VISION_MODEL).strip() or "minicpm-v:latest"
    timeout = max(10, int(timeout_sec or VISION_TIMEOUT_SEC))
    tries = max(1, int(retries or VISION_RETRIES))

    api_chat = _ollama_api_url("/api/chat")

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        if _DEBUG_IMG_VALIDATION:
            print(f"[VISION] No se pudo leer imagen: {e}")
        return 1

    note_line = (note or "").strip()
    req = (requirement or "").strip() or "photo"

                                                                    
                                                                                    
    prompt = (
        "You are an image-to-requirement evaluator for selecting REAL internet photos for a YouTube video segment. "
        "Return ONLY ONE DIGIT (1-5). No words.\n"
        "Scoring rubric:\n"
        "5 = perfect: clearly shows the exact subject (object/person/place) and matches context, time, and key attributes.\n"
        "4 = good: correct subject, minor mismatch (angle/variant/background) but still clearly usable.\n"
        "3 = acceptable: somewhat relevant, but missing key attribute; could work if nothing better.\n"
        "2 = weak: loosely related category, likely confusing.\n"
        "1 = unrelated/wrong/abstract/AI art/diagram/screenshot/meme/text-heavy.\n"
        "Penalties: prefer real photos; penalize illustrations, fanart, AI images, heavy text overlays, collages, watermarks.\n"
        f"REQUIREMENT: {req}\n"
        + (f"EXTRA_CONTEXT: {note_line}\n" if note_line else "")
        + "Answer:"
    )

    payload = {
        "model": model_name,
        "stream": False,
                                                                                                           
                                                             
        **({"keep_alive": int(os.environ.get("VISION_KEEP_ALIVE", "") or 0)} if os.environ.get("VISION_KEEP_ALIVE") else {}),
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }
        ],
        "options": {"temperature": 0.05},
    }

    last_err: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.post(api_chat, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json() if r.content else {}
            maybe_print_ollama_speed(data, tag="VISION", enabled=_VISION_PRINT_SPEED)
            _AVAILABLE = True
            ans = str((data.get("message") or {}).get("content") or "").strip().upper()
            if _DEBUG_IMG_VALIDATION:
                print(f"[VISION] {model_name} => {ans[:120]}")
            m = re.search(r"\b([1-5])\b", ans)
            if not m:
                return 1
            return int(m.group(1))
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "[winerror 10061" in msg or "connection refused" in msg or "failed to establish a new connection" in msg:
                _AVAILABLE = False
                return 1

            try:
                if isinstance(e, requests.HTTPError):
                    resp = getattr(e, "response", None)
                    code = getattr(resp, "status_code", None)
                    if code == 500:
                        _AVAILABLE = False
                        return 1
            except Exception:
                pass

            if attempt < tries - 1:
                if _DEBUG_IMG_VALIDATION:
                    print(f"[VISION] ⚠️ fallo {attempt+1}/{tries}: {e}")
                time.sleep(1.5 * (attempt + 1))
                continue
            break

    if last_err and _is_transient_error(last_err):
        _AVAILABLE = False
        return 1

    _AVAILABLE = False
    return 1
