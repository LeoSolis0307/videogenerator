import json
import html
import hashlib
import builtins
import math
import os
import random
import re
import sys
import subprocess
import concurrent.futures
import requests
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote_plus
from typing import Tuple
import time
import base64
from urllib.parse import urlparse, urljoin

                                                                                
from core.config import settings
from core.llm.client import llm_client
from core.models import VideoPlan, ScriptSegment, TimelineItem
from utils.fs import crear_carpeta_proyecto
from core import tts, image_downloader, text_processor, reddit_scraper, story_generator
from core.video_renderer import (
    append_intro_to_video,
    audio_duration_seconds,
    combine_audios_with_silence,
    render_video_ffmpeg,
)

try:
    from ddgs import DDGS as DDGS
    _DDG_BACKEND = "ddgs"
except Exception:
    try:
        from duckduckgo_search import DDGS as DDGS
        _DDG_BACKEND = "duckduckgo_search"
    except Exception:
        DDGS = None
        _DDG_BACKEND = "none"

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# Configuración migrada a core/config.py (settings)


def print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        file_obj = kwargs.get("file", sys.stdout)
        enc = getattr(file_obj, "encoding", None) or "utf-8"
        safe_args = []
        for a in args:
            s = str(a)
            safe_args.append(s.encode(enc, errors="replace").decode(enc, errors="replace"))
        builtins.print(*safe_args, **kwargs)

def _text_model_for_seconds(target_seconds: int | None) -> str:
    try:
        sec = int(target_seconds or 0)
    except Exception:
        sec = 0

    # Para Shorts prioriza siempre modelo corto (más estable y rápido).
    if sec > 0 and sec < 300:
        return settings.ollama_text_model_short

    # En largos, si el usuario fijó modelo explícito, respétalo.
    if settings.ollama_text_model:
        return settings.ollama_text_model

    return settings.ollama_text_model_long if sec >= 300 else settings.ollama_text_model_short

WIKI_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_APIS = [
    "https://api.openverse.org/v1/images",
    "https://api.openverse.engineering/v1/images",
]
FLICKR_FEED_API = "https://www.flickr.com/services/feeds/photos_public.gne"
BING_IMG_SEARCH_API = "https://www.bing.com/images/search"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

_USER_AGENTS = [
    USER_AGENT,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

ALLOW_AVIF = bool(settings.allow_avif)
_DEBUG_IMG_VALIDATION = False
IMG_ANTIBOT_PROXY = bool(settings.img_antibot_proxy)
IMG_ANTIBOT_BROWSER = bool(settings.img_antibot_browser)

_SITE_IMAGE_SOURCES: dict[str, str] = {
    "cleanpng": "cleanpng.com",
    "stickpng": "stickpng.com",
    "freepngimg": "freepngimg.com",
    "toppng": "toppng.com",
    "similarpng": "similarpng.com",
    "flaticon": "flaticon.com",
    "pixabay": "pixabay.com",
    "pexels": "pexels.com",
    "unsplash": "unsplash.com",
    "flickr": "flickr.com",
    "wikipedia": "commons.wikimedia.org",
    "stockvault": "stockvault.net",
    "pxhere": "pxhere.com",
    "publicdomainpictures": "publicdomainpictures.net",
    "kaboompics": "kaboompics.com",
    "burst": "burst.shopify.com",
    "imgur": "imgur.com",
}

_VISION_AVAILABLE = None
_ALLOW_PLACEHOLDER_IMAGES = bool(settings.custom_allow_placeholder_images)
_COMFY_FALLBACK_ON_MISSING = bool(
    settings.custom_comfy_fallback_on_missing_images and settings.comfyui_enabled
)
_DDG_DISABLED_RUNTIME = False
_OPENVERSE_DISABLED_RUNTIME = False


def _detect_image_format(path: str) -> str | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            fmt = (img.format or "").strip().lower()
            return fmt or None
    except Exception:
        pass

    try:
        with open(path, "rb") as f:
            head = f.read(64)
    except Exception:
        return None

    if len(head) >= 3 and head[0:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(head) >= 8 and head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(head) >= 6 and (head.startswith(b"GIF87a") or head.startswith(b"GIF89a")):
        return "gif"
    if len(head) >= 2 and head.startswith(b"BM"):
        return "bmp"
    if len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if len(head) >= 12 and head[4:8] == b"ftyp" and b"avif" in head[8:16].lower():
        return "avif"
    return None



_STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "u", "de", "del", "al", "a",
    "en", "por", "para", "con", "sin", "sobre", "entre", "tras", "desde", "hasta", "que", "como",
    "cuando", "donde", "quien", "quienes", "cual", "cuales", "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas", "mi", "mis", "tu", "tus", "su", "sus", "me", "te", "se",
    "lo", "le", "les", "ya", "muy", "mas", "más", "menos", "tambien", "también", "pero", "porque",
    "si", "sí", "no", "ni", "solo", "sólo", "todo", "toda", "todos", "todas", "cada",
    "esta", "está", "estan", "están", "era", "eran", "fue", "fueron", "ser", "estar", "haber",
    "hay", "habia", "había", "habian", "habían", "tiene", "tienen", "tenia", "tenía", "tener",
    "hace", "hacen", "hacer", "dijo", "dijeron", "dice", "dicen",
    "ahi", "ahí", "aqui", "aquí", "alli", "allí",
}

_STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "without", "from", "by",
    "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "as", "at", "it", "its", "into", "over", "under", "about", "between", "after", "before",
    "photo", "image", "picture", "stock", "illustration",
}

_BAD_QUERY_WORDS = {
    "pinche", "verga", "chingue", "chingar", "chingado", "chingada", "chingados", "chingadas",
    "pendejo", "pendeja", "pendejos", "pendejas", "cabrón", "cabron", "cabrona", "culero", "culera",
    "mierda", "puta", "puto", "putas", "putos", "carajo", "mamada", "mamadisimo", "mamadísimo",
    "güey", "wey", "wey", "compa", "banda", "morra", "vato", "ruca", "carnal",
}

_QUERY_TERM_MAP_ES_EN: dict[str, str] = {
    "gato": "cat",
    "gata": "cat",
    "gris": "gray",
    "perro": "dog",
    "persona": "person",
    "hombre": "man",
    "mujer": "woman",
    "niño": "boy",
    "niña": "girl",
    "cerebro": "brain",
    "mano": "hand",
    "ojos": "eyes",
    "ojo": "eye",
    "robot": "robot",
    "implante": "implant",
    "neural": "neural",
    "parálisis": "paralysis",
    "paralisis": "paralysis",
    "médico": "medical",
    "medico": "medical",
    "realista": "realistic",
    "retrato": "portrait",
    "primer": "close",
    "plano": "up",
}

_ANIMAL_QUERY_TERMS: set[str] = {
    "cat", "gato", "gata", "dog", "perro", "perra", "rat", "rata", "mouse", "mice",
    "bird", "pajaro", "pájaro", "horse", "caballo", "cow", "vaca", "lion", "leon", "león",
    "tiger", "tigre", "wolf", "lobo", "fox", "zorro", "bear", "oso", "fish", "pez",
}


def _tokenize_words(text: str) -> list[str]:
    if not text:
        return []
                                                                    
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{2,}", str(text))
    return [w.lower() for w in words]


def _extract_keywords(text: str, *, max_keywords: int = 7) -> list[str]:
    words = _tokenize_words(text)
    if not words:
        return []

    freq: dict[str, int] = {}
    for w in words:
        if len(w) < 3:
            continue
        if w in _STOPWORDS_ES or w in _STOPWORDS_EN:
            continue
        if w in _BAD_QUERY_WORDS:
            continue
        if w.isdigit():
            continue
        freq[w] = freq.get(w, 0) + 1

    if not freq:
        return []

    scored = sorted(
        freq.items(),
        key=lambda kv: (kv[1] * math.log(1 + len(kv[0])), kv[1], len(kv[0])),
        reverse=True,
    )
    out: list[str] = []
    for w, _ in scored:
        if w in out:
            continue
        out.append(w)
        if len(out) >= int(max_keywords):
            break
    return out


def _build_better_image_query(*, base_query: str, text_es: str = "", note: str = "", brief: str = "") -> str:
    base = (base_query or "").strip()
    context = " ".join([brief or "", note or "", text_es or ""]).strip()
    kws = _extract_keywords(context, max_keywords=6)

    base_low = base.lower()
    generic = any(g in base_low for g in ["photo", "person", "people", "scene", "dramatic", "story", "background"])

    parts: list[str] = []
    if generic and kws:
        parts.extend(kws[:4])
        if base:
            parts.append(base)
    else:
        if base:
            parts.append(base)
        if kws:
            parts.extend(kws[:4])

    q = " ".join([p for p in parts if p]).strip()
    q_words = q.split()
    if len(q_words) > 12:
        q = " ".join(q_words[:12])
    return _sanitize_image_query(q or (base or "photo"))


def _sanitize_image_query(query: str, *, max_words: int = 10) -> str:
    q = re.sub(r"\s+", " ", str(query or "").strip())
    if not q:
        return "person portrait documentary photo"

    words = _tokenize_words(q)
    clean: list[str] = []
    for w in words:
        if len(w) < 3:
            continue
        if w in _STOPWORDS_ES or w in _STOPWORDS_EN:
            continue
        if w in _BAD_QUERY_WORDS:
            continue
        if w.isdigit():
            continue
        clean.append(w)

    if len(clean) < 2:
        clean = _extract_keywords(q, max_keywords=max(4, max_words // 2))

    if not clean:
        return "person portrait documentary photo"

    out = " ".join(clean[: max(3, int(max_words))]).strip()
    return out or "person portrait documentary photo"


def _candidate_text_relevance(url: str, title: str, *, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hay_low = ((title or "") + " " + (url or "")).lower()
    hits = 0
    for kw in keywords:
        if not kw or len(kw) < 3:
            continue
        if kw.lower() in hay_low:
            hits += 1
                                                    
    score = float(hits) + (0.05 if (title or "").strip() else 0.0)
    if "neuralink" in hay_low:
        if any(k in hay_low for k in ["implant", "implante", "chip", "electrode", "n1", "telepathy", "thread"]):
            score += 2.0
        elif any(k in hay_low for k in ["elon", "musk", "future", "master plan"]):
            score -= 0.75
    return score


def _expand_query_terms_es_en(query: str) -> list[str]:
    toks = _tokenize_words(query)
    if not toks:
        return []
    out: list[str] = []
    for w in toks:
        mapped = _QUERY_TERM_MAP_ES_EN.get(w)
        if mapped and mapped not in out:
            out.append(mapped)
    return out


def _animal_query_relevance_ok(query: str, title: str, url: str) -> bool:
    q_words = set(_tokenize_words(_simplify_query(query)))
    if not q_words:
        return True

    animal_need = {w for w in q_words if w in _ANIMAL_QUERY_TERMS}
    for w in _expand_query_terms_es_en(" ".join(q_words)):
        if w in _ANIMAL_QUERY_TERMS:
            animal_need.add(w)

    if not animal_need:
        return True

    hay = f"{title or ''} {url or ''}".lower()
    return any(a in hay for a in animal_need)


def _query_has_neuralink_implant_intent(query: str) -> bool:
    q_words = set(_tokenize_words(_simplify_query(query)))
    if "neuralink" not in q_words:
        return False
    implant_terms = {"implant", "implante", "chip", "electrode", "electrodo", "n1", "thread", "hilo", "telepathy"}
    return any(t in q_words for t in implant_terms)


def _candidate_matches_neuralink_implant(title: str, url: str) -> bool:
    text = f"{title or ''} {url or ''}".lower()
    implant_terms = {"implant", "implante", "chip", "electrode", "electrodo", "n1", "thread", "hilo", "telepathy"}
    return ("neuralink" in text) and any(t in text for t in implant_terms)

                                                                      
                                                                                                                    
IMG_BLOCKED_HOSTS = settings.blocked_hosts_set


def _is_blocked_image_host(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower()
        if not host:
            return False
        for bad in IMG_BLOCKED_HOSTS:
            if host == bad or host.endswith("." + bad):
                return True
        return False
    except Exception:
        return False


_WIKIMEDIA_UA = settings.wikimedia_user_agent
_WARNED_WIKIMEDIA_UA = False


def _is_wikimedia_host(host: str) -> bool:
    host = (host or "").lower().strip()
    return host.endswith("wikimedia.org") or host.endswith("wikipedia.org")


def _maybe_warn_wikimedia_ua() -> None:
    global _WARNED_WIKIMEDIA_UA
    if _WARNED_WIKIMEDIA_UA:
        return
    if not _WIKIMEDIA_UA:
        _WARNED_WIKIMEDIA_UA = True
        print(
            "[IMG-WEB] ℹ️ Consejo: para evitar 429 de Wikimedia, define WIKIMEDIA_USER_AGENT "
            "con un UA identificable (incluye contacto)."
        )


def _resolve_wikimedia_thumb_via_api(url: str) -> str | None:
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if not host.endswith("upload.wikimedia.org"):
            return None

        parts = [x for x in (p.path or "").split("/") if x]
                                                                            
        if "thumb" not in parts:
            return None
        i = parts.index("thumb")
        if len(parts) <= i + 3:
            return None
        filename = parts[i + 3]
        if not filename:
            return None

                                                                          
                                  
        width = 1600
        if parts and parts[-1]:
            m = re.match(r"^(\d{2,5})px-", parts[-1])
            if m:
                try:
                    width = int(m.group(1))
                except Exception:
                    width = 1600
        width = max(256, min(int(width), 2400))

        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": f"File:{filename}",
            "iiprop": "url|mime",
            "iiurlwidth": width,
        }

        _maybe_warn_wikimedia_ua()
        headers = {"User-Agent": (_WIKIMEDIA_UA or USER_AGENT)}
        resp = requests.get(WIKI_API, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            info = page.get("imageinfo") if isinstance(page, dict) else None
            if not info:
                continue
            entry = info[0]
            thumb = entry.get("thumburl")
            if thumb:
                return str(thumb)
            direct = entry.get("url")
            if direct:
                return str(direct)
        return None
    except Exception:
        return None



def generar_guion_personalizado_a_plan(
    brief: str,
    *,
    min_seconds: int | None = None,
    max_seconds: int | None = None,
    seleccionar_imagenes: bool = False,
) -> str | None:
    carpeta = crear_carpeta_proyecto(prefix="custom")
    try:
        plan = generar_plan_personalizado(brief, min_seconds=min_seconds)
    except Exception as e:
        print(f"[CUSTOM] ❌ No se pudo crear el plan: {e}")
        return None

                                                                       
    if not plan.youtube_title_es:
        plan.youtube_title_es = plan.title_es or "Video personalizado"

                                                                                 
                                                                              
    try:
        ts_val = plan.target_seconds
        if not ts_val:
            ts_val = min_seconds or settings.custom_min_video_sec

        if int(ts_val) == 60:
            base_title = str(plan.youtube_title_es or "").strip()
            script_es = str(plan.script_es or "").strip()
            plan.youtube_title_es = _append_shorts_hashtags_to_title(
                base_title,
                brief=str(plan.brief or brief or "").strip(),
                script_es=script_es,
            )
    except Exception:
        pass

                                                                                          
    try:
        yt_title_txt = os.path.join(carpeta, "youtube_title.txt")
        with open(yt_title_txt, "w", encoding="utf-8") as f:
            f.write(str(plan.youtube_title_es or "").strip() + "\n")
    except Exception:
        pass
    plan.seleccionar_imagenes = bool(seleccionar_imagenes)

    # Metadatos de duración para render posterior.
    try:
        plan.extra_duration = {
            "min_seconds": int(min_seconds) if min_seconds is not None else None,
            "max_seconds": int(max_seconds) if max_seconds is not None else None,
        }
    except Exception:
        pass

    plan_path = os.path.join(carpeta, "custom_plan.json")
    try:
        data = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
        if isinstance(data, dict):
            data["min_seconds"] = int(min_seconds) if min_seconds is not None else data.get("min_seconds")
            data["max_seconds"] = int(max_seconds) if max_seconds is not None else data.get("max_seconds")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[CUSTOM] ✅ Plan (solo guion) guardado en {plan_path}")
    except Exception as e:
        print(f"[CUSTOM] ⚠️ No se pudo guardar el plan: {e}")
        return None

                                                                      
                                                                  
    try:
        if (os.environ.get("settings.ollama_text_num_ctx") or "1").strip().lower() in {"1", "true", "yes", "si", "sí"}:
                                                                                 
            ok_any = False
            for m in {settings.ollama_text_model_short, settings.ollama_text_model_long}:
                if m and ollama_vram.try_unload_model(m):
                    ok_any = True
                    print(f"[CUSTOM] ℹ️ Modelo de texto descargado: {m}")
            if not ok_any:
                pass
    except Exception:
        pass

    return carpeta


def intentar_descargar_modelo_texto() -> bool:
    try:
        ok_any = False
        for m in {settings.ollama_text_model_short, settings.ollama_text_model_long}:
            if m and ollama_vram.try_unload_model(m):
                ok_any = True
        return ok_any
    except Exception:
        return False


def _export_long_videos_to_videos_dir(
    *,
    plan: Dict[str, Any],
    carpeta_plan: str,
    video_final: str | None,
    video_con_intro: str | None,
) -> tuple[str | None, Dict[str, Any]]:
    """Mueve los MP4 de videos largos a la carpeta raíz `videos/`.

    - Mantiene `output/...` para planes/activos.
    - Para planes largos, el entregable queda en `videos/<nombre_carpeta_plan>/`.
    - Devuelve (export_video_path, export_info_dict).
    """

    try:
        target_seconds = int(plan.get("target_seconds") or 0)
    except Exception:
        target_seconds = 0

    try:
        min_long = int((os.environ.get("LONG_VIDEO_MIN_SECONDS_FOR_VIDEOS_DIR") or "300").strip() or "300")
    except Exception:
        min_long = 300

    if target_seconds < min_long:
        return (video_con_intro or video_final), {}

    root_dir = (os.environ.get("LONG_VIDEOS_DIR") or "videos").strip() or "videos"
    videos_root = os.path.abspath(root_dir)
    dest_dir = os.path.join(videos_root, os.path.basename(os.path.abspath(carpeta_plan)))
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except Exception:
        return (video_con_intro or video_final), {}

    def _rel_from_cwd(p: str) -> str:
        try:
            return os.path.relpath(p, os.path.abspath(".")).replace("\\", "/")
        except Exception:
            return p

    def _move(src: str | None) -> str | None:
        if not src:
            return None
        src_abs = os.path.abspath(src)
        if not os.path.exists(src_abs):
            return None
        dst_abs = os.path.join(dest_dir, os.path.basename(src_abs))
        if os.path.abspath(dst_abs) == src_abs:
            return dst_abs

        if os.path.exists(dst_abs):
            base, ext = os.path.splitext(dst_abs)
            try:
                import time as _time

                stamp = _time.strftime("%H%M%S")
            except Exception:
                stamp = "dup"
            dst_abs = f"{base}_{stamp}{ext}"

        try:
            os.replace(src_abs, dst_abs)
        except Exception:
            # Fallback copia si el move falla
            try:
                with open(src_abs, "rb") as fsrc, open(dst_abs, "wb") as fdst:
                    fdst.write(fsrc.read())
            except Exception:
                return None
            try:
                os.remove(src_abs)
            except Exception:
                pass

        return dst_abs

    moved_video_final = _move(video_final)
    moved_video_intro = _move(video_con_intro)
    export_video = moved_video_intro or moved_video_final

    export_info: Dict[str, Any] = {
        "export_dir": _rel_from_cwd(dest_dir),
        "export_video": _rel_from_cwd(export_video) if export_video else None,
        "moved": {
            "video_final": _rel_from_cwd(moved_video_final) if moved_video_final else None,
            "video_con_intro": _rel_from_cwd(moved_video_intro) if moved_video_intro else None,
        },
    }

    try:
        p = os.path.join(carpeta_plan, "EXPORT_VIDEO.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write((export_info.get("export_video") or "") + "\n")
    except Exception:
        pass

    return export_video, export_info


def check_text_llm_ready() -> bool:
    try:
        _ = _ollama_generate("Reply only with: OK", temperature=0.0, max_tokens=8, model=settings.ollama_text_model_short)
        return True
    except Exception as e:
        print(f"[CUSTOM] ❌ Ollama no está listo para generar guiones: {e}")
        return False


def _sanitize_brief_for_duration(brief: str) -> str:
    b = (brief or "").strip()
    if not b:
        return ""

    b = re.sub(r"\b(shorts?|yt\s*shorts?)\b", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\b\d{1,4}\s*(segundos|segundo|s)\b", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\b\d{1,3}\s*(minutos|minuto|min)\b", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\s+", " ", b).strip()
    return b


def _extract_hook_from_brief(brief: str) -> str:
    b = (brief or "").strip()
    if not b:
        return ""
    m = re.search(r"(?is)\bgancho\s*:\s*\"([^\"]{6,220})\"", b)
    if not m:
        return ""
    hook = (m.group(1) or "").strip()
    hook = re.sub(r"\s+", " ", hook)
    return hook


def _extract_tone_from_brief(brief: str) -> str:
    b = (brief or "").strip()
    if not b:
        return ""

    m = re.search(r"(?is)\btono\s*:\s*(\"[^\"]{2,140}\"|[^\n]{2,140})", b)
    if not m:
        return ""

    tone = (m.group(1) or "").strip().strip('"')
    tone = re.sub(r"\s+", " ", tone)
    return tone


def _requires_curious_final(brief: str, tone_hint: str = "") -> bool:
    txt = ((brief or "") + " " + (tone_hint or "")).lower()
    if not txt.strip():
        return False
    triggers = [
        "dato curioso",
        "datos curiosos",
        "curiosidad",
        "curiosidades",
        "fun fact",
        "did you know",
        "sabías que",
        "sabias que",
    ]
    return any(t in txt for t in triggers)


def _tiene_saludo_o_outro(texto: str) -> bool:
    t = (texto or "").strip().lower()
    if not t:
        return False
                                             
    bad = [
        "hola",
        "hey",
        "buenas",
        "qué tal",
        "que tal",
        "bienvenid",
        "en este video",
        "en este vídeo",
        "hoy vamos",
        "hoy te",
        "hoy les",
        "suscr",
        "dale like",
        "like y",
        "comenta",
        "sígueme",
        "sigueme",
    ]
    return any(x in t for x in bad)


def _prompt_rewrite_opening_segment(
    brief: str,
    contexto: str,
    hook_es: str,
    first_segment: Dict[str, Any],
    *,
    target_seconds: int,
) -> str:
    contexto_line = contexto if contexto else "Sin contexto web disponible."
    seg_json = json.dumps(first_segment or {}, ensure_ascii=False)[:1500]

                                                          
    if int(target_seconds) >= 300:
        wmin, wmax = 45, 85
    else:
        wmin, wmax = 24, 55

    hook_line = (hook_es or "").strip()
    if hook_line:
        hook_line = re.sub(r"\s+", " ", hook_line)

    is_doc_long = int(target_seconds) >= 300
    role_line = (
        "Actúa como un editor experto de documentales de YouTube en español (México) especializado en RETENCIÓN. "
        if is_doc_long
        else "Eres guionista/editor de YouTube Shorts en español especializado en RETENCIÓN. "
    )

    return (
        role_line
        + "Reescribe SOLO el PRIMER segmento para que sea un HOOK agresivo y curioso. "
        + "Debe enganchar en 1 segundo, sin saludos ni introducciones flojas.\n"
        + f"BRIEF: {brief}\n"
        + f"DATOS RAPIDOS: {contexto_line}\n"
        + f"HOOK_OBLIGATORIO: {hook_line}\n"
        + f"SEGMENTO_ACTUAL_JSON: {seg_json}\n\n"
        + "REGLAS ESTRICTAS:\n"
        + "- text_es DEBE EMPEZAR EXACTAMENTE con HOOK_OBLIGATORIO (primera oración).\n"
        + "- PROHIBIDO: 'hola', 'bienvenidos', 'hoy', 'en este video', 'suscríbete', 'dale like'.\n"
        + (
            "- Usa una escena dramática o paradoja, y plantea una GRAN PREGUNTA que se responderá al final.\n"
            if is_doc_long
            else ""
        )
        + "- Abre un 'open loop': deja algo pendiente que se resuelve en el siguiente segmento.\n"
        + "- Frases cortas, ritmo rápido, cero relleno.\n"
        + "- Mantén el mismo tema; no inventes afirmaciones dudosas.\n\n"
        + "Devuelve SOLO un objeto JSON con claves exactas: text_es, image_query, image_prompt, note. "
        + f"text_es debe tener {wmin}-{wmax} palabras. "
        + "JSON válido, nada más."
    )


def _words(text: str) -> int:
    return len((text or "").split())


def _estimar_segundos(texto: str) -> float:
    palabras = _words(texto or "")
    if palabras <= 0:
        return 0.0
    wpm = 140.0
    estimado = (palabras / wpm) * 60.0
    estimado = estimado * 1.35 + 1.5
    return max(3.0, float(estimado))


def _segments_word_stats(segmentos: List[Dict[str, Any]]) -> tuple[int, int, int]:
    ws = [_words(str(s.get("text_es") or "")) for s in (segmentos or []) if isinstance(s, dict)]
    if not ws:
        return 0, 0, 0
    return min(ws), int(sum(ws) / len(ws)), max(ws)


def _plan_quality_issues(segmentos: List[Dict[str, Any]], *, target_seconds: int) -> List[str]:
    issues: List[str] = []
    if not segmentos:
        return ["sin_segmentos"]

    texts = [str(s.get("text_es") or "").strip() for s in segmentos]

                         
    lowered = [t.lower() for t in texts]

                                                                                
    if texts:
        first = texts[0]
        if _tiene_saludo_o_outro(first):
            issues.append("saludo_en_hook")
    intro_hits = sum(
        1
        for t in lowered
        if any(x in t for x in ["hola", "bienvenid", "hoy", "en este video", "en este vídeo", "hoy vamos"])
    )
    if intro_hits > 1:
        issues.append("intro_repetida")

                                                                            
    if _requires_curious_final(" ".join(texts)):
        def _has_final_marker(t: str) -> bool:
            tl = t.lower()
            return ("dato curioso final" in tl) or ("¿sabías que" in tl) or ("sabias que" in tl) or ("sabías que" in tl)

        final_markers = [i for i, t in enumerate(lowered) if _has_final_marker(t)]
        if not final_markers:
            issues.append("sin_dato_curioso_final")
        else:
            if len(final_markers) != 1:
                issues.append("multiples_datos_curiosos_finales")
            if final_markers and final_markers[-1] != len(segmentos) - 1:
                issues.append("dato_curioso_no_es_ultimo")

                               
    if len(set(texts)) != len(texts):
        issues.append("segmentos_duplicados")

                                                                    
    if int(target_seconds) >= 300:
        w_min, w_avg, w_max = _segments_word_stats(segmentos)
        if w_min < 30:
            issues.append("segmentos_demasiado_cortos")
        if w_max > 120:
            issues.append("segmentos_demasiado_largos")
    return issues


def _prompt_add_segments(brief: str, contexto: str, *, need_words: int, n_segments: int, last_note: str = "") -> str:
    contexto_line = contexto if contexto else "Sin contexto web disponible."
    last_line = (last_note or "").strip()
    parts = [
        "You are continuing a Spanish YouTube video script plan. ",
        "Add NEW segments that continue the narrative and add real information (no filler). ",
        "Style: energetic and human (not monotone), with light sarcasm and occasional short jokes; avoid offensive humor, profanity, and emojis. ",
        "These are MID segments: do NOT write an ending/closure, no generic outro, no 'suscríbete', no despedidas. ",
        "Focus on concrete curious facts, examples, and details about the chosen topic. ",
        "Return ONLY a JSON array (no object) with segment objects.\n",
        f"BRIEF: {brief}\n",
        f"FAST FACTS: {contexto_line}\n",
        f"TARGET: add about {need_words} words total across {n_segments} segments.\n",
    ]
    if last_line:
        parts.append(f"LAST_SEGMENT_NOTE: {last_line}\n")
    parts.extend(
        [
            "Each segment object must have keys: text_es (80-140 words), image_query (short English), image_prompt (English), note (Spanish). ",
            "Return valid JSON, nothing else.",
        ]
    )
    return "".join(parts)


def _es_cierre_valido(texto: str) -> bool:
    t = (texto or "").lower()
    if not t.strip():
        return False
                                 
    if t.strip().endswith("..."):
        return False
    if "¿sabías que..." in t or "sabías que..." in t or "sabias que..." in t:
        return False
                                                                        
    buenas = [
        "dato curioso",
        "dato final",
        "¿sabías que",
        "sabías que",
        "sabias que",
        "lo más curioso",
        "lo mas curioso",
        "poca gente sabe",
        "detalle curioso",
    ]
    malas = [
        "en este video exploramos",
        "en este vídeo exploramos",
        "en este video vimos",
        "y eso es todo",
        "y se acaba",
    ]
    if any(m in t for m in malas) and not any(b in t for b in buenas):
        return False
    return any(b in t for b in buenas)


def _prompt_rewrite_closing_segment(brief: str, contexto: str, last_segment: Dict[str, Any], *, target_seconds: int) -> str:
    contexto_line = contexto if contexto else "Sin contexto web disponible."
    last_json = json.dumps(last_segment or {}, ensure_ascii=False)[:1500]
                                                               
    if int(target_seconds) >= 300:
        wmin, wmax = 70, 140
    else:
        wmin, wmax = 25, 65
    is_doc_long = int(target_seconds) >= 300
    return (
        ("Actúa como un guionista senior de documentales para YouTube en español (México). " if is_doc_long else "Eres guionista de YouTube en español. ")
        + "Reescribe SOLO el ÚLTIMO segmento para que sea un cierre con un DATO CURIOSO FINAL. "
        + "Tono: energético y humano (no plano), con sarcasmo ligero y, si encaja, un chiste corto (sin groserías ni humor ofensivo). "
        + "No cierres con frases vagas tipo 'exploramos el mundo mágico'. Debe quedar una idea concreta memorable.\n"
        + f"BRIEF: {brief}\n"
        + f"DATOS RAPIDOS: {contexto_line}\n"
        + f"SEGMENTO_ACTUAL_JSON: {last_json}\n\n"
        + "Devuelve SOLO un objeto JSON con claves exactas: text_es, image_query, image_prompt, note. "
        + f"text_es debe tener {wmin}-{wmax} palabras. "
        + "Incluye explícitamente una frase tipo 'Dato curioso final:' o '¿Sabías que...?' PERO con el dato completo en la misma oración. "
        + "PROHIBIDO usar puntos suspensivos '...'. No dejes preguntas incompletas. "
        + "El dato debe ser específico (nombres propios, obra/película/libro, o detalle verificable). "
        + ("Cierra el círculo narrativo: referencia algo del inicio y deja una reflexión concreta. " if is_doc_long else "")
        + "JSON válido, nada más."
    )


def _infer_target_seconds_from_brief(brief: str) -> int:
    _ = brief
    return int(settings.custom_min_video_sec)


def _target_word_range(min_seconds: int) -> tuple[int, int]:
                                                        
    sec = max(30, int(min_seconds))
    base = int(140 * (sec / 60.0))
    return max(90, int(base * 0.95)), max(120, int(base * 1.35))


def _looks_like_full_story_text(brief: str) -> bool:
    b = (brief or "").strip()
    if not b:
        return False
    w = _words(b)
    if w >= 220:
        return True
    if w >= 120 and ("\n\n" in b or b.count(".") >= 8):
        return True
    return False


def _is_transformation_request(brief: str) -> bool:
    b = (brief or "").strip().lower()
    if not b:
        return False
    head = b[:320]
    verbs = (
        "convierte",
        "transforma",
        "reescribe",
        "adapta",
        "resume",
        "traduce",
    )
    if not any(v in head for v in verbs):
        return False
    targets = ("historia", "guion", "narrativo", "video", "español", "espanol")
    return any(t in head for t in targets)


def _extract_source_text_from_transform_brief(brief: str) -> str:
    b = (brief or "").strip()
    if not b:
        return ""
    if not _is_transformation_request(b):
        return b

    m = re.search(r"\n\s*\n+", b)
    if m:
        tail = b[m.end():].strip()
        if _words(tail) >= 40:
            return tail

    lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
    if len(lines) >= 2:
        tail2 = "\n".join(lines[1:]).strip()
        if _words(tail2) >= 40:
            return tail2
    return b


def _build_preserved_text_plan(brief: str, *, target_seconds: int) -> VideoPlan:
    source_text = (brief or "").strip()
    if not source_text:
        return _build_debug_quick_plan("", target_seconds=max(5, int(target_seconds or 60)))

    if int(target_seconds) >= 300:
        seg_target = 12
    else:
        seg_target = max(6, min(14, int(math.ceil(_words(source_text) / 55.0))))

    seg_texts = _segmentar_texto_en_prompts(source_text, ["x"] * seg_target)
    if not seg_texts:
        seg_texts = [source_text]

    segmentos: list[dict[str, Any]] = []
    for txt in seg_texts:
        q = _build_better_image_query(
            base_query="story scene portrait",
            text_es=txt,
            note="",
            brief=source_text[:400],
        )
        ip = f"cinematic documentary photo, {q}".strip()
        note = (txt[:140] + "...") if len(txt) > 140 else txt
        segmentos.append({
            "text_es": txt,
            "image_query": q,
            "image_prompt": ip,
            "note": note,
        })

    first_sentence = re.split(r"(?<=[.!?¡¿])\s+", source_text)[0].strip()
    hook = first_sentence if first_sentence else "Historia completa sin recortes"
    title = hook[:80] if hook else "Historia personalizada"

    script_es = source_text
    prompts_final = [s.get("image_prompt") or s.get("image_query") or "photo" for s in segmentos]
    timeline_data = _generar_timeline(prompts_final, _estimar_segundos(script_es))

    return VideoPlan(
        brief=brief,
        target_seconds=int(target_seconds),
        title_es=title or "Video personalizado",
        youtube_title_es=title or "Video personalizado",
        hook_es=hook,
        script_es=script_es,
        segments=[ScriptSegment(**s) for s in segmentos],
        prompts=prompts_final,
        timeline=[TimelineItem(**t) for t in timeline_data] if timeline_data else [],
        contexto_web="",
        raw_plan={"preserved_source_text": True},
    )


def _build_debug_quick_plan(brief: str, *, target_seconds: int) -> VideoPlan:
    sec = max(3, int(target_seconds or 5))
    brief_clean = re.sub(r"\s+", " ", (brief or "").strip())
    seed = brief_clean[:120] if brief_clean else "prueba de render"
    title = f"Debug rápido {sec}s"
    hook = "Prueba rápida de pipeline completo"
    text_es = f"Prueba rápida: {seed}. Render de diagnóstico completado."
    image_query = "dramatic portrait close up"
    image_prompt = "close-up portrait, cinematic light, realistic"
    note = "Prueba debug: validar generación rápida de plan, TTS, imagen y render."

    segmentos = [{
        "text_es": text_es,
        "image_query": image_query,
        "image_prompt": image_prompt,
        "note": note,
    }]
    timeline = [{"prompt": image_prompt, "start": 0.0, "end": float(sec)}]

    return VideoPlan(
        brief=brief,
        target_seconds=sec,
        title_es=title,
        youtube_title_es=title,
        hook_es=hook,
        script_es=text_es,
        segments=[ScriptSegment(**s) for s in segmentos],
        prompts=[image_prompt],
        timeline=[TimelineItem(**t) for t in timeline],
        contexto_web="",
        raw_plan={"debug_mode": True, "debug_target_seconds": sec},
    )


def _segmentar_texto_en_prompts(historia: str, prompts: List[str]) -> List[str]:
    texto = (historia or "").strip()
    prompts_limpios = [p for p in (prompts or []) if p]
    if not texto:
        return []
    if not prompts_limpios:
        return [texto]

    oraciones = re.split(r"(?<=[.!?¡¿])\s+", texto)
    oraciones = [o.strip() for o in oraciones if o.strip()]
    if not oraciones:
        return [texto]

    objetivo = min(len(prompts_limpios), max(1, len(oraciones)))
    chunk = max(1, math.ceil(len(oraciones) / objetivo))
    segmentos: List[str] = []
    for i in range(0, len(oraciones), chunk):
        segmentos.append(" ".join(oraciones[i : i + chunk]).strip())

    segmentos = [s for s in segmentos if s]
    if len(segmentos) > objetivo:
        segmentos = segmentos[:objetivo]

    return segmentos


def _generar_timeline(prompts: List[str], dur_est: float) -> List[Dict[str, Any]]:
    prompts = [p for p in (prompts or []) if p]
    if not prompts:
        return []
    dur = max(5.0, dur_est or 0)
    n = max(len(prompts), math.ceil(dur / 10.0))
    seg_dur = max(5.0, min(10.0, dur / n))
    timeline = []
    start = 0.0
    for i in range(n):
        end = start + seg_dur
        prompt = prompts[i % len(prompts)]
        timeline.append({"prompt": prompt, "start": round(start, 2), "end": round(end, 2)})
        start = end
    return timeline


def _extract_json_value(raw: str) -> Any:
    return llm_client._extract_json_value(raw)

def _extract_json_object(raw: str) -> Dict[str, Any]:
    val = llm_client._extract_json_value(raw)
    if isinstance(val, dict):
        return val
    raise ValueError("Not a JSON object")

def _extract_json_array(raw: str) -> List[Any]:
    val = llm_client._extract_json_value(raw)
    if isinstance(val, list):
        return val
    raise ValueError("Not a JSON array")

def _ollama_generate(prompt: str, *, temperature: float = 0.65, max_tokens: int = 900, model: str | None = None) -> str:
    return llm_client.generate(prompt, temperature=temperature, max_tokens=max_tokens, model=model)

def _ollama_generate_with_timeout(
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    min_ctx: int = None,
    model: str | None = None,
) -> str:
    return llm_client.generate(prompt, temperature=temperature, max_tokens=max_tokens, timeout_sec=timeout_sec, model=model, min_ctx=min_ctx)

def _ollama_generate_json(
    prompt: str,
    *,
    temperature: float = 0.65,
    max_tokens: int = 900,
    model: str | None = None,
) -> Dict[str, Any]:
    return llm_client.generate_json(prompt, temperature=temperature, max_tokens=max_tokens, model=model)

def _ollama_generate_json_with_timeout(
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    min_ctx: int = None,
    model: str | None = None,
) -> Dict[str, Any]:
    return llm_client.generate_json(prompt, temperature=temperature, max_tokens=max_tokens, timeout_sec=timeout_sec, model=model, min_ctx=min_ctx)



def _query_matches(query: str, title: str, url: str) -> bool:
    q_words = [w.lower() for w in re.split(r"[^\w]+", query) if len(w) >= 3]
    if not q_words:
        return False
    hay = 0
    text = f"{title} {url}".lower()
    anchor_terms = {"neuralink", "tesla", "elon", "musk"}
    implant_terms = {"implant", "implante", "chip", "electrode", "electrodo", "n1", "thread", "hilo", "telepathy"}
    query_has_implant_intent = ("neuralink" in q_words) and any(t in q_words for t in implant_terms)
    if query_has_implant_intent:
        if ("neuralink" in text) and any(t in text for t in implant_terms):
            return True
        # Si pides implante/chip, descarta resultados de marca/persona sin señales de hardware.
        if ("neuralink" in text) and (not any(t in text for t in implant_terms)):
            return False
    for a in anchor_terms:
        if a in q_words and a in text:
            return True
    for w in q_words:
        if w in text:
            hay += 1
    # Mantiene relevancia sin bloquear consultas largas; pide 1-2 coincidencias útiles.
    required = max(1, min(2, len(q_words) // 2))
    return hay >= required


def _simplify_query(query: str) -> str:
    query = _sanitize_image_query(query)
    query = re.sub(r"\s+", " ", query)
    return query[:140]


def _wikimedia_image_url(query: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 20,
        "gsrnamespace": 6,
        "gsrsort": "relevance",
        "iiprop": "url",
        "iiurlwidth": 1600,
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
    except Exception as e:
        print(f"[IMG-WEB] Wikimedia fallo: {e}")
        return None

    for page in pages.values():
        title = str((page or {}).get("title") or "Wikimedia Commons")
        info = page.get("imageinfo")
        if not info:
            continue
        entry = info[0]
        mime = entry.get("mime", "")
        url = entry.get("responsiveUrls", {}).get("1600") or entry.get("url")
        if not url:
            continue
                                                                 
        if mime and not mime.lower().startswith("image/"):
            continue
        if any(url.lower().endswith(ext) for ext in (".pdf", ".svg", ".djvu")):
            continue
        if not _query_matches(query, title, str(url)):
            continue
        if _query_has_neuralink_implant_intent(query) and (not _candidate_matches_neuralink_implant(title, str(url))):
            continue
        return url
    return None


def _buscar_url_imagen(query: str) -> str | None:
    query_main = _simplify_query(query)
    return _wikimedia_image_url(query_main)


def _descargar_imagen(url: str, carpeta: str, idx: int) -> str | None:
    ext = "jpg"
    for cand in [".jpg", ".jpeg", ".png", ".webp", ".avif"]:
        if cand in (url or "").lower():
            ext = cand.replace(".", "")
            break
    os.makedirs(carpeta, exist_ok=True)
    dst = os.path.join(carpeta, f"img_{idx}.{ext}")
    return _descargar_imagen_a_archivo(url, dst)


def _request_headers_for(url: str, *, attempt: int, accept_header: str, referer: str = "") -> dict[str, str]:
    idx = max(0, int(attempt) - 1) % len(_USER_AGENTS)
    headers = {
        "User-Agent": _USER_AGENTS[idx],
        "Accept": accept_header,
        "Accept-Language": "en-US,en;q=0.8,es-ES,es;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "DNT": "1",
    }
    if referer:
        headers["Referer"] = referer
    try:
        host = (urlparse(url).netloc or "").lower()
        if _is_wikimedia_host(host):
            headers["User-Agent"] = (_WIKIMEDIA_UA or headers["User-Agent"])
    except Exception:
        pass
    return headers


def _extract_og_image_url(html_bytes: bytes, base_url: str) -> str | None:
    try:
        txt = (html_bytes or b"")[:500_000].decode("utf-8", errors="ignore")
        patterns = [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<img[^>]+src=["\']([^"\']+)["\']',
        ]
        for pat in patterns:
            m = re.search(pat, txt, flags=re.IGNORECASE)
            if not m:
                continue
            raw = (m.group(1) or "").strip()
            if not raw:
                continue
            cand = urljoin(base_url, raw)
            if cand.startswith("http://") or cand.startswith("https://"):
                return cand
    except Exception:
        return None
    return None


_WARNED_BROWSER_ANTIBOT = False


def _browser_extract_image_url(page_url: str, *, timeout_ms: int = 18000) -> str | None:
    global _WARNED_BROWSER_ANTIBOT
    if not IMG_ANTIBOT_BROWSER:
        return None
    if sync_playwright is None:
        if not _WARNED_BROWSER_ANTIBOT:
            _WARNED_BROWSER_ANTIBOT = True
            print("[IMG-WEB] ℹ️ IMG_ANTIBOT_BROWSER=1 pero Playwright no está instalado.")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(user_agent=random.choice(_USER_AGENTS), locale="en-US")
                page = context.new_page()
                page.goto(page_url, wait_until="domcontentloaded", timeout=int(timeout_ms))

                probes = [
                    "meta[property='og:image']",
                    "meta[name='twitter:image']",
                    "img[src]",
                ]
                for sel in probes:
                    el = page.query_selector(sel)
                    if not el:
                        continue
                    attr = "content" if "meta" in sel else "src"
                    raw = (el.get_attribute(attr) or "").strip()
                    if not raw:
                        continue
                    cand = urljoin(page.url or page_url, raw)
                    if cand.startswith("http://") or cand.startswith("https://"):
                        return cand
            finally:
                browser.close()
    except Exception:
        return None
    return None


def _download_candidate_urls(url: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(u: str | None) -> None:
        uu = str(u or "").strip()
        if not uu:
            return
        if not (uu.startswith("http://") or uu.startswith("https://")):
            return
        if uu in seen:
            return
        seen.add(uu)
        out.append(uu)

    _add(url)
    try:
        resolved = _resolve_wikimedia_thumb_via_api(url)
        _add(resolved)
    except Exception:
        pass

    if IMG_ANTIBOT_PROXY:
        try:
            p = urlparse(url)
            if p.scheme in {"http", "https"} and p.netloc:
                raw = f"{p.netloc}{p.path or ''}"
                if p.query:
                    raw += "?" + p.query
                from urllib.parse import quote

                q = quote(raw, safe="")
                _add(f"https://images.weserv.nl/?url={q}&w=1600&output=jpg")
                _add(f"https://wsrv.nl/?url={q}&w=1600&output=jpg")
        except Exception:
            pass

    return out


def _descargar_imagen_a_archivo(url: str, dst_path: str) -> str | None:
                                                                   
                                                                                               
    referer = ""
    try:
        p = urlparse(url)
        if p.scheme and p.netloc:
            referer = f"{p.scheme}://{p.netloc}/"
    except Exception:
        referer = ""

                                                                                  
    accept_avif = _can_convert_avif() or ALLOW_AVIF
    accept_header = "image/webp,image/apng,image/*,*/*;q=0.8"
    if accept_avif:
        accept_header = "image/avif," + accept_header
    resp = None
    used_url = url
    last_err: Exception | None = None
    candidate_urls = _download_candidate_urls(url)
    for candidate in candidate_urls:
        used_url = candidate
        for attempt in range(1, 4):
            try:
                headers = _request_headers_for(candidate, attempt=attempt, accept_header=accept_header, referer=referer)
                resp = requests.get(candidate, headers=headers, timeout=30)
                if resp.status_code in {401, 403, 404}:
                    resp.raise_for_status()
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = 2.5 * attempt
                    if retry_after:
                        try:
                            wait = max(wait, float(retry_after))
                        except Exception:
                            pass
                    time.sleep(min(30.0, wait + random.random()))
                    continue
                resp.raise_for_status()

                ct_now = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" in ct_now:
                    alt = _extract_og_image_url(resp.content or b"", candidate)
                    if alt and alt not in candidate_urls:
                        candidate_urls.append(alt)
                    if IMG_ANTIBOT_BROWSER:
                        alt_browser = _browser_extract_image_url(candidate)
                        if alt_browser and alt_browser not in candidate_urls:
                            candidate_urls.append(alt_browser)
                    resp = None
                    break

                break
            except Exception as e:
                last_err = e
                if attempt < 3:
                    time.sleep(min(8.0, 1.2 * attempt + random.random()))
                resp = None
        if resp is not None:
            break

    if resp is None:
        print(f"[IMG-WEB] Falló descarga: {last_err}")
        return None

    ct = (resp.headers.get("Content-Type") or "").lower()
    if ct and ("text/html" in ct or "application/pdf" in ct or "image/svg" in ct):
        print(f"[IMG-WEB] Contenido no-imagen ({ct}), descartado: {used_url}")
        return None

    if ("image/avif" in ct) and not (ALLOW_AVIF or _can_convert_avif()):
        print(f"[IMG-WEB] AVIF no soportado (Content-Type: {ct}), descartado: {used_url}")
        return None

                                                                                   
    try:
        head = (resp.content or b"")[:512].lstrip().lower()
        if head.startswith(b"<") and (b"<html" in head or b"doctype" in head or b"captcha" in head):
            print(f"[IMG-WEB] Respuesta parece HTML/captcha, descartado: {used_url}")
            return None
    except Exception:
        pass

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(resp.content)

        real_fmt = _detect_image_format(tmp_path)
        if real_fmt == "jpeg":
            real_ext = "jpg"
        else:
            real_ext = real_fmt

        final_path = dst_path
        base, ext = os.path.splitext(dst_path)
        ext = (ext or "").lower().lstrip(".")
        if real_ext and real_ext != ext:
            final_path = base + "." + real_ext

        os.replace(tmp_path, final_path)

        if not _es_imagen_valida(final_path):
            try:
                os.remove(final_path)
            except Exception:
                pass
            print(f"[IMG-WEB] Imagen corrupta, descartada: {used_url}")
            return None
                                                                           
        final_path = _convert_webp_to_png_if_needed(final_path)
        final_path = _convert_avif_to_png_if_needed(final_path)
        return final_path
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"[IMG-WEB] No se pudo guardar imagen: {e}")
        return None


def _convert_webp_to_png_if_needed(path: str) -> str:
    try:
        base, ext = os.path.splitext(path)
        if ext.lower() != ".webp":
            return path
        try:
            from PIL import Image
        except Exception:
                                                          
            return path
        new_path = base + ".png"
        with Image.open(path) as img:
            try:
                if getattr(img, "is_animated", False):
                    img.seek(0)
            except Exception:
                pass
                                               
            if img.mode not in ("RGBA", "LA"):
                img = img.convert("RGBA")
            img.save(new_path, format="PNG", optimize=True)
        try:
            os.remove(path)
        except Exception:
            pass
        return new_path
    except Exception as e:
                                                                  
        ff = _convert_to_png_with_ffmpeg(path)
        if ff:
            return ff
        print(f"[IMG-WEB] No se pudo convertir WEBP a PNG: {e}")
        return path


def _ffmpeg_exe_for_images() -> str | None:
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        return str(exe) if exe else None
    except Exception:
        return None


def _convert_to_png_with_ffmpeg(path: str) -> str | None:
    ffmpeg = _ffmpeg_exe_for_images()
    if not ffmpeg:
        return None

    base, _ext = os.path.splitext(path)
    out_path = base + ".png"
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-frames:v",
        "1",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(out_path) and os.path.getsize(out_path) >= 1024:
            try:
                os.remove(path)
            except Exception:
                pass
            return out_path
    except subprocess.CalledProcessError as e:
        if _DEBUG_IMG_VALIDATION:
            err = (e.stderr or "")[:800]
            print(f"[IMG-WEB] ffmpeg no pudo convertir a PNG: {err}")
    except Exception as e:
        if _DEBUG_IMG_VALIDATION:
            print(f"[IMG-WEB] ffmpeg no pudo convertir a PNG: {e}")
    return None


def _normalizar_imagen_descargada(path: str, content_type: str = "") -> str:
    try:
        if not path or not os.path.exists(path):
            return path

        ext = os.path.splitext(path)[1].lower()
        fmt = (_detect_image_format(path) or "").lower()
        ct = (content_type or "").lower()

                                                       
        if ext == ".avif" or "image/avif" in ct:
            return _convert_avif_to_png_if_needed(path)

        is_webp = ext == ".webp" or fmt == "webp" or "image/webp" in ct
        if not is_webp:
            return path

                                                   
        converted = _convert_webp_to_png_if_needed(path)
        if converted != path:
            return converted

                                                                      
        ff = _convert_to_png_with_ffmpeg(path)
        return ff or path
    except Exception:
        return path


def _can_convert_avif() -> bool:
    try:
                                                         
        import pillow_avif                
        from PIL import Image              
        return True
    except Exception:
        return False


def _convert_avif_to_png_if_needed(path: str) -> str:
    try:
        base, ext = os.path.splitext(path)
        if ext.lower() != ".avif":
            return path
        if not _can_convert_avif():
            return path
        from PIL import Image                

        new_path = base + ".png"
        with Image.open(path) as img:
            try:
                if getattr(img, "is_animated", False):
                    img.seek(0)
            except Exception:
                pass
            if img.mode not in ("RGBA", "LA"):
                img = img.convert("RGBA")
            img.save(new_path, format="PNG", optimize=True)
        try:
            os.remove(path)
        except Exception:
            pass
        return new_path
    except Exception as e:
        print(f"[IMG-WEB] No se pudo convertir AVIF a PNG: {e}")
        return path


def _es_imagen_valida(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < 1024:                                     
            return False

                                                                     
        try:
            from PIL import Image                              

            with Image.open(path) as img:
                img.load()
            return True
        except ImportError:
                                                
            return bool(_detect_image_format(path))
        except Exception:
                                                                           
                                                                                                          
            fmt = (_detect_image_format(path) or "").lower()
            return fmt in {"jpeg", "png", "gif", "bmp", "webp"}
    except Exception:
        return False


def _crear_placeholder_imagen(carpeta: str, seg_tag: str) -> str | None:
    os.makedirs(carpeta, exist_ok=True)

                                                   
    try:
        from PIL import Image                

        path = os.path.join(carpeta, f"{seg_tag}_chosen.png")
        img = Image.new("RGB", (768, 768), (18, 18, 18))
        img.save(path, format="PNG", optimize=True)
        if _es_imagen_valida(path):
            return path
    except Exception:
        pass

                                                                    
    try:
        path = os.path.join(carpeta, f"{seg_tag}_chosen.ppm")
        w, h = 768, 768
        header = f"P6\n{w} {h}\n255\n".encode("ascii")
                     
        pixel = bytes([18, 18, 18])
        with open(path, "wb") as f:
            f.write(header)
            f.write(pixel * (w * h))
        if _es_imagen_valida(path):
            return path
    except Exception:
        pass
    return None


def _es_placeholder_generado(path: str) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as im:
            rgb = im.convert("RGB")
            ex = rgb.getextrema()
        if not ex or len(ex) != 3:
            return False
        return all(lo == hi and lo <= 24 for (lo, hi) in ex)
    except Exception:
        return False


def _imagen_fingerprint(path: str) -> str | None:
    try:
        if not path or (not os.path.exists(path)):
            return None
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _recuperar_candidato_local_unico(carpeta: str, seg_tag: str, usados_fp: set[str]) -> str | None:
    cand_paths: list[str] = []
    try:
        cand_dir = os.path.join(carpeta, "candidates")
        if os.path.isdir(cand_dir):
            for n in sorted(os.listdir(cand_dir)):
                nlow = n.lower()
                if not n.startswith(f"{seg_tag}_"):
                    continue
                if not nlow.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    continue
                cand_paths.append(os.path.join(cand_dir, n))
    except Exception:
        pass

    for absp in cand_paths:
        if not _es_imagen_valida(absp):
            continue
        if _es_placeholder_generado(absp):
            continue
        fp = _imagen_fingerprint(absp)
        if fp and fp in usados_fp:
            continue
        ext = os.path.splitext(absp)[1].lower() or ".jpg"
        stable = os.path.join(carpeta, f"{seg_tag}_chosen{ext}")
        try:
            with open(absp, "rb") as src, open(stable, "wb") as dst:
                dst.write(src.read())
            fp2 = _imagen_fingerprint(stable)
            if fp2:
                usados_fp.add(fp2)
            return stable
        except Exception:
            if fp:
                usados_fp.add(fp)
            return absp
    return None


def _recuperar_candidato_local(carpeta: str, seg_tag: str) -> str | None:
    cand_paths: list[str] = []
    try:
        cand_dir = os.path.join(carpeta, "candidates")
        if os.path.isdir(cand_dir):
            for n in sorted(os.listdir(cand_dir)):
                nlow = n.lower()
                if not n.startswith(f"{seg_tag}_"):
                    continue
                if not nlow.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    continue
                cand_paths.append(os.path.join(cand_dir, n))
    except Exception:
        pass

    try:
        for n in sorted(os.listdir(carpeta)):
            if not n.startswith(f"{seg_tag}_cand_"):
                continue
            cand_paths.append(os.path.join(carpeta, n))
    except Exception:
        pass

    for absp in cand_paths:
        if not _es_imagen_valida(absp):
            continue
        if _es_placeholder_generado(absp):
            continue
        ext = os.path.splitext(absp)[1].lower() or ".jpg"
        stable = os.path.join(carpeta, f"{seg_tag}_chosen{ext}")
        try:
            with open(absp, "rb") as src, open(stable, "wb") as dst:
                dst.write(src.read())
            return stable
        except Exception:
            return absp
    return None


def _reusar_chosen_existente(carpeta: str, seg_tag_destino: str) -> str | None:
    try:
        candidatos: list[str] = []
        for n in sorted(os.listdir(carpeta)):
            nlow = n.lower()
            if not n.startswith("seg_"):
                continue
            if "_chosen" not in n:
                continue
            if not nlow.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                continue
            p = os.path.join(carpeta, n)
            if not _es_imagen_valida(p):
                continue
            if _es_placeholder_generado(p):
                continue
            candidatos.append(p)

        if not candidatos:
            return None

        idx = 0
        try:
            m = re.search(r"seg_(\d+)", str(seg_tag_destino or "").lower())
            if m:
                idx = max(0, int(m.group(1)) - 1)
        except Exception:
            idx = 0

        src = candidatos[idx % len(candidatos)]
        ext = os.path.splitext(src)[1].lower() or ".jpg"
        dst = os.path.join(carpeta, f"{seg_tag_destino}_chosen{ext}")
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
        return dst
    except Exception:
        return None


def _intentar_fallback_comfy(carpeta: str, seg_tag: str, prompt_hint: str) -> str | None:
    if not _COMFY_FALLBACK_ON_MISSING:
        return None
    try:
        prompt = (prompt_hint or "").strip() or "cinematic realistic photo"
        tmp_dir = os.path.join(carpeta, "_fallback_comfy", seg_tag)
        os.makedirs(tmp_dir, exist_ok=True)
        rutas = image_downloader.descargar_imagenes_desde_prompts(tmp_dir, [prompt], dur_audio=None)
        if not rutas:
            return None
        src = os.path.abspath(rutas[0])
        if not _es_imagen_valida(src):
            return None
        ext = os.path.splitext(src)[1].lower() or ".png"
        dst = os.path.join(carpeta, f"{seg_tag}_chosen{ext}")
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
        return dst
    except Exception:
        return None


def _limpiar_chosen_obsoletos(carpeta: str) -> int:
    try:
        names = sorted(os.listdir(carpeta))
    except Exception:
        return 0

    per_seg: dict[str, list[str]] = {}
    for n in names:
        low = n.lower()
        if not low.startswith("seg_"):
            continue
        if "_chosen" not in low:
            continue
        if not low.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            continue
        seg = n.split("_chosen", 1)[0]
        per_seg.setdefault(seg, []).append(os.path.join(carpeta, n))

    removed = 0
    for _, paths in per_seg.items():
        non_placeholder = [p for p in paths if _es_imagen_valida(p) and not _es_placeholder_generado(p)]
        if not non_placeholder:
            continue
        for p in paths:
            if p in non_placeholder:
                continue
            if not _es_imagen_valida(p):
                continue
            if not _es_placeholder_generado(p):
                continue
            try:
                os.remove(p)
                removed += 1
            except Exception:
                pass
    return removed


def _buscar_ddg_imagenes(query: str, *, max_results: int = 8) -> list[Tuple[str, str]]:
    global _DDG_DISABLED_RUNTIME
    if _DDG_DISABLED_RUNTIME:
        return []
    if not DDGS:
        return []

    try:
        kwargs = {
            "max_results": max_results,
            "safesearch": "off",
        }
        if _DDG_BACKEND == "ddgs":
            kwargs["backend"] = settings.ddg_images_backend

        def _run() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.images(query, **kwargs))

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(_run)
        try:
            res = fut.result(timeout=max(5.0, float(settings.ddg_search_timeout_sec)))
        finally:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    except concurrent.futures.TimeoutError:
        _DDG_DISABLED_RUNTIME = True
        print(f"[IMG-DDG] Timeout buscando imágenes (>{settings.ddg_search_timeout_sec:.0f}s): {query[:120]}")
        print("[IMG-DDG] Desactivado en esta ejecución por timeout (fallback a otras fuentes)")
        return []
    except Exception as e:
        low = str(e).lower()
        if "403" in low or "forbidden" in low:
            _DDG_DISABLED_RUNTIME = True
            print("[IMG-DDG] Desactivado en esta ejecución por bloqueos 403 (fallback a otras fuentes)")
        print(f"[IMG-DDG] Falló búsqueda ({_DDG_BACKEND}): {e}")
        return []

    if not res:
        return []
    urls: list[Tuple[str, str]] = []
    for r in res:
        url = r.get("image") or ""
        title = r.get("title") or ""
        if not url:
            continue
        if _is_blocked_image_host(url):
            continue
                                                                                      
                                                              
        ext_path = url.split("?")[0].lower()
        if any(ext_path.endswith(suf) for suf in (".pdf", ".svg", ".djvu")):
            continue
        if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
            continue
        urls.append((url, title))
    return urls


def _buscar_wikimedia_imagenes(query: str, *, max_results: int = 8) -> list[Tuple[str, str]]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "generator": "search",
        "gsrsearch": _simplify_query(query),
        "gsrlimit": max(1, min(50, int(max_results) * 3)),
        "gsrnamespace": 6,
        "gsrsort": "relevance",
        "iiprop": "url|mime",
        "iiurlwidth": 1600,
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers={"User-Agent": (_WIKIMEDIA_UA or USER_AGENT)}, timeout=20)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        pages = (data.get("query") or {}).get("pages") or {}
    except Exception as e:
        print(f"[IMG-WEB] Wikimedia fallo: {e}")
        return []

    out: list[Tuple[str, str]] = []
    seen: set[str] = set()
    for page in pages.values():
        if len(out) >= int(max_results):
            break
        if not isinstance(page, dict):
            continue
        info = page.get("imageinfo")
        if not info:
            continue
        entry = info[0] if isinstance(info, list) and info else None
        if not isinstance(entry, dict):
            continue
        mime = (entry.get("mime") or "").lower()
        url = (entry.get("responsiveUrls") or {}).get("1600") or entry.get("url")
        if not url:
            continue
        if mime and not mime.startswith("image/"):
            continue
        if any(str(url).lower().endswith(ext) for ext in (".pdf", ".svg", ".djvu")):
            continue
        if _is_blocked_image_host(str(url)):
            continue
        title = str(page.get("title") or "Wikimedia Commons")
        if not _query_matches(query, title, str(url)):
            continue
        if not _animal_query_relevance_ok(query, title, str(url)):
            continue
        if str(url) in seen:
            continue
        seen.add(str(url))
        out.append((str(url), title))
    return out


def _buscar_openverse_imagenes(query: str, *, max_results: int = 8) -> list[Tuple[str, str]]:
    global _OPENVERSE_DISABLED_RUNTIME
    if _OPENVERSE_DISABLED_RUNTIME:
        return []

    q = _simplify_query(query)
    if not q:
        return []

    params = {
        "q": q,
        "page_size": max(1, min(50, int(max_results) * 3)),
        "mature": "false",
    }
    headers = {"User-Agent": USER_AGENT}

    results = []
    last_err: Exception | None = None
    for api in OPENVERSE_APIS:
        try:
            resp = requests.get(api, params=params, headers=headers, timeout=max(5.0, float(settings.openverse_timeout_sec)))
            if resp.status_code == 401:
                _OPENVERSE_DISABLED_RUNTIME = True
                print("[IMG-OV] Openverse 401 Unauthorized; desactivado en esta ejecución")
                return []
            if resp.status_code == 429:
                continue
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            results = data.get("results") or []
            if isinstance(results, list):
                break
        except Exception as e:
            last_err = e
            continue
    if not isinstance(results, list) or not results:
        if last_err is not None:
            print(f"[IMG-OV] Openverse fallo: {last_err}")
        return []

    out: list[Tuple[str, str]] = []
    seen: set[str] = set()
    for r in results:
        if len(out) >= int(max_results):
            break
        if not isinstance(r, dict):
            continue
        title = str(r.get("title") or "Openverse")
        url = (
            r.get("thumbnail")
            or r.get("url")
            or r.get("image")
            or r.get("thumbnail_url")
        )
        if not url:
            continue
        url = str(url)
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if any(url.lower().endswith(ext) for ext in (".pdf", ".svg", ".djvu")):
            continue
        if _is_blocked_image_host(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((url, title))
    return out


def _buscar_flickr_imagenes(query: str, *, max_results: int = 8) -> list[Tuple[str, str]]:
    q = _simplify_query(query)
    if not q:
        return []

    params = {
        "format": "json",
        "nojsoncallback": "1",
        "lang": "en-us",
        "tagmode": "all",
        "tags": ",".join(_extract_keywords(q, max_keywords=6) or q.split()[:4]),
    }
    try:
        resp = requests.get(
            FLICKR_FEED_API,
            params=params,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/plain,*/*"},
            timeout=max(8.0, float(settings.openverse_timeout_sec)),
        )
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        items = data.get("items") or []
        if not isinstance(items, list):
            return []
    except Exception as e:
        print(f"[IMG-FLICKR] Flickr fallo: {e}")
        return []

    out: list[Tuple[str, str]] = []
    seen: set[str] = set()
    for it in items:
        if len(out) >= int(max_results):
            break
        if not isinstance(it, dict):
            continue
        media = it.get("media") or {}
        url = str(media.get("m") or "").strip()
        if not url:
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if _is_blocked_image_host(url):
            continue
        title = str(it.get("title") or "Flickr")
        if not _query_matches(query, title, url):
            continue
        if not _animal_query_relevance_ok(query, title, url):
            continue
        if any(url.lower().endswith(ext) for ext in (".svg", ".pdf", ".djvu")):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((url, title))
    return out


def _buscar_bing_imagenes(query: str, *, max_results: int = 8) -> list[Tuple[str, str]]:
    q = _simplify_query(query)
    if not q:
        return []

    params = {
        "q": q,
        "form": "HDRSC2",
        "first": "1",
        "tsc": "ImageBasicHover",
    }
    try:
        resp = requests.get(
            BING_IMG_SEARCH_API,
            params=params,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=max(8.0, float(settings.openverse_timeout_sec)),
        )
        resp.raise_for_status()
        page_html = resp.text or ""
    except Exception as e:
        print(f"[IMG-BING] Bing fallo: {e}")
        return []

    out: list[Tuple[str, str]] = []
    seen: set[str] = set()

    payloads = re.findall(r'\bm="([^"]{80,3500})"', page_html)
    for raw_payload in payloads:
        if len(out) >= int(max_results):
            break
        try:
            txt = html.unescape(raw_payload)
            data = json.loads(txt)
            if not isinstance(data, dict):
                continue
            url = str(data.get("murl") or "").strip()
            title = str(data.get("t") or data.get("desc") or "Bing Images").strip()
        except Exception:
            continue

        if not url:
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if any(url.lower().endswith(ext) for ext in (".svg", ".pdf", ".djvu")):
            continue
        if _is_blocked_image_host(url):
            continue
        if not _query_matches(query, title, url):
            continue
        if not _animal_query_relevance_ok(query, title, url):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((url, title or "Bing Images"))

    return out


def _buscar_imagenes_multi(query: str, *, max_results: int = 8) -> list[Tuple[str, str, str]]:
    sources = settings.img_sources_list
    if not sources:
        sources = ["ddg", "openverse", "wikimedia", "flickr", "sites"]

    out: list[Tuple[str, str, str]] = []
    seen: set[str] = set()

    def _add(items: list[Tuple[str, str]], src: str) -> None:
        nonlocal out
        for u, t in items:
            if len(out) >= int(max_results):
                return
            if not u or u in seen:
                continue
            seen.add(u)
            out.append((u, t, src))

    def _buscar_en_sitios(keys: list[str]) -> list[Tuple[str, str, str]]:
        out_sites: list[Tuple[str, str, str]] = []
        seen_sites: set[str] = set()
        if _DDG_DISABLED_RUNTIME or (DDGS is None):
            return out_sites
        if not keys:
            keys = list(_SITE_IMAGE_SOURCES.keys())
        per_site_max = 2 if int(max_results) > 3 else 1
        for k in keys:
            dom = _SITE_IMAGE_SOURCES.get(k)
            if not dom:
                continue
            q_site = f"site:{dom} {query}"
            res = _buscar_ddg_imagenes(q_site, max_results=per_site_max)
            for u, t in res:
                if u in seen_sites:
                    continue
                seen_sites.add(u)
                out_sites.append((u, t, k))
                if len(out_sites) >= int(max_results):
                    return out_sites
        return out_sites

    for src in sources:
        if len(out) >= int(max_results):
            break
        if src in {"ddg", "duckduckgo"}:
            _add(_buscar_ddg_imagenes(query, max_results=max_results), "ddg")
        elif src in {"wikimedia", "wiki"}:
            _add(_buscar_wikimedia_imagenes(query, max_results=max_results), "wikimedia")
        elif src in {"openverse", "ov"}:
            _add(_buscar_openverse_imagenes(query, max_results=max_results), "openverse")
        elif src in {"flickr", "flickr_public"}:
            _add(_buscar_flickr_imagenes(query, max_results=max_results), "flickr")
        elif src in {"bing", "bing_images"}:
            _add(_buscar_bing_imagenes(query, max_results=max_results), "bing")
        elif src in {"sites", "pngsites", "stocksites"}:
            for u, t, sname in _buscar_en_sitios(list(_SITE_IMAGE_SOURCES.keys())):
                _add([(u, t)], sname)
        elif src in _SITE_IMAGE_SOURCES:
            for u, t, sname in _buscar_en_sitios([src]):
                _add([(u, t)], sname)
        else:
            continue

    # Rescate: aunque IMG_SOURCES venga limitado o una fuente falle (p.ej. Openverse 401),
    # intenta fuentes estables para no perder candidatos de calidad.
    if len(out) < int(max_results):
        rescue_order = ["wikimedia", "flickr", "bing"]
        for src in rescue_order:
            if len(out) >= int(max_results):
                break
            if src in {"wikimedia", "wiki"}:
                _add(_buscar_wikimedia_imagenes(query, max_results=max_results), "wikimedia")
            elif src in {"flickr", "flickr_public"}:
                _add(_buscar_flickr_imagenes(query, max_results=max_results), "flickr")
            elif src in {"bing", "bing_images"}:
                _add(_buscar_bing_imagenes(query, max_results=max_results), "bing")

    # Último rescate (si DDG está disponible) con consulta por sitios conocidos.
    allow_site_rescue = any(s in {"sites", "pngsites", "stocksites"} for s in sources)
    if allow_site_rescue and len(out) < int(max_results):
        for u, t, sname in _buscar_en_sitios(list(_SITE_IMAGE_SOURCES.keys())):
            _add([(u, t)], sname)
            if len(out) >= int(max_results):
                break

    return out


def _build_query_variants(q_search: str, note: str, original_q: str) -> list[str]:
    base = _sanitize_image_query(q_search)
    out: list[str] = []

    def _add(q: str) -> None:
        qq = _sanitize_image_query(q)
        if qq and qq not in out:
            out.append(qq)

    _add(base)
    if note:
        _add(f"{base} {note}")
    if original_q:
        _add(original_q)

    translated_terms = _expand_query_terms_es_en(f"{base} {note} {original_q}")
    if translated_terms:
        _add(" ".join(translated_terms))
        _add(f"{' '.join(translated_terms)} realistic photo")
        _add(f"{' '.join(translated_terms)} close up portrait")

    kws = _extract_keywords(f"{base} {note} {original_q}", max_keywords=8)
    if kws:
        _add(" ".join(kws[:6]))
        if len(kws) >= 3:
            _add(" ".join(kws[:3]))

    # Variantes fotográficas para mejorar recall sin bajar demasiado calidad.
    _add(f"{base} documentary photo")
    _add(f"{base} realistic photo")

    if _query_has_neuralink_implant_intent(f"{base} {note} {original_q}"):
        _add("neuralink implant n1 device close up")
        _add("neuralink chip electrode thread brain implant")
        _add("brain computer interface neuralink implant hardware")
        _add("neuralink n1 implant chip close-up macro -elon -musk -presentation -stage")
        _add("neuralink electrode threads implant surgery device photo")

    return out[:10]


def _puntuar_con_moondream(path: str, query: str, *, note: str = "") -> int:
    global _VISION_AVAILABLE
    if _VISION_AVAILABLE is False:
        return 1

    try:
        score = vision_llava_phi3.score_image(
            path,
            query,
            note=note,
            model=settings.vision_model,
            timeout_sec=settings.vision_timeout_sec,
            retries=settings.vision_retries,
        )
        _VISION_AVAILABLE = True
        return int(score)
    except Exception:
        _VISION_AVAILABLE = False
        return 1


def descargar_mejores_imagenes_ddg(
    carpeta: str,
    queries: List[str],
    notes: List[str] | None = None,
    *,
    max_per_query: int = 8,
    segment_numbers: List[int] | None = None,
) -> tuple[List[str], List[Dict[str, Any]]]:
    rutas: List[str] = []
    meta_all: List[Dict[str, Any]] = []
    notes = notes or [""] * len(queries)

    cand_dir = os.path.join(carpeta, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

                                                                                       
    if max_per_query == 8 and settings.custom_img_max_per_query != 8:
        max_per_query = max(1, int(settings.custom_img_max_per_query))

                                                                    
    if settings.custom_img_quality in {"high", "alta"}:
        max_per_query = max(max_per_query, 14)
    elif settings.custom_img_quality in {"best", "max", "maxima", "máxima"}:
        max_per_query = max(max_per_query, 24)

    if segment_numbers is not None and len(segment_numbers) != len(queries):
        raise ValueError("segment_numbers debe tener la misma longitud que queries")

    def _uniq_candidates(cands: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        seen_u = set()
        out_u: List[Tuple[str, str, str]] = []
        for u, t, src in cands:
            if not u:
                continue
            if u in seen_u:
                continue
            seen_u.add(u)
            out_u.append((u, t, src))
        return out_u

    used_urls: set[str] = set()
    used_fingerprints: set[str] = set()
    try:
        for n in os.listdir(carpeta):
            low = n.lower()
            if not low.startswith("seg_"):
                continue
            if "_chosen" not in low:
                continue
            if not low.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                continue
            p = os.path.join(carpeta, n)
            if (not _es_imagen_valida(p)) or _es_placeholder_generado(p):
                continue
            fp0 = _imagen_fingerprint(p)
            if fp0:
                used_fingerprints.add(fp0)
    except Exception:
        pass

    for idx, q in enumerate(queries):
        note = notes[idx] if idx < len(notes) else ""
        q_search = _sanitize_image_query(q)
        strict_neuralink_implant = _query_has_neuralink_implant_intent(q)
        min_score_effective = max(1, int(settings.custom_min_img_score))
        if strict_neuralink_implant:
            min_score_effective = max(min_score_effective, 3)

        seg_n = (segment_numbers[idx] if segment_numbers is not None else (idx + 1))
        is_hook = int(seg_n) <= int(settings.custom_hook_segments)

                                                                                                                  
        local_max = int(max_per_query)
        if is_hook:
            local_max = max(local_max, int(max_per_query) + max(0, int(settings.custom_hook_extra_candidates)))

        candidatos = _buscar_imagenes_multi(q_search, max_results=local_max)

        # Reintento inteligente con variantes de query si no llegaron candidatos.
        if not candidatos:
            merged_retry: List[Tuple[str, str, str]] = []
            for qv in _build_query_variants(q_search, note, q):
                merged_retry.extend(_buscar_imagenes_multi(qv, max_results=max(8, local_max // 2)))
                if len(merged_retry) >= int(local_max):
                    break
            candidatos = _uniq_candidates(merged_retry)[: int(local_max)]

        if is_hook:
            variants = [
                q_search,
                f"{q_search} close up photo",
                f"{q_search} dramatic lighting photo",
            ]
            merged: List[Tuple[str, str, str]] = []
            for v in variants:
                merged.extend(_buscar_imagenes_multi(v, max_results=max(8, local_max // 2)))
            candidatos = _uniq_candidates(merged)[: max(local_max, len(candidatos))]

                                                                                   
        if settings.enable_text_rank and candidatos:
            kw = _extract_keywords(f"{q} {note}", max_keywords=10)
            if kw:
                candidatos = sorted(
                    candidatos,
                    key=lambda ut: _candidate_text_relevance(ut[0], ut[1], keywords=kw),
                    reverse=True,
                )
                candidatos = candidatos[: int(local_max)]

        if candidatos and _query_has_neuralink_implant_intent(q):
            fuertes = [
                (u, t, src)
                for (u, t, src) in candidatos
                if _candidate_matches_neuralink_implant(t, u)
            ]
            if fuertes:
                candidatos = fuertes[: int(local_max)]
                                                                     
        if candidatos:
            candidatos = [(u, t, src) for (u, t, src) in candidatos if u not in used_urls]

        if not candidatos:
                                                                                  
            wiki_url = _wikimedia_image_url(_simplify_query(q_search))
            if wiki_url:
                seg_tag = f"seg_{int(seg_n):02d}"
                dst = os.path.join(cand_dir, f"{seg_tag}_wiki_01.jpg")
                saved = _descargar_imagen_a_archivo(wiki_url, dst)
                if saved:
                                                                   
                    ext = os.path.splitext(saved)[1].lower() or ".jpg"
                    stable = os.path.join(carpeta, f"{seg_tag}_chosen{ext}")
                    try:
                        if os.path.abspath(saved) != os.path.abspath(stable):
                            with open(saved, "rb") as src, open(stable, "wb") as dstf:
                                dstf.write(src.read())
                        saved = stable
                    except Exception:
                        pass

                    used_urls.add(wiki_url)
                    rutas.append(saved)
                    meta_all.append({
                        "query": q,
                        "note": note,
                        "candidates": [],
                        "selected": {
                            "candidate_index": 1,
                            "score": int(min_score_effective),
                            "url": wiki_url,
                            "title": "Wikimedia Commons",
                            "path": os.path.relpath(saved, carpeta).replace("\\", "/"),
                        },
                        "fallback": "wikimedia",
                    })
                    continue

            meta_all.append({"query": q, "note": note, "candidates": [], "selected": None})
            continue

        seg_tag = f"seg_{int(seg_n):02d}"
        cand_meta: List[Dict[str, Any]] = []
        best_score = -1
        best_path = None
        best_url = ""
        best_title = ""
        best_k = None

        for k, (url, title, src) in enumerate(candidatos, start=1):
                                                  
            parsed = url.split("?")[0].lower()
            ext = ".jpg"
            for suf in (".jpg", ".jpeg", ".png", ".webp"):
                if parsed.endswith(suf):
                    ext = ".jpg" if suf == ".jpeg" else suf
                    break
            dst = os.path.join(cand_dir, f"{seg_tag}_{k:02d}{ext}")
            saved = _descargar_imagen_a_archivo(url, dst)
            if not saved:
                continue

            fp = _imagen_fingerprint(saved)
            if fp and fp in used_fingerprints:
                continue

            try:
                score_note = (note or "").strip()
                if is_hook:
                    score_note = (
                        "HOOK (primeros segundos): elige la imagen MÁS llamativa para retener audiencia. "
                        "Prefiere close-up, alto contraste, emoción/acción, objeto icónico real; evita texto/diagramas/AI. "
                        + (score_note if score_note else "")
                    ).strip()
                score = _puntuar_con_moondream(saved, q, note=score_note)
            except Exception as e:
                print(f"[IMG-WEB] ❌ No se pudo puntuar candidato {k} de '{q}': {e}")
                score = 1

            cand_meta.append({
                "candidate_index": k,
                "url": url,
                "title": title,
                "source": src,
                "path": os.path.relpath(saved, carpeta).replace("\\", "/"),
                "score": int(score),
            })

            if score > best_score:
                best_score = score
                best_path = saved
                best_url = url
                best_title = title
                best_k = k

            if best_score >= 5:
                break

                                                                                                  
        if best_path is None:
            wiki_url = _wikimedia_image_url(_simplify_query(q_search))
            if wiki_url:
                dst = os.path.join(cand_dir, f"{seg_tag}_wiki_01.jpg")
                saved = _descargar_imagen_a_archivo(wiki_url, dst)
                if saved:
                    best_path = saved
                    best_url = wiki_url
                    best_title = "Wikimedia Commons"
                    best_score = max(1, int(settings.custom_min_img_score))
                    best_k = 1

                                                                                                         
        if is_hook and best_score < int(settings.custom_hook_min_img_score):
            extra_max = max(local_max, int(max_per_query) + int(settings.custom_hook_extra_candidates) + 12)
            variants2 = [
                q_search,
                f"{q_search} close up",
                f"{q_search} high contrast photo",
                f"{q_search} cinematic still photo",
            ]
            seen = {m.get("url") for m in cand_meta if isinstance(m, dict)}
            merged2: List[Tuple[str, str, str]] = []
            for v in variants2:
                merged2.extend(_buscar_imagenes_multi(v, max_results=max(10, extra_max // 2)))
            nuevos2 = [(u, t, src) for (u, t, src) in _uniq_candidates(merged2) if u not in seen]
            for k2, (url, title, src) in enumerate(nuevos2, start=len(cand_meta) + 1):
                parsed = url.split("?")[0].lower()
                ext = ".jpg"
                for suf in (".jpg", ".jpeg", ".png", ".webp"):
                    if parsed.endswith(suf):
                        ext = ".jpg" if suf == ".jpeg" else suf
                        break
                dst = os.path.join(cand_dir, f"{seg_tag}_{k2:02d}{ext}")
                saved = _descargar_imagen_a_archivo(url, dst)
                if not saved:
                    continue
                fp = _imagen_fingerprint(saved)
                if fp and fp in used_fingerprints:
                    continue
                try:
                    score_note = (note or "").strip()
                    score_note = (
                        "HOOK (primeros segundos): elige la imagen MÁS llamativa para retener audiencia. "
                        "Prefiere close-up, alto contraste, emoción/acción, objeto icónico real; evita texto/diagramas/AI. "
                        + (score_note if score_note else "")
                    ).strip()
                    score = _puntuar_con_moondream(saved, q, note=score_note)
                except Exception as e:
                    print(f"[IMG-WEB] ❌ No se pudo puntuar candidato {k2} de '{q}': {e}")
                    score = 1
                cand_meta.append({
                    "candidate_index": k2,
                    "url": url,
                    "title": title,
                    "source": src,
                    "path": os.path.relpath(saved, carpeta).replace("\\", "/"),
                    "score": int(score),
                })
                if score > best_score:
                    best_score = score
                    best_path = saved
                    best_url = url
                    best_title = title
                    best_k = k2
                if best_score >= 5:
                    break

                                                                                                        
        if (best_score < int(settings.custom_min_img_score)) and (settings.custom_img_quality in {"high", "alta", "best", "max", "maxima", "máxima"}):
            extra = 32 if settings.custom_img_quality in {"best", "max", "maxima", "máxima"} else 20
            candidatos2 = _buscar_imagenes_multi(q_search, max_results=max(max_per_query, extra))
                                             
            seen = {m.get("url") for m in cand_meta if isinstance(m, dict)}
            nuevos = [(u, t, src) for (u, t, src) in candidatos2 if u not in seen]
            for k2, (url, title, src) in enumerate(nuevos, start=len(cand_meta) + 1):
                parsed = url.split("?")[0].lower()
                ext = ".jpg"
                for suf in (".jpg", ".jpeg", ".png", ".webp"):
                    if parsed.endswith(suf):
                        ext = ".jpg" if suf == ".jpeg" else suf
                        break
                dst = os.path.join(cand_dir, f"{seg_tag}_{k2:02d}{ext}")
                saved = _descargar_imagen_a_archivo(url, dst)
                if not saved:
                    continue
                fp = _imagen_fingerprint(saved)
                if fp and fp in used_fingerprints:
                    continue
                try:
                    score = _puntuar_con_moondream(saved, q, note=note)
                except Exception as e:
                    print(f"[IMG-WEB] ❌ No se pudo puntuar candidato {k2} de '{q}': {e}")
                    score = 1
                cand_meta.append({
                    "candidate_index": k2,
                    "url": url,
                    "title": title,
                    "source": src,
                    "path": os.path.relpath(saved, carpeta).replace("\\", "/"),
                    "score": int(score),
                })
                if score > best_score:
                    best_score = score
                    best_path = saved
                    best_url = url
                    best_title = title
                    best_k = k2
                if best_score >= 5:
                    break

        selected = None
        if best_path and best_score >= int(min_score_effective):
                                                           
            ext = os.path.splitext(best_path)[1].lower() or ".jpg"
            stable = os.path.join(carpeta, f"{seg_tag}_chosen{ext}")
            try:
                if os.path.abspath(best_path) != os.path.abspath(stable):
                    with open(best_path, "rb") as src, open(stable, "wb") as dst:
                        dst.write(src.read())
                best_path = stable
            except Exception:
                pass

            rutas.append(best_path)
            if best_url:
                used_urls.add(best_url)
            fp_best = _imagen_fingerprint(best_path)
            if fp_best:
                used_fingerprints.add(fp_best)
            selected = {
                "candidate_index": best_k,
                "score": int(best_score),
                "url": best_url,
                "title": best_title,
                "path": os.path.relpath(best_path, carpeta).replace("\\", "/"),
            }

        meta_all.append({
            "query": q,
            "note": note,
            "candidates": cand_meta,
            "selected": selected,
        })

    return rutas, meta_all


def _seleccionar_candidatos_interactivo(
    carpeta: str,
    segmentos: List[Dict[str, Any]],
    imagenes: List[str],
    img_meta: List[Dict[str, Any]],
    audios: List[str],
    duraciones: List[float],
) -> tuple[List[str], List[Dict[str, Any]]]:
    n = min(len(segmentos), len(imagenes), len(img_meta), len(audios), len(duraciones))
    if n <= 0:
        return imagenes, img_meta

    print("\n[CUSTOM] Selección manual de imágenes (opcional)")
    print("Se recorre segmento por segmento; puedes volver atrás.")

    def _open_file_default(path: str):
        try:
            if os.name == "nt":
                os.startfile(path)                              
                return
        except Exception:
            pass
        try:
            import webbrowser

            webbrowser.open("file://" + os.path.abspath(path))
        except Exception:
            pass

    i = 0
    while i < n:
        seg_idx_1 = i + 1
        seg_tag = f"seg_{seg_idx_1:02d}"
        meta = img_meta[i] if i < len(img_meta) else {}
        cands = meta.get("candidates") if isinstance(meta, dict) else None
        cands = cands if isinstance(cands, list) else []
        sel = meta.get("selected") if isinstance(meta, dict) else None

        note = str((segmentos[i] or {}).get("note") or "").strip() if isinstance(segmentos[i], dict) else ""
        q = str((segmentos[i] or {}).get("image_query") or "").strip() if isinstance(segmentos[i], dict) else ""

        audio_path = audios[i]
        dur = float(duraciones[i]) if i < len(duraciones) else 0.0

        print("\n" + ("=" * 70))
        print(f"Segmento {seg_idx_1}/{n}: {seg_tag}")
        if q:
            print(f"Query: {q[:160]}")
        if note:
            print(f"Nota: {note[:220]}")
        print(f"Audio: {audio_path} ({dur:.2f}s)")

        if sel:
            try:
                print(
                    "Actual: "
                    f"cand={sel.get('candidate_index')} score={sel.get('score')} "
                    f"path={sel.get('path')}"
                )
            except Exception:
                pass
        else:
            print("Actual: (sin selección registrada)")

        if not cands:
            print("Candidatos: (no hay candidatos guardados para este segmento)")
                                               
            print("\nOpciones: 6) Atrás   7) Salir y renderizar   1) Siguiente")
            opt = input("> ").strip()
            if opt == "6" and i > 0:
                i -= 1
                continue
            if opt == "7":
                print("[CUSTOM] Selección manual finalizada.")
                break
            i += 1
            continue

        print("Candidatos disponibles:")
        for c in cands:
            if not isinstance(c, dict):
                continue
            ci = c.get("candidate_index")
            sc = c.get("score")
            p = c.get("path")
            t = (c.get("title") or "")
            t = re.sub(r"\s+", " ", str(t)).strip()
            print(f"  {ci}. score={sc} path={p} title={t[:80]}")

        print("\nOpciones:")
        print("  1) Mantener imagen actual y seguir")
        print("  2) Elegir candidato (por número)")
        print("  3) Ver un candidato (por número)")
        print("  4) Ver imagen actual")
        print("  5) Abrir audio del segmento")
        print("  6) Atrás (segmento anterior)")
        print("  7) Salir y renderizar")
        print("  8) Abrir TODOS los candidatos + audio")

        opt = input("> ").strip()

        if opt == "6":
            if i > 0:
                i -= 1
            else:
                print("[CUSTOM] Ya estás en el primer segmento.")
            continue
        if opt == "7":
            print("[CUSTOM] Selección manual finalizada.")
            break
        if opt == "4":
            cur = imagenes[i] if i < len(imagenes) else ""
            if cur and os.path.exists(cur):
                _open_file_default(cur)
            else:
                print("[CUSTOM] No hay imagen actual para abrir.")
            continue
        if opt == "5":
            ap = audio_path
            if ap and os.path.exists(ap):
                _open_file_default(ap)
            else:
                print("[CUSTOM] No hay audio para abrir.")
            continue
        if opt == "8":
                                                                         
            ap = audio_path
            if ap and os.path.exists(ap):
                _open_file_default(ap)
            else:
                print("[CUSTOM] No hay audio para abrir.")

            opened = 0
            for c in cands:
                if not isinstance(c, dict):
                    continue
                relp = str(c.get("path") or "").strip()
                if not relp:
                    continue
                absp = os.path.join(carpeta, relp.replace("/", os.sep))
                if not os.path.exists(absp):
                    continue
                _open_file_default(absp)
                opened += 1
            print(f"[CUSTOM] Abiertos {opened} candidatos.")
            continue
        if opt == "3":
            kraw = input("Número de candidato a ver: ").strip()
            try:
                kview = int(kraw)
            except Exception:
                continue

            chosen_view = None
            for c in cands:
                if isinstance(c, dict) and int(c.get("candidate_index") or -1) == kview:
                    chosen_view = c
                    break
            if not chosen_view:
                print("[CUSTOM] Ese candidato no existe.")
                continue
            relp = str(chosen_view.get("path") or "").strip()
            if not relp:
                print("[CUSTOM] Candidato sin path.")
                continue
            absp = os.path.join(carpeta, relp.replace("/", os.sep))
            if not os.path.exists(absp):
                print("[CUSTOM] Archivo no existe:", absp)
                continue
            _open_file_default(absp)
            continue

        if opt == "1":
            i += 1
            continue

        if opt != "2":
            continue

        kraw = input("Número de candidato a elegir: ").strip()
        try:
            chosen_k = int(kraw)
        except Exception:
            continue

        chosen = None
        for c in cands:
            if isinstance(c, dict) and int(c.get("candidate_index") or -1) == chosen_k:
                chosen = c
                break
        if not chosen:
            print("[CUSTOM] Candidato no encontrado; se mantiene el actual.")
            continue

        rel = str(chosen.get("path") or "").strip()
        if not rel:
            print("[CUSTOM] Candidato inválido (sin path); se mantiene el actual.")
            continue

        abs_candidate = os.path.join(carpeta, rel.replace("/", os.sep))
        if not _es_imagen_valida(abs_candidate):
            print(f"[CUSTOM] Candidato no es imagen válida: {abs_candidate}")
            continue

        ext = os.path.splitext(abs_candidate)[1].lower() or ".jpg"
        stable = os.path.join(carpeta, f"{seg_tag}_chosen{ext}")
        try:
            with open(abs_candidate, "rb") as src, open(stable, "wb") as dst:
                dst.write(src.read())
        except Exception as e:
            print(f"[CUSTOM] No se pudo fijar imagen elegida: {e}")
            continue

                                           
        imagenes[i] = stable

                                         
        selected = {
            "candidate_index": int(chosen.get("candidate_index") or chosen_k),
            "score": int(chosen.get("score") or 1),
            "url": chosen.get("url"),
            "title": chosen.get("title"),
            "path": os.path.relpath(stable, carpeta).replace("\\", "/"),
        }
        if isinstance(meta, dict):
            meta["selected"] = selected
        img_meta[i] = meta
        if isinstance(segmentos[i], dict):
            segmentos[i]["image_selection"] = meta

        print(f"[CUSTOM] ✅ Elegida imagen candidato {chosen_k} para segmento {seg_idx_1}")
        i += 1

    return imagenes, img_meta


def _prompt_titulo_youtube(brief: str, script_es: str) -> str:
    script_snip = (script_es or "").strip().replace("\n", " ")
    script_snip = re.sub(r"\s+", " ", script_snip)[:1200]
    return (
        "Eres experto en títulos virales de YouTube en español. "
        "Genera UN SOLO título llamativo, claro y específico. "
        "Reglas: sin comillas, sin hashtags, sin emojis. "
        "Debe ser apto para subir tal cual.\n"
        f"TEMA/BRIEF: {brief}\n"
        f"GUIÓN (resumen): {script_snip}\n\n"
        "Devuelve SOLO el título."
    )


def generar_titulo_youtube(brief: str, script_es: str) -> str:
    prompt = _prompt_titulo_youtube(brief, script_es)
    titulo = _ollama_generate_with_timeout(
        prompt,
        temperature=0.55,
        max_tokens=180,
        timeout_sec=max(settings.ollama_timeout, 60),
        model=settings.ollama_text_model_short,
    )
    titulo = (titulo or "").strip().strip('"').strip("'")
    titulo = re.sub(r"\s+", " ", titulo).strip()
    if not titulo:
        raise RuntimeError("Ollama no devolvió un título")
    return titulo


def _prompt_hashtags_shorts(brief: str, title_es: str, script_es: str) -> str:
    script_snip = (script_es or "").strip().replace("\n", " ")
    script_snip = re.sub(r"\s+", " ", script_snip)[:900]
    title_es = re.sub(r"\s+", " ", (title_es or "").strip())[:120]
    brief = re.sub(r"\s+", " ", (brief or "").strip())[:200]
    return (
        "Eres experto en SEO para YouTube Shorts en español. "
        "Genera de 3 a 5 hashtags óptimos y relevantes para el video. "
        "Reglas estrictas: \n"
        "- Devuelve SOLO los hashtags separados por espacios (nada más).\n"
        "- Deben empezar con # y no llevar espacios.\n"
        "- Máximo 18 caracteres por hashtag.\n"
        "- Sin tildes/acentos y sin signos raros; usa letras/números/underscore.\n"
        "- Sin duplicados.\n"
        "- Incluye 1 hashtag amplio de Shorts (por ejemplo #shorts o #youtubeshorts).\n"
        "- Incluye 1-2 hashtags generales del formato (por ejemplo #curiosidades, #datoscuriosos).\n"
        "- El resto deben ser nicho del tema (nombres propios/franquicia/objeto).\n\n"
        f"TITULO_BASE: {title_es}\n"
        f"BRIEF: {brief}\n"
        f"GUION_RESUMEN: {script_snip}\n"
    )


def _normalizar_hashtag(tag: str) -> str:
    t = (tag or "").strip()
    if not t:
        return ""
    if not t.startswith("#"):
        t = "#" + t
    body = t[1:]
                                                                                    
    body = re.sub(r"[^A-Za-z0-9_]", "", body)
    body = body.strip("_")
    if not body:
        return ""
    return "#" + body.lower()


def _extraer_hashtags(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
                                        
    found = re.findall(r"#[^\s#]+", raw)
    out: list[str] = []
    seen = set()
    for f in found:
        h = _normalizar_hashtag(f)
        if not h:
            continue
        if len(h) > 19:            
            continue
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
        if len(out) >= 5:
            break
    return out


def _generar_hashtags_shorts(brief: str, title_es: str, script_es: str) -> list[str]:
    prompt = _prompt_hashtags_shorts(brief, title_es, script_es)
    raw = _ollama_generate_with_timeout(
        prompt,
        temperature=0.35,
        max_tokens=60,
        timeout_sec=max(settings.ollama_timeout, 60),
        model=settings.ollama_text_model_short,
    )
    tags = _extraer_hashtags(raw)

                                                           
    if len(tags) < 3:
        fallback = ["#shorts", "#curiosidades", "#datoscuriosos"]
        for t in fallback:
            h = _normalizar_hashtag(t)
            if h and h not in tags:
                tags.append(h)
            if len(tags) >= 3:
                break

                  
    tags = tags[:5]
    if len(tags) < 3:
                                                 
        for t in ("#shorts", "#curiosidades", "#viral", "#historia", "#misterio"):
            h = _normalizar_hashtag(t)
            if h and h not in tags:
                tags.append(h)
            if len(tags) >= 3:
                break
        tags = tags[:5]

    return tags


def _append_shorts_hashtags_to_title(title_es: str, *, brief: str, script_es: str) -> str:
    raw = re.sub(r"\s+", " ", (title_es or "").strip())
    if not raw:
        raw = "Video personalizado"

                                                                               
    existing = _extraer_hashtags(raw)
    base = re.sub(r"(?:\s*#[^\s#]+)+\s*$", "", raw).strip()
    if not base:
        base = "Video personalizado"

    if len(existing) >= 3:
        return f"{base} {' '.join(existing[:5])}".strip()

    tags = _generar_hashtags_shorts(brief, base, script_es)
    merged: list[str] = []
    seen = set()
    for t in existing + tags:
        h = _normalizar_hashtag(t)
        if not h:
            continue
        if h in seen:
            continue
        seen.add(h)
        merged.append(h)
        if len(merged) >= 5:
            break

                       
    if len(merged) < 3:
        for t in ("#shorts", "#curiosidades", "#datoscuriosos", "#viral", "#misterio"):
            h = _normalizar_hashtag(t)
            if h and h not in seen:
                merged.append(h)
                seen.add(h)
            if len(merged) >= 3:
                break

    return f"{base} {' '.join(merged[:5])}".strip()


def _copiar_imagen_manual_a_segmento(carpeta: str, seg_index_1: int, src: str) -> str:
    seg_tag = f"seg_{seg_index_1:02d}"
    dst = os.path.join(carpeta, f"{seg_tag}_chosen")

    src = (src or "").strip().strip('"').strip("'")
    if not src:
        raise ValueError("Ruta/URL vacía")

    if src.lower().startswith("http://") or src.lower().startswith("https://"):
                             
        url = src
                       
        ext = ".jpg"
        parsed = url.split("?")[0].lower()
        for suf in (".jpg", ".jpeg", ".png", ".webp"):
            if parsed.endswith(suf):
                ext = ".jpg" if suf == ".jpeg" else suf
                break
        out = _descargar_imagen_a_archivo(url, dst + ext)
        if not out:
            raise RuntimeError("No se pudo descargar la imagen")
        return out

                                
    if not os.path.exists(src):
        raise FileNotFoundError(f"No existe: {src}")
    if not _es_imagen_valida(src):
        raise RuntimeError("La imagen local no es válida o está corrupta")

    ext = os.path.splitext(src)[1].lower() or ".jpg"
    out = dst + ext
    os.makedirs(carpeta, exist_ok=True)
    with open(src, "rb") as fsrc, open(out, "wb") as fdst:
        fdst.write(fsrc.read())
    return out


def descargar_imagenes_web(carpeta: str, queries: List[str]) -> List[str]:
    rutas: List[str] = []
    for idx, q in enumerate(queries):
        query = q.strip() or "reference photo"
        url = _buscar_url_imagen(query)
        if not url:
            print(f"[IMG-WEB] Sin resultados para: {query}")
            continue
        ruta = _descargar_imagen(url, carpeta, idx)
        if ruta:
            rutas.append(ruta)
        else:
            print(f"[IMG-WEB] No se guardó imagen para: {query}")
    return rutas


def _buscar_contexto_web(query: str, *, max_snippets: int = 4) -> str:
    q = quote_plus(query.strip())
    url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return ""

    textos: List[str] = []
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        textos.append(abstract)
    related = data.get("RelatedTopics") or []
    for topic in related:
        if isinstance(topic, dict) and topic.get("Text"):
            textos.append(str(topic.get("Text")).strip())
            if len(textos) >= max_snippets:
                break
    return " ".join(textos[:max_snippets]).strip()


def _prompt_plan(
    brief: str,
    contexto: str,
    *,
    target_seconds: int,
    max_prompts: int = 12,
    hook_hint: str = "",
    tone_hint: str = "",
    require_curious_final: bool = False,
) -> str:
    contexto_line = contexto if contexto else "Sin contexto web disponible, apóyate en conocimiento general y datos comprobables."
    target_seconds = int(max(15, target_seconds))
    target_minutes = max(1, int(round(target_seconds / 60.0)))

    if target_seconds >= 300:
                                                                              
                                                                                   
        # Escala para 5-20 minutos. Para 5 min, _target_word_range da ~650-950.
        total_words_min, total_words_max = _target_word_range(target_seconds)

        if target_seconds >= 480:
            # Documental largo (8-20 min): segmentos más densos.
            per_seg_min, per_seg_max = 70, 120
            # Aproximación: ~95 palabras por segmento.
            approx_segments = int(round(((total_words_min + total_words_max) / 2.0) / 95.0))
            approx_segments = max(18, min(44, approx_segments))
            seg_min, seg_max = max(18, approx_segments - 6), min(48, approx_segments + 6)
            humor_line = "Incluye sarcasmo ligero y humor inteligente (1-2 chistes cortos por minuto, no stand-up). "
        else:
            # Largo estándar (~5-8 min)
            seg_min, seg_max = 12, 18
            per_seg_min, per_seg_max = 45, 85
            humor_line = "Incluye sarcasmo ligero y humor inteligente (1-2 chistes cortos por minuto, no stand-up). "
    else:
                                   
        seg_min, seg_max = 6, 10
        total_words_min, total_words_max = _target_word_range(target_seconds)
        per_seg_min, per_seg_max = 22, 55

        humor_line = "Incluye sarcasmo ligero y 1-2 chistes cortos en total (breves, naturales). "

    estilo_line = (
        "Tono: energético y humano (no plano), con variaciones de ritmo y emoción (sorpresa, emoción, seriedad breve cuando toque). "
        + humor_line
        + "Evita groserías y humor ofensivo. No uses emojis. Usa signos ¿¡ cuando corresponda. "
        "Mantén el contenido informativo y verificable: el humor es un condimento, no el objetivo."
    )

    hook_hint = re.sub(r"\s+", " ", (hook_hint or "").strip())
    hook_line = f"GANCHO_OBLIGATORIO (si aplica): {hook_hint}\n" if hook_hint else ""

    tone_hint = re.sub(r"\s+", " ", (tone_hint or "").strip())
    tone_line = f"TONO (si aplica): {tone_hint}\n" if tone_hint else ""

    # Prompt maestro para Shorts (60s). Mantiene salida JSON para el pipeline.
    if target_seconds <= 75:
        return (
            "Actúa como un editor experto de YouTube Shorts. Tu objetivo es la Retención Máxima. "
            "Escribe un guion sobre: [TEMA].\n\n"
            "REGLAS CRÍTICAS:\n\n"
            "- NADA de saludos ('Hola', 'Bienvenidos'). Empieza con el dato más brutal en el segundo 0.\n"
            "- 0-3s (El Gancho): una frase visual o polémica que obligue a no deslizar.\n"
            "- 3-15s (El Contexto): explica el problema rápido.\n"
            "- 15-45s (El Desarrollo): datos curiosos rápidos.\n"
            "- 45-60s (Cierre Cíclico): una frase que conecte con el inicio (Loop).\n\n"
            "Tono: [TONO]. Usa lenguaje coloquial de México, directo y rápido.\n"
            "Formato: dame el guion listo para TTS (Texto a Voz).\n\n"
            "---\n"
            "IMPORTANTE: Este proyecto usa imágenes REALES de internet (no IA). Debes alinear el guion con búsquedas de fotos reales.\n"
            "Sustituye [TEMA] usando BRIEF. Sustituye [TONO] usando TONO si se proporciona.\n\n"
            f"BRIEF (TEMA): {brief}\n"
            f"DATOS RAPIDOS: {contexto_line}\n"
            + hook_line
            + tone_line
            + "\n"
            "MAPEO A SEGMENTOS (OBLIGATORIO):\n"
            "- segments debe tener EXACTAMENTE 6 objetos, en orden: \n"
            "  1) Hook (0-3s)\n"
            "  2) Contexto (3-15s)\n"
            "  3) Desarrollo (15-28s)\n"
            "  4) Desarrollo (28-38s)\n"
            "  5) Desarrollo (38-45s)\n"
            "  6) Cierre cíclico (45-60s) que haga LOOP y conecte con el Hook.\n"
            "- PROHIBIDO: 'suscríbete', 'dale like', despedidas, muletillas de intro.\n"
            "- El primer segmento debe ser brutal y abrir un 'open loop'.\n"
            "- El último segmento debe cerrar con LOOP: referencia clara a la idea/frase del Hook sin repetirlo literal palabra por palabra.\n"
            "- Contenido verificable y específico; cero relleno.\n\n"
            "FORMATO DE RESPUESTA OBLIGATORIO:\n"
            "- Responde SOLO con un objeto JSON válido, sin texto adicional.\n"
            "- NO uses markdown.\n\n"
            "Entrega SOLO JSON con estas claves:\n"
            "- title_es: titulo atractivo (<=80 chars).\n"
            "- hook_es: UNA sola frase de gancho inmediato (no lista).\n"
            "- segments: lista de 6 objetos. Cada objeto: {\n"
            f"    text_es: parte del guion en español ({per_seg_min}-{per_seg_max} palabras) para narrar en TTS;\n"
            "    image_query: frase corta en ingles para buscar foto real exacta;\n"
            "    image_prompt: descripcion en ingles de la escena para contexto visual;\n"
            "    note: detalle breve de lo que debe verse para validar que coincide.\n"
            "  }.\n"
            "- script_es: concatenación de todos los text_es en orden, como guion completo.\n"
            f"Verificación interna: el script_es final debe tener aprox {total_words_min}-{total_words_max} palabras.\n"
            "NO agregues comentarios adicionales, SOLO el objeto JSON válido."
        )

    # Prompt maestro documental (>=5 min). Mantiene salida JSON para el pipeline.
    # Nota: NO ponemos etiquetas [VISUAL]/[SONIDO] dentro de text_es porque TTS las leería.
    # En su lugar, empujamos la guía visual a note/image_prompt.
    if target_seconds >= 300:
        tone_fill = tone_hint or "Narrativo, misterio/tecnología, profesional con carisma"

        # Segmentos objetivo escalados a la duración.
        wmin_doc, wmax_doc = _target_word_range(target_seconds)
        avg_words_doc = (wmin_doc + wmax_doc) / 2.0
        # Para 5-8 min: segmentos más cortos; para 8-20: segmentos más densos.
        words_per_seg = 70.0 if target_seconds < 480 else 95.0
        approx_segments = int(round(avg_words_doc / max(55.0, words_per_seg)))
        approx_segments = max(12, min(44, approx_segments))
        seg_min_doc, seg_max_doc = max(12, approx_segments - 5), min(48, approx_segments + 6)

        seg_word_line = "45-85" if target_seconds < 480 else "70-120"
        return (
            "Actúa como un Guionista Senior de Documentales para YouTube, experto en retención de audiencia y Storytelling.\n\n"
            "Tu Misión: escribir el guion completo para un video de análisis profundo sobre el tema del BRIEF.\n"
            "Configuración de Idioma: Español de México (neutro, profesional pero con carisma).\n\n"
            "Estructura narrativa obligatoria (a escala del video):\n"
            "- Hook (0:00-1:00): no saludes. Empieza con escena dramática/dato chocante/paradoja. Plantea la GRAN PREGUNTA que se responderá al final.\n"
            "- Contexto (1:00-3:00): cómo era el mundo antes; establece reglas; crea tensión/nostalgia.\n"
            "- Conflicto (3:00-6:00): el problema central con 'Pero entonces...'. Explica lo técnico con analogías simples.\n"
            "- Clímax/caída (6:00-8:00): el punto exacto donde todo cambió; tono dramático.\n"
            "- Conclusión/reflexión (8:00-10:00+): cierra el círculo volviendo al inicio, responde la gran pregunta y deja una lección.\n\n"
            "REGLAS CRÍTICAS DE RETENCIÓN:\n"
            "- Nada de saludos, nada de 'en este video', nada de 'suscríbete/like'.\n"
            "- Cada segmento debe aportar un dato NUEVO, concreto y verificable (no paja).\n"
            "- Estilo: narrativo, como misterio/crimen incluso si es tecnología. Une puntos con narrativa, evita listas aburridas.\n"
            f"TONO: {tone_fill}.\n\n"
            "IMPORTANTE SOBRE VISUALES:\n"
            "- text_es debe ser SOLO narración (TTS).\n"
            "- Las ideas visuales van en note (español) y en image_prompt (inglés).\n"
            "- image_query debe ser una búsqueda corta en inglés para una foto/clip REAL.\n\n"
            f"BRIEF: {brief}\n"
            f"DATOS RAPIDOS: {contexto_line}\n"
            + hook_line
            + tone_line
            + "\n"
            "FORMATO DE RESPUESTA OBLIGATORIO:\n"
            "- Responde SOLO con un objeto JSON válido (sin markdown).\n\n"
            "Entrega SOLO JSON con estas claves:\n"
            "- title_es: titulo atractivo (<=80 chars).\n"
            "- hook_es: UNA sola frase de gancho inmediato (no lista).\n"
            f"- segments: lista de {seg_min_doc}-{seg_max_doc} objetos. Cada objeto: {{\n"
            f"    text_es: narración en español ({seg_word_line} palabras);\n"
            "    image_query: frase corta en inglés para buscar foto real exacta;\n"
            "    image_prompt: descripción en inglés de la escena/visual sugerido;\n"
            "    note: guía visual en español (qué debe verse) + si aplica un [SONIDO: ...] (pero no lo metas en text_es).\n"
            "  }.\n"
            "- script_es: concatenación de todos los text_es en orden.\n"
            f"Verificación interna: script_es debe durar ~{target_seconds} segundos (~{target_minutes} min) y tener aprox {wmin_doc}-{wmax_doc} palabras.\n"
            + (
                "Cierre obligatorio: el último segmento debe incluir 'Dato curioso final:' o '¿Sabías que...?' exactamente una vez y además cerrar el círculo narrativo.\n"
                if require_curious_final
                else "Cierre obligatorio: el último segmento debe cerrar el círculo narrativo con un dato concreto y memorable.\n"
            )
            +
            "NO agregues comentarios adicionales, SOLO el objeto JSON válido."
        )

    return (
        "Eres productor de YouTube Shorts en español. Diseña un video informativo, claro y carismático para retener audiencia. "
        "Usarás imágenes REALES de internet (no IA). Necesito que alinees guion, segmentos y qué debe verse en cada momento.\n"
        f"BRIEF: {brief}\n"
        f"DATOS RAPIDOS: {contexto_line}\n\n"
        + hook_line
        + tone_line
        + "REGLAS DE RETENCION (CRITICAS):\n"
        "- PROHIBIDO en el inicio: 'hola', 'bienvenidos', 'hoy te voy a', 'en este video', 'suscríbete', 'dale like'.\n"
        "- El PRIMER segmento (segments[0].text_es) debe empezar fuerte: una afirmación impactante + curiosidad (open loop).\n"
        "- Si se proporciona GANCHO_OBLIGATORIO, hook_es DEBE ser exactamente ese texto y el primer segmento DEBE arrancar con esa frase.\n"
        "- Ritmo: frases cortas, cero relleno. Cada segmento aporta un dato NUEVO (no reexplicar).\n\n"
        f"ESTILO: {estilo_line}\n\n"
        f"OBJETIVO: El guion narrado debe durar ~{target_seconds} segundos (~{target_minutes} min). "
        "NO rellenes con silencio: escribe contenido real.\n\n"
        "ESTRUCTURA OBLIGATORIA (en este orden):\n"
        "1) Introducción: presenta el tema elegido de forma concreta (qué es y por qué importa).\n"
        "2) Datos curiosos: entrega datos específicos, ejemplos, mini-historia o contexto; evita generalidades.\n"
        + (
            "3) Cierre: termina con un DATO CURIOSO FINAL memorable (incluye 'Dato curioso final:' o '¿Sabías que...?').\n"
            if require_curious_final
            else "3) Cierre: termina con un dato final concreto y memorable, sin muletillas genéricas.\n"
        )
        +
        "PROHIBIDO: terminar con frases vagas tipo 'en este video exploramos el mundo mágico' sin dar un dato final.\n\n"
        "Evita repeticiones: cero saludos; cero muletillas tipo 'hoy exploraremos'/'en este video'. Cada segmento debe aportar un dato nuevo.\n"
        "Evita redundancia: no repitas el mismo ejemplo/dato en más de un segmento. No rellenes con frases vacías. No repitas la misma pregunta retórica en varios segmentos.\n"
        "Hook único: hook_es debe ser UNA sola frase corta, no un array.\n"
        + (
            "Cierre obligatorio: el último segmento debe incluir 'Dato curioso final:' o '¿Sabías que...?' una sola vez.\n\n"
            if require_curious_final
            else "Cierre obligatorio: el último segmento debe cerrar con una conclusión concreta y sin fórmulas repetitivas.\n\n"
        )
        +
        "FORMATO DE RESPUESTA OBLIGATORIO:\n"
        "- Responde SOLO con un objeto JSON válido, sin texto adicional antes o después.\n"
        "- NO uses bloques de código markdown (```json o ```).\n"
        "- NO agregues comas al final de la última propiedad en objetos o arrays.\n"
        "- Escapa todas las comillas dobles dentro de las cadenas (usa \\\" ).\n"
        "- Asegura que el JSON sea válido y parseable desde la primera línea hasta la última.\n\n"
        "Entrega SOLO JSON con estas claves:\n"
        "- title_es: titulo atractivo (<=80 chars).\n"
        "- hook_es: UNA sola frase de gancho inmediato (no lista).\n"
        f"- segments: lista de {seg_min}-{seg_max} objetos, estricto orden cronologico. Cada objeto: {{\n"
        f"    text_es: parte del guion en español ({per_seg_min}-{per_seg_max} palabras) para narrar en TTS;\n"
        "    image_query: frase corta en ingles para buscar foto real exacta (ej: 'elder wand prop from deathly hallows movie');\n"
        "    image_prompt: descripcion en ingles de la escena para contexto visual;\n"
        "    note: detalle breve de lo que debe verse para validar que coincide.\n"
        "  }.\n"
        "- script_es: concatenación de todos los text_es en orden, como guion completo.\n"
        "Reglas: prioriza objetos/lugares/personas reales, evita conceptos abstractos. Para nombres concretos (ej. 'varita de saúco') usa queries precisas del item real. "
        f"Verificación interna: el script_es final debe tener aprox {total_words_min}-{total_words_max} palabras. "
        "NO agregues comentarios adicionales, SOLO el objeto JSON válido."
    )


def _prompt_expand_to_min_duration(brief: str, contexto: str, plan_raw: Dict[str, Any], *, min_seconds: int) -> str:
    contexto_line = contexto if contexto else "Sin contexto web disponible."
    raw_str = json.dumps(plan_raw, ensure_ascii=False)[:4000]
    wmin, wmax = _target_word_range(min_seconds)
    return (
        "Reescribe y EXPANDE el plan para cumplir duración mínima sin añadir relleno vacío. "
        "Mantén el mismo tema y estructura, pero agrega más segmentos y más detalle narrativo.\n"
        "Tono: energético y humano (no plano), con sarcasmo ligero y chistes breves ocasionales (sin groserías ni humor ofensivo). "
        "Usa variaciones de ritmo y emoción; no uses emojis.\n"
        f"BRIEF: {brief}\n"
        f"DATOS RAPIDOS: {contexto_line}\n"
        f"DURACION MINIMA: {min_seconds} segundos (>= {max(5, min_seconds//60)} minutos).\n"
        f"PLAN ACTUAL (puede estar incompleto): {raw_str}\n\n"
        + (
            "ESTRUCTURA OBLIGATORIA: intro concreta → datos curiosos específicos → cierre con DATO CURIOSO FINAL (incluye 'Dato curioso final:' o '¿Sabías que...?').\n"
            if _requires_curious_final(brief)
            else "ESTRUCTURA OBLIGATORIA: intro concreta → datos específicos → cierre con dato final concreto y memorable.\n"
        )
        +
        "Entrega SOLO JSON con las mismas claves: title_es, hook_es, segments (12-24), script_es. "
        "Cada segment.text_es 80-140 palabras. "
        f"script_es total ~{wmin}-{wmax} palabras. "
        "JSON valido, nada mas."
    )


def generar_plan_personalizado(brief: str, *, min_seconds: int | None = None, max_prompts: int = 12) -> VideoPlan:
    brief_in = (brief or "").strip()
    if not brief_in:
        raise ValueError("Brief vacio")

    brief_source = _extract_source_text_from_transform_brief(brief_in)

                                                               
    target_seconds = int(min_seconds or settings.custom_min_video_sec)
    if target_seconds <= 10:
        target_seconds = max(3, target_seconds)
    elif target_seconds not in (60, 300):
        target_seconds = max(60, target_seconds)

    if target_seconds <= 10:
        return _build_debug_quick_plan(brief_in, target_seconds=target_seconds)

    if _looks_like_full_story_text(brief_in) and not _is_transformation_request(brief_in):
        return _build_preserved_text_plan(brief_in, target_seconds=target_seconds)

    brief = _sanitize_brief_for_duration(brief_source) or brief_source
    hook_hint = _extract_hook_from_brief(brief_source)
    tone_hint = _extract_tone_from_brief(brief_source)
    require_curious_final = _requires_curious_final(brief_source, tone_hint)

    # Modo documental (chunking por capítulos) para 5-20 minutos.
    # Se puede desactivar con env CUSTOM_LONG_CHUNKING=0.
    doc_long = int(target_seconds) >= 300
    chunking_enabled = (os.environ.get("CUSTOM_LONG_CHUNKING") or "1").strip().lower() not in {"0", "false", "no"}

    # En modo corto priorizamos velocidad y evitamos búsqueda web para bajar latencia.
    if int(target_seconds) < 300:
        contexto = ""
    else:
        contexto = _buscar_contexto_web(brief)

    fast_short_mode = (
        int(target_seconds) < 300
        and (os.environ.get("CUSTOM_FAST_MODE") or "").strip().lower() in {"1", "true", "yes", "si", "sí", "on"}
    )

                                                                          
    plan: Dict[str, Any] | None = None
    segmentos: List[Dict[str, Any]] = []
    titulo = ""
    hook = ""
    script = ""

    model_text = _text_model_for_seconds(target_seconds)
    print(f"[CUSTOM] 🧠 Modelo texto por duración: {model_text} (target_seconds={target_seconds})")

                                                           
                                                                                      
                                                                       
                                                                            
    if target_seconds >= 300:
                                                                                  
        is_qwen = "qwen" in model_text.lower()
        # Escala suave hasta ~20 min. Conservador para no romper modelos con ctx/tokens bajos.
        if target_seconds >= 900:
            tokens_limit = 4600 if is_qwen else 4200
            timeout = 360
            min_ctx = 8192 if is_qwen else 7168
        elif target_seconds >= 480:
            tokens_limit = 4100 if is_qwen else 3800
            timeout = 300
            min_ctx = 7168 if is_qwen else 6144
        else:
            tokens_limit = 3600 if is_qwen else 3400
            timeout = 240
            min_ctx = 6144 if is_qwen else 5632

        print(f"[CUSTOM] 📝 Generando plan largo (~{target_seconds}s) (ctx={min_ctx}, tokens={tokens_limit})")
    else:
        tokens_limit = 1300
        timeout = 140
        min_ctx = None

    if fast_short_mode:
        tokens_limit = min(tokens_limit, 900)
        timeout = min(timeout, 120)

    timeout = max(settings.ollama_timeout, timeout)
    # Evita bloqueos muy largos en guiones cortos si OLLAMA_TIMEOUT está alto (ej. 900).
    if int(target_seconds) < 300:
        timeout = min(timeout, 240)
    else:
        timeout = min(timeout, 900)

    def _prompt_doc_outline(
        brief_txt: str,
        contexto_txt: str,
        *,
        tone: str,
        hook_force: str = "",
        chapters_n: int,
        target_sec: int,
    ) -> str:
        contexto_line2 = contexto_txt if contexto_txt else "Sin contexto web disponible, apóyate en conocimiento general y datos comprobables."
        tone2 = tone or "Documental narrativo, profesional con carisma"
        hook_force = re.sub(r"\s+", " ", (hook_force or "").strip())
        hook_line2 = f"GANCHO_OBLIGATORIO (si aplica): {hook_force}\n" if hook_force else ""
        return (
            "Actúa como un guionista senior de documentales para YouTube, experto en retención.\n"
            "Devuelve SOLO JSON válido. Sin markdown.\n\n"
            f"BRIEF: {brief_txt}\n"
            f"DATOS RAPIDOS: {contexto_line2}\n"
            f"TONO: {tone2}\n"
            + hook_line2
            + "\n"
            f"Tarea: crea un ESQUELETO (outline) de {chapters_n} capítulos para un video largo (~{max(8, int(target_sec//60))}-{max(10, int(target_sec//60)+3)} min) sobre el BRIEF.\n"
            "Reglas: narrativa con 'Pero entonces...', analogías simples para lo técnico, y una GRAN PREGUNTA que se responde al final.\n\n"
            "JSON requerido:\n"
            "{\n"
            "  title_es: string (<=80 chars),\n"
            "  hook_es: string (1 frase corta),\n"
            "  gran_pregunta_es: string (1 frase),\n"
            "  chapters: [\n"
            f"    {{ idx: 1..{chapters_n}, title_es: string, goal_es: string, key_points_es: [3-5 strings] }}\n"
            "  ]\n"
            "}\n\n"
            "Si se proporciona GANCHO_OBLIGATORIO, hook_es DEBE ser exactamente ese texto."
        )

    def _prompt_doc_chapter_segments(
        brief_txt: str,
        contexto_txt: str,
        *,
        title_es: str,
        gran_pregunta_es: str,
        chapters: list[dict[str, Any]],
        chapter_idx: int,
        chapter_title: str,
        chapter_goal: str,
        prev_bridge_es: str,
        n_segments: int,
        wmin_seg: int,
        wmax_seg: int,
        hook_force: str = "",
        tone: str = "",
        is_last_chapter: bool = False,
    ) -> str:
        contexto_line2 = contexto_txt if contexto_txt else "Sin contexto web disponible."
        tone2 = tone or "Documental narrativo, profesional con carisma"
        hook_force = re.sub(r"\s+", " ", (hook_force or "").strip())
        hook_line2 = f"GANCHO_OBLIGATORIO: {hook_force}\n" if hook_force else ""
        chapters_str = json.dumps(chapters or [], ensure_ascii=False)[:3500]

        last_rules = (
            "- Este es el ÚLTIMO capítulo: responde explícitamente la GRAN PREGUNTA y cierra el círculo narrativo volviendo al inicio.\n"
            + (
                "- El ÚLTIMO segmento debe incluir exactamente una vez 'Dato curioso final:' o '¿Sabías que...?' con el dato completo en la misma oración.\n"
                if require_curious_final
                else "- El ÚLTIMO segmento debe cerrar con una conclusión concreta y memorable (sin frases plantilla).\n"
            )
        ) if is_last_chapter else (
            "- NO cierres el video aquí: no hagas conclusión final ni despedidas. Deja un puente al siguiente capítulo.\n"
        )

        return (
            "Actúa como un Guionista Senior de Documentales para YouTube (español de México), experto en retención.\n"
            "Genera SOLO una lista JSON (array) de segmentos para este capítulo. Sin markdown.\n\n"
            f"VIDEO_TITLE_ES: {title_es}\n"
            f"BRIEF: {brief_txt}\n"
            f"DATOS RAPIDOS: {contexto_line2}\n"
            f"TONO: {tone2}\n"
            + hook_line2
            + f"GRAN_PREGUNTA: {gran_pregunta_es}\n"
            f"CHAPTER_{chapter_idx}_TITLE: {chapter_title}\n"
            f"CHAPTER_{chapter_idx}_GOAL: {chapter_goal}\n"
            f"CHAPTERS_OUTLINE_JSON: {chapters_str}\n"
            + (f"PUENTE_DESDE_ANTERIOR: {prev_bridge_es}\n" if prev_bridge_es else "")
            + "\n"
            "REGLAS CRÍTICAS:\n"
            "- Nada de saludos, nada de 'en este video', nada de 'suscríbete/like'.\n"
            "- Cada segmento aporta un dato NUEVO y verificable. Evita paja.\n"
            "- Usa narrativa tipo misterio: escena → dato → consecuencia (pero sin inventar).\n"
            "- Lo técnico con analogías simples. Usa 'Pero entonces...' cuando encaje.\n"
            + last_rules
            + "\n"
            f"Devuelve EXACTAMENTE {n_segments} segmentos. Cada segmento objeto con claves: text_es, image_query, image_prompt, note.\n"
            f"- text_es: SOLO narración TTS en español ({wmin_seg}-{wmax_seg} palabras).\n"
            "- image_query: búsqueda corta en inglés para una foto/clip real exacto.\n"
            "- image_prompt: descripción en inglés del visual sugerido.\n"
            "- note: guía visual en español (qué debe verse). Si quieres indicar audio, ponlo aquí como '[SONIDO: ...]'.\n"
            "Devuelve JSON válido, nada más."
        )

    def _distribute_ints(total: int, weights: list[float]) -> list[int]:
        if total <= 0:
            return [0 for _ in weights]
        if not weights:
            return []
        ws = [max(0.0, float(w)) for w in weights]
        s = sum(ws) or 1.0
        raw = [w / s * total for w in ws]
        base = [int(math.floor(x)) for x in raw]
        rem = total - sum(base)
        # reparte el rem a los mayores decimales
        frac = sorted([(i, raw[i] - base[i]) for i in range(len(base))], key=lambda t: t[1], reverse=True)
        for k in range(rem):
            base[frac[k % len(base)][0]] += 1
        return base


    def _doc_long_generate_with_chunking() -> tuple[Dict[str, Any], List[Dict[str, Any]], str, str, str]:
        # 1) Outline por capítulos (escala 5-20 min)
        if target_seconds < 480:
            chapters_n = 4
        elif target_seconds < 900:
            chapters_n = 5
        elif target_seconds < 1140:
            chapters_n = 6
        else:
            chapters_n = 7
        outline_prompt = _prompt_doc_outline(
            brief,
            contexto,
            tone=tone_hint,
            hook_force=hook_hint,
            chapters_n=chapters_n,
            target_sec=target_seconds,
        )
        outline = _ollama_generate_json_with_timeout(
            outline_prompt,
            temperature=0.6,
            max_tokens=900,
            timeout_sec=max(settings.ollama_timeout, 180),
            model=model_text,
        )

        if not isinstance(outline, dict):
            raise RuntimeError("Outline inválido")

        title_es = str(outline.get("title_es") or "").strip() or "Video personalizado"
        hook_es = outline.get("hook_es")
        if isinstance(hook_es, list):
            hook_es = hook_es[0] if hook_es else ""
        hook_es = str(hook_es or "").strip()
        if hook_hint:
            hook_es = hook_hint
        gran_pregunta_es = str(outline.get("gran_pregunta_es") or "").strip()
        chapters = outline.get("chapters")
        if not isinstance(chapters, list) or len(chapters) < 3:
            raise RuntimeError("Outline sin chapters")

        # Normaliza chapters a dicts simples
        ch_norm: list[dict[str, Any]] = []
        for c in chapters:
            if not isinstance(c, dict):
                continue
            ch_norm.append({
                "idx": int(c.get("idx") or (len(ch_norm) + 1)),
                "title_es": str(c.get("title_es") or "").strip(),
                "goal_es": str(c.get("goal_es") or "").strip(),
                "key_points_es": c.get("key_points_es") if isinstance(c.get("key_points_es"), list) else [],
            })
        if len(ch_norm) < 3:
            raise RuntimeError("Chapters inválidos")

        # 2) Generación secuencial por capítulos
        wmin_total, wmax_total = _target_word_range(target_seconds)
        avg_target_words = int(round((wmin_total + wmax_total) / 2.0))

        # Por segmento en doc: 5-8 min es más corto; 8-20 más denso
        if target_seconds < 480:
            wmin_seg, wmax_seg = 45, 85
        else:
            wmin_seg, wmax_seg = 70, 120

        # Segmentos objetivo: 5-8 min usa chunks más pequeños; 8-20 más densos
        words_per_seg = 70.0 if target_seconds < 480 else 95.0
        total_segments_target = int(round(avg_target_words / max(55.0, words_per_seg)))
        total_segments_target = max(12, min(44, total_segments_target))

        # Pesos por capítulo: hook/setup/body/climax/conclusion, extendidos si hay más capítulos
        if len(ch_norm) <= 5:
            weights = [0.16, 0.18, 0.24, 0.22, 0.20]
            weights = weights[: len(ch_norm)]
        else:
            # Reparte más capítulos al "body".
            weights = [0.14, 0.16] + [0.10] * max(0, len(ch_norm) - 3) + [0.20]
            # Ajuste: sube un poco el capítulo final
            weights[-1] = 0.22

        seg_plan = _distribute_ints(total_segments_target, weights)
        # Mínimo 2 segmentos por capítulo
        seg_plan = [max(2, int(n)) for n in seg_plan]

        segs: List[Dict[str, Any]] = []
        prev_bridge = ""
        for idx, ch in enumerate(ch_norm, start=1):
            nseg = seg_plan[idx - 1] if idx - 1 < len(seg_plan) else 3
            is_last = idx == len(ch_norm)
            p = _prompt_doc_chapter_segments(
                brief,
                contexto,
                title_es=title_es,
                gran_pregunta_es=gran_pregunta_es,
                chapters=ch_norm,
                chapter_idx=idx,
                chapter_title=str(ch.get("title_es") or f"Capítulo {idx}").strip(),
                chapter_goal=str(ch.get("goal_es") or "").strip(),
                prev_bridge_es=prev_bridge,
                n_segments=nseg,
                wmin_seg=wmin_seg,
                wmax_seg=wmax_seg,
                hook_force=hook_es if idx == 1 else "",
                tone=tone_hint,
                is_last_chapter=is_last,
            )
            raw = _ollama_generate_with_timeout(
                p,
                temperature=0.6,
                max_tokens=1400,
                timeout_sec=max(settings.ollama_timeout, 220),
                model=model_text,
            )
            try:
                arr = _extract_json_array(raw)
            except Exception:
                fix = (
                    "You returned INVALID JSON array. Fix it and return ONLY a valid JSON array.\n"
                    "INVALID_JSON_START\n" + (raw or "")[:8000] + "\nINVALID_JSON_END\n"
                )
                raw2 = _ollama_generate_with_timeout(
                    fix,
                    temperature=0.2,
                    max_tokens=1400,
                    timeout_sec=max(settings.ollama_timeout, 220),
                    model=model_text,
                )
                arr = _extract_json_array(raw2)

            nuevos: List[Dict[str, Any]] = []
            for s in arr:
                if not isinstance(s, dict):
                    continue
                texto = str(s.get("text_es") or "").strip()
                if not texto:
                    continue
                query = str(s.get("image_query") or "").strip()
                iprompt = str(s.get("image_prompt") or "").strip() or query
                note = str(s.get("note") or "").strip()
                nuevos.append({
                    "text_es": texto,
                    "image_query": query,
                    "image_prompt": iprompt,
                    "note": note,
                })

            if not nuevos:
                continue
            segs.extend(nuevos)
            # Puente simple: última oración del último segmento
            try:
                last_text = str(nuevos[-1].get("text_es") or "").strip()
                prev_bridge = (last_text[-220:] if len(last_text) > 220 else last_text)
            except Exception:
                prev_bridge = ""

        if not segs:
            raise RuntimeError("No se generaron segmentos por capítulos")

        script_es = " ".join([s.get("text_es", "") for s in segs]).strip()
        if _words(script_es) < int(wmin_total * 0.60):
            # Si salió muy corto, dejamos que el pipeline existente lo expanda luego.
            pass

        plan_obj: Dict[str, Any] = {
            "title_es": title_es,
            "hook_es": hook_es,
            "script_es": script_es,
            "segments": segs,
            "gran_pregunta_es": gran_pregunta_es,
            "chapters": ch_norm,
        }
        return plan_obj, segs, title_es, hook_es, script_es

    max_attempts = 2 if int(target_seconds) < 300 else 3
    if fast_short_mode:
        max_attempts = 1
    for _ in range(max_attempts):
        if doc_long and chunking_enabled:
            plan, segmentos, titulo, hook, script = _doc_long_generate_with_chunking()
        else:
            prompt = _prompt_plan(
                brief,
                contexto,
                target_seconds=target_seconds,
                max_prompts=max_prompts,
                hook_hint=hook_hint,
                tone_hint=tone_hint,
                require_curious_final=require_curious_final,
            )
            plan = _ollama_generate_json_with_timeout(
                prompt,
                temperature=0.58,                                  
                max_tokens=tokens_limit,
                timeout_sec=timeout,
                min_ctx=min_ctx,
                model=model_text,
            )

        titulo = str(plan.get("title_es") or "").strip()
        hook_val = plan.get("hook_es")
        if isinstance(hook_val, list):
            hook = str(hook_val[0] if hook_val else "").strip()
        else:
            hook = str(hook_val or "").strip()

                                                                                        
        if hook_hint:
            hook = hook_hint
        plan["hook_es"] = hook
        script = str(plan.get("script_es") or "").strip()

        raw_segments = plan.get("segments") or []
        segmentos = []
        seen = set()
        if isinstance(raw_segments, list):
            for s in raw_segments:
                if not isinstance(s, dict):
                    continue
                texto = str(s.get("text_es") or "").strip()
                if not texto:
                    continue
                if texto in seen:
                    continue
                seen.add(texto)
                query = str(s.get("image_query") or "").strip()
                iprompt = str(s.get("image_prompt") or "").strip() or query
                note = str(s.get("note") or "").strip()
                segmentos.append({
                    "text_es": texto,
                    "image_query": query,
                    "image_prompt": iprompt,
                    "note": note,
                })

                                                                                                    
        if segmentos:
            script = " ".join(seg["text_es"] for seg in segmentos)
            plan["script_es"] = script

        if not segmentos:
            continue

                                                                        
        seg_count = len(segmentos)
        minw, avgw, _maxw = _segments_word_stats(segmentos)
        total_words = _words(script) if script else sum(_words(s.get("text_es", "")) for s in segmentos)

        if target_seconds >= 300:
            if seg_count < 12:
                continue
        else:
            if seg_count < (5 if fast_short_mode else 6):
                continue
        if target_seconds >= 300 and minw < 25:
            continue
        if target_seconds < 300 and minw < (28 if fast_short_mode else 40):
            continue

        quality_issues = _plan_quality_issues(segmentos, target_seconds=target_seconds)
        if quality_issues:
            print(f"[CUSTOM] ⚠️ Plan rechazado por calidad: {', '.join(quality_issues)}")
            if not fast_short_mode:
                continue
        wmin, _wmax = _target_word_range(target_seconds)
        min_ratio = 0.55 if fast_short_mode else 0.70
        if total_words < int(wmin * min_ratio):
            continue

        break

    if not plan or not segmentos:
        raise RuntimeError("El plan no devolvió 'segments' válidos")

                                                                                   
    try:
        if segmentos and _tiene_saludo_o_outro(str(segmentos[0].get("text_es") or "")):
            opener_prompt = _prompt_rewrite_opening_segment(
                brief,
                contexto,
                hook,
                segmentos[0],
                target_seconds=target_seconds,
            )
            opener_obj = _ollama_generate_json_with_timeout(
                opener_prompt,
                temperature=0.25,
                max_tokens=520,
                timeout_sec=max(settings.ollama_timeout, 120),
            )
            if isinstance(opener_obj, dict):
                texto = str(opener_obj.get("text_es") or "").strip()
                if texto and not _tiene_saludo_o_outro(texto):
                    segmentos[0]["text_es"] = texto
                    q = str(opener_obj.get("image_query") or "").strip()
                    ip = str(opener_obj.get("image_prompt") or "").strip()
                    nt = str(opener_obj.get("note") or "").strip()
                    if q:
                        segmentos[0]["image_query"] = q
                    if ip:
                        segmentos[0]["image_prompt"] = ip
                    if nt:
                        segmentos[0]["note"] = nt
                    script = " ".join([s.get("text_es", "") for s in segmentos]).strip()
                    plan["script_es"] = script
    except Exception:
        pass

    if not script:
        script = " ".join([s.get("text_es", "") for s in segmentos]).strip()

                                                                                  
    est = _estimar_segundos(script)
    if est < float(target_seconds):
        expand_timeout = min(max(settings.ollama_timeout, 140), 220 if target_seconds < 300 else 320)
        for _ in range(6):
                                                     
            cur_words = _words(script)
            wmin, _wmax = _target_word_range(target_seconds)
            target_words = max(wmin + 40, int(wmin * 1.10))
            need_words = max(200, target_words - cur_words)
            n_segments = 6 if target_seconds >= 300 else 3
            last_note = str((segmentos[-1] or {}).get("note") or "").strip() if segmentos else ""
            add_prompt = _prompt_add_segments(brief, contexto, need_words=need_words, n_segments=n_segments, last_note=last_note)
            raw = _ollama_generate_with_timeout(
                add_prompt,
                temperature=0.55,
                max_tokens=1800,
                timeout_sec=expand_timeout,
                model=model_text,
            )
            try:
                arr = _extract_json_array(raw)
            except Exception:
                                                                      
                fix = (
                    "You returned INVALID JSON array. Fix it and return ONLY a valid JSON array.\n"
                    "INVALID_JSON_START\n" + (raw or "")[:8000] + "\nINVALID_JSON_END\n"
                )
                raw2 = _ollama_generate_with_timeout(
                    fix,
                    temperature=0.2,
                    max_tokens=1800,
                    timeout_sec=expand_timeout,
                    model=model_text,
                )
                arr = _extract_json_array(raw2)

            nuevos: List[Dict[str, Any]] = []
            for s in arr:
                if not isinstance(s, dict):
                    continue
                texto = str(s.get("text_es") or "").strip()
                if not texto:
                    continue
                query = str(s.get("image_query") or "").strip()
                iprompt = str(s.get("image_prompt") or "").strip() or query
                note = str(s.get("note") or "").strip()
                nuevos.append({
                    "text_es": texto,
                    "image_query": query,
                    "image_prompt": iprompt,
                    "note": note,
                })

            if not nuevos:
                continue

            segmentos.extend(nuevos)
            script = (script + " " + " ".join([s.get("text_es", "") for s in nuevos])).strip()
            est = _estimar_segundos(script)
            if est >= float(target_seconds):
                break

        if est < float(target_seconds):
            # Fallback final: reescribir TODO el plan para cumplir mínimo de duración.
            try:
                expand_prompt = _prompt_expand_to_min_duration(
                    brief,
                    contexto,
                    plan if isinstance(plan, dict) else {},
                    min_seconds=int(target_seconds),
                )
                expand_model = settings.ollama_text_model_short if int(target_seconds) < 300 else model_text
                expanded = _ollama_generate_json_with_timeout(
                    expand_prompt,
                    temperature=0.45,
                    max_tokens=2400,
                    timeout_sec=expand_timeout,
                    model=expand_model,
                )
                if isinstance(expanded, dict):
                    raw_segments2 = expanded.get("segments") or []
                    seg2: List[Dict[str, Any]] = []
                    seen2 = set()
                    if isinstance(raw_segments2, list):
                        for s in raw_segments2:
                            if not isinstance(s, dict):
                                continue
                            texto2 = str(s.get("text_es") or "").strip()
                            if not texto2 or texto2 in seen2:
                                continue
                            seen2.add(texto2)
                            query2 = str(s.get("image_query") or "").strip()
                            iprompt2 = str(s.get("image_prompt") or "").strip() or query2
                            note2 = str(s.get("note") or "").strip()
                            seg2.append({
                                "text_es": texto2,
                                "image_query": query2,
                                "image_prompt": iprompt2,
                                "note": note2,
                            })

                    if seg2:
                        segmentos = seg2
                        script = " ".join([s.get("text_es", "") for s in seg2]).strip()
                        plan["segments"] = seg2
                        plan["script_es"] = script
                        if expanded.get("title_es"):
                            plan["title_es"] = str(expanded.get("title_es") or "").strip()
                        if expanded.get("hook_es"):
                            plan["hook_es"] = str(expanded.get("hook_es") or "").strip()
                        est = _estimar_segundos(script)
            except Exception:
                pass

        if est < float(target_seconds):
            raise RuntimeError(
                f"El guion estimado ({int(est)}s) no cumple el mínimo ({target_seconds}s)."
            )

                                                                               
    try:
        if require_curious_final and segmentos and not _es_cierre_valido(str(segmentos[-1].get("text_es") or "")):
            for attempt in range(2):
                cierre_prompt = _prompt_rewrite_closing_segment(brief, contexto, segmentos[-1], target_seconds=target_seconds)
                cierre_obj = _ollama_generate_json_with_timeout(
                    cierre_prompt,
                    temperature=0.25 if attempt == 1 else 0.35,
                    max_tokens=520,
                    timeout_sec=max(settings.ollama_timeout, 120),
                )
                if not isinstance(cierre_obj, dict):
                    continue
                texto = str(cierre_obj.get("text_es") or "").strip()
                if not texto:
                    continue
                if not _es_cierre_valido(texto):
                    continue

                segmentos[-1]["text_es"] = texto
                q = str(cierre_obj.get("image_query") or "").strip()
                ip = str(cierre_obj.get("image_prompt") or "").strip()
                nt = str(cierre_obj.get("note") or "").strip()
                if q:
                    segmentos[-1]["image_query"] = q
                if ip:
                    segmentos[-1]["image_prompt"] = ip
                if nt:
                    segmentos[-1]["note"] = nt
                script = " ".join([s.get("text_es", "") for s in segmentos]).strip()
                break
    except Exception:
                                                                          
        pass

    prompts_final = [s.get("image_prompt") or s.get("image_query") or "photo" for s in segmentos]

    timeline_data = _generar_timeline(prompts_final, _estimar_segundos(script))

    return VideoPlan(
        brief=brief,
        target_seconds=int(target_seconds),
        title_es=titulo or "Video personalizado",
        youtube_title_es=titulo or "Video personalizado",
        hook_es=hook,
        script_es=script,
        segments=[ScriptSegment(**s) for s in segmentos],
        prompts=prompts_final,
        timeline=[TimelineItem(**t) for t in timeline_data] if timeline_data else [],
        contexto_web=contexto,
        raw_plan=plan,
    )


def generar_video_personalizado(
    brief: str,
    *,
    voz: str,
    velocidad: str,
    min_seconds: int | None = None,
    seleccionar_imagenes: bool = False,
) -> bool:
    carpeta = crear_carpeta_proyecto(prefix="custom")
    try:
        plan_model = generar_plan_personalizado(brief, min_seconds=min_seconds)
        plan = plan_model.model_dump()
    except Exception as e:
        print(f"[CUSTOM] ❌ No se pudo crear el plan: {e}")
        return False

    plan_path = os.path.join(carpeta, "custom_plan.json")
    try:
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"[CUSTOM] Plan guardado en {plan_path}")
    except Exception as e:
        print(f"[CUSTOM] ⚠️ No se pudo guardar el plan: {e}")

    segmentos = plan.get("segments") or []
    if not segmentos:
        print("[CUSTOM] Plan sin segmentos")
        return False

    textos = [s.get("text_es", "") for s in segmentos]
                                                                                     
    notes = [str(s.get("note") or "").strip() for s in segmentos]
    queries = []
    for i, s in enumerate(segmentos):
        base_q = (str(s.get("image_query") or "").strip() or str(s.get("image_prompt") or "").strip() or brief)
        q2 = _build_better_image_query(
            base_query=base_q,
            text_es=str(s.get("text_es") or ""),
            note=notes[i] if i < len(notes) else "",
            brief=str(brief or "").strip(),
        )
        queries.append(q2)

                                                                                   
    imagenes, img_meta = descargar_mejores_imagenes_ddg(carpeta, queries, notes, max_per_query=8)
    if len(imagenes) != len(textos):
        print(f"[CUSTOM] ❌ Imágenes insuficientes: {len(imagenes)}/{len(textos)}. Se aborta.")
        return False

                                                          
    for i, seg in enumerate(segmentos):
        meta = img_meta[i] if i < len(img_meta) else None
        if not isinstance(seg, dict) or not meta:
            continue
        seg["image_selection"] = meta

                                                
    try:
        yt_title = generar_titulo_youtube(brief, str(plan.get("script_es") or ""))
                                                  
        if int(plan.get("target_seconds") or (min_seconds or settings.custom_min_video_sec)) == 60:
            yt_title = _append_shorts_hashtags_to_title(
                yt_title,
                brief=str(plan.get("brief") or brief or "").strip(),
                script_es=str(plan.get("script_es") or "").strip(),
            )
        plan["youtube_title_es"] = yt_title

                                                   
        try:
            yt_title_txt = os.path.join(carpeta, "youtube_title.txt")
            with open(yt_title_txt, "w", encoding="utf-8") as f:
                f.write(str(yt_title).strip() + "\n")
        except Exception:
            pass
    except Exception as e:
        print(f"[CUSTOM] ❌ No se pudo generar título YouTube: {e}")
        return False

    audios = tts.generar_audios(textos, carpeta, voz=voz, velocidad=velocidad)
    if not audios:
        print("[CUSTOM] No se generaron audios")
        return False

    if len(audios) != len(textos):
        print(
            f"[CUSTOM] ❌ Falló TTS: se generaron {len(audios)}/{len(textos)} audios. "
            "Se cancela el render para evitar video incompleto o con silencio."
        )
        return False

    if len(imagenes) != len(textos):
        print(
            f"[CUSTOM] ❌ Desbalance plan/imagenes: imagenes={len(imagenes)} textos={len(textos)}. "
            "Se cancela el render."
        )
        return False

    min_items = len(textos)

    duraciones = [max(0.6, audio_duration_seconds(a)) for a in audios]

                                                                         
    if seleccionar_imagenes:
        try:
            imagenes, img_meta = _seleccionar_candidatos_interactivo(
                carpeta,
                segmentos,
                imagenes,
                img_meta,
                audios,
                duraciones,
            )
        except Exception as e:
            print(f"[CUSTOM] ⚠️ No se pudo hacer selección manual: {e}")
    timeline = []
    pos = 0.0
    for q, d in zip(queries[:min_items], duraciones):
        start = pos
        end = pos + d
        timeline.append({"prompt": q, "start": round(start, 2), "end": round(end, 2)})
        pos = end
    plan["timeline"] = timeline

                                                       
    try:
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    debug_min_seconds = int(plan.get("target_seconds") or 0)
    if not (0 < debug_min_seconds <= 10):
        debug_min_seconds = None

    plan_min_seconds = plan.get("min_seconds") if isinstance(plan, dict) else None
    plan_max_seconds = plan.get("max_seconds") if isinstance(plan, dict) else None
    try:
        plan_min_seconds = int(plan_min_seconds) if plan_min_seconds is not None else None
    except Exception:
        plan_min_seconds = None
    try:
        plan_max_seconds = int(plan_max_seconds) if plan_max_seconds is not None else None
    except Exception:
        plan_max_seconds = None

    try:
        default_short_min = int(settings.custom_min_video_sec or 40)
    except Exception:
        default_short_min = 40

    # Compatibilidad: planes antiguos guardaban 60s por defecto en modo corto.
    if plan_min_seconds is None:
        plan_min_seconds = default_short_min
    elif int(plan_min_seconds) == 60 and default_short_min == 40:
        plan_min_seconds = 40

    # En modo corto, fija rango 40-180 por defecto, conservando modo largo/debug.
    if plan_min_seconds < 300:
        if plan_min_seconds > 10 and plan_min_seconds < 40:
            plan_min_seconds = 40
        if plan_max_seconds is None:
            plan_max_seconds = 180
        else:
            plan_max_seconds = min(int(plan_max_seconds), 180)

    preserve_full_text = False
    try:
        raw_plan = plan.get("raw_plan") if isinstance(plan, dict) else None
        preserve_full_text = bool((raw_plan or {}).get("preserved_source_text")) if isinstance(raw_plan, dict) else False
    except Exception:
        preserve_full_text = False

    audio_min = debug_min_seconds if debug_min_seconds is not None else plan_min_seconds
    audio_max = None if preserve_full_text else plan_max_seconds

    audio_final = combine_audios_with_silence(
        audios,
        carpeta,
        gap_seconds=0,
        min_seconds=audio_min,
        max_seconds=audio_max,
    )

    video_final = render_video_ffmpeg(imagenes, audio_final, carpeta, tiempo_img=None, durations=duraciones)

    try:
        append_intro_to_video(video_final, title_text=plan.get("youtube_title_es") or plan.get("title_es"))
    except Exception as e:
        print(f"[CUSTOM] ⚠️ No se pudo agregar intro: {e}")

    # Exporta SOLO videos largos a la carpeta raíz `videos/`.
    try:
        _export_long_videos_to_videos_dir(
            plan=plan,
            carpeta_plan=carpeta,
            video_final=video_final,
            video_con_intro=None,
        )
    except Exception:
        pass

    print("[CUSTOM] ✅ Video personalizado generado")
    return True


def renderizar_video_personalizado_desde_plan(
    carpeta_plan: str,
    *,
    voz: str,
    velocidad: str,
    interactive: bool = True,
) -> bool:
    input_path = os.path.abspath(carpeta_plan)
    if os.path.isdir(input_path):
        carpeta_plan = input_path
        plan_file = os.path.join(carpeta_plan, "custom_plan.json")
    else:
        plan_file = input_path
        carpeta_plan = os.path.dirname(plan_file) or os.getcwd()

    if not os.path.isdir(carpeta_plan):
        print(f"[CUSTOM] ❌ Carpeta inválida: {carpeta_plan}")
        return False

    if not os.path.exists(plan_file):
        print(f"[CUSTOM] ❌ No existe: {plan_file}")
        return False

    try:
        with open(plan_file, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as e:
        print(f"[CUSTOM] ❌ No se pudo leer plan: {e}")
        return False

    segmentos = plan.get("segments") or []
    if not isinstance(segmentos, list) or not segmentos:
        print("[CUSTOM] ❌ Plan sin segmentos")
        return False

    plan_changed = False
    for i, seg in enumerate(segmentos, start=1):
        if not isinstance(seg, dict):
            continue
        seg_tag = f"seg_{i:02d}"
        img_sel = seg.get("image_selection") if isinstance(seg.get("image_selection"), dict) else {}
        sel = img_sel.get("selected") if isinstance(img_sel, dict) else None
        rel = (sel or {}).get("path") if isinstance(sel, dict) else None
        abs_path = os.path.join(carpeta_plan, rel.replace("/", os.sep)) if rel else ""
        sel_title = str((sel or {}).get("title") or "").strip().lower() if isinstance(sel, dict) else ""
        sel_fallback = str((img_sel or {}).get("fallback") or "").strip().lower() if isinstance(img_sel, dict) else ""
        is_placeholder = ("placeholder" in sel_title) or (sel_fallback == "placeholder") or _es_placeholder_generado(abs_path)
        if rel and _es_imagen_valida(abs_path) and not is_placeholder:
            continue

        rec = _recuperar_candidato_local(carpeta_plan, seg_tag)
        if not rec:
            continue
        rel_rec = os.path.relpath(rec, carpeta_plan).replace("\\", "/")
        seg["image_selection"] = {
            "query": str(seg.get("image_query") or seg.get("image_prompt") or "").strip(),
            "note": str(seg.get("note") or "").strip(),
            "candidates": img_sel.get("candidates") if isinstance(img_sel, dict) else [],
            "selected": {
                "candidate_index": None,
                "score": 1,
                "url": None,
                "title": "recovered_local_candidate",
                "path": rel_rec,
            },
        }
        plan_changed = True

    if plan_changed:
        plan["segments"] = segmentos
        try:
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    try:
        removed = _limpiar_chosen_obsoletos(carpeta_plan)
        if removed > 0:
            print(f"[CUSTOM] ℹ️ Limpieza: eliminados {removed} chosen placeholder obsoletos")
    except Exception:
        pass

                                                                                                                   
                                                                                               
    faltantes: List[int] = []
    q_falt: List[str] = []
    n_falt: List[str] = []
    seen_img_fp: set[str] = set()
    for i, seg in enumerate(segmentos, start=1):
        if not isinstance(seg, dict):
            continue
        img_sel = seg.get("image_selection") if isinstance(seg.get("image_selection"), dict) else {}
        sel = img_sel.get("selected") if isinstance(img_sel, dict) else None
        rel = (sel or {}).get("path") if isinstance(sel, dict) else None
        abs_path = os.path.join(carpeta_plan, rel.replace("/", os.sep)) if rel else ""
        sel_title = str((sel or {}).get("title") or "").strip().lower() if isinstance(sel, dict) else ""
        sel_fallback = str((img_sel or {}).get("fallback") or "").strip().lower() if isinstance(img_sel, dict) else ""
        is_placeholder = ("placeholder" in sel_title) or (sel_fallback == "placeholder")
        is_dup = False
        if rel and _es_imagen_valida(abs_path) and (not _es_placeholder_generado(abs_path)):
            fp = _imagen_fingerprint(abs_path)
            if fp:
                if fp in seen_img_fp:
                    is_dup = True
                else:
                    seen_img_fp.add(fp)

        is_reused = ("reused_existing_chosen" in sel_title) or (sel_fallback == "reused")

        if (not rel) or (not _es_imagen_valida(abs_path)) or is_placeholder or _es_placeholder_generado(abs_path) or is_dup or is_reused:
            faltantes.append(i)
            base_q = (
                str(seg.get("image_query") or "").strip()
                or str(seg.get("image_prompt") or "").strip()
                or str(plan.get("brief") or "").strip()
            )
            note = str(seg.get("note") or "").strip()
            q2 = _build_better_image_query(
                base_query=base_q or "photo",
                text_es=str(seg.get("text_es") or ""),
                note=note,
                brief=str(plan.get("brief") or "").strip(),
            )
            q_falt.append(q2)
            n_falt.append(note)

    if faltantes:
        print(f"[CUSTOM] ℹ️ Faltan imágenes en {len(faltantes)}/{len(segmentos)} segmentos. Intentando autodescarga...")
        try:
            _rutas, metas = descargar_mejores_imagenes_ddg(
                carpeta_plan,
                q_falt,
                n_falt,
                max_per_query=8,
                segment_numbers=faltantes,
            )

            def _normalizar_meta_seleccion(seg_idx_1: int, meta: Dict[str, Any]) -> Dict[str, Any]:
                if not isinstance(meta, dict):
                    return meta

                seg_tag = f"seg_{int(seg_idx_1):02d}"
                selected = meta.get("selected") if isinstance(meta.get("selected"), dict) else None

                if selected:
                    rel_sel = str(selected.get("path") or "").strip()
                    abs_sel = os.path.join(carpeta_plan, rel_sel.replace("/", os.sep)) if rel_sel else ""
                    if rel_sel and _es_imagen_valida(abs_sel) and not _es_placeholder_generado(abs_sel):
                        return meta

                cands = meta.get("candidates") if isinstance(meta.get("candidates"), list) else []
                best: dict | None = None
                best_abs: str | None = None
                for c in cands:
                    if not isinstance(c, dict):
                        continue
                    rel_c = str(c.get("path") or "").strip()
                    abs_c = os.path.join(carpeta_plan, rel_c.replace("/", os.sep)) if rel_c else ""
                    if not rel_c or (not _es_imagen_valida(abs_c)) or _es_placeholder_generado(abs_c):
                        continue
                    if (best is None) or (int(c.get("score") or 1) > int(best.get("score") or 1)):
                        best = c
                        best_abs = abs_c

                if best and best_abs:
                    ext = os.path.splitext(best_abs)[1].lower() or ".jpg"
                    stable = os.path.join(carpeta_plan, f"{seg_tag}_chosen{ext}")
                    final_abs = best_abs
                    try:
                        if os.path.abspath(best_abs) != os.path.abspath(stable):
                            with open(best_abs, "rb") as src, open(stable, "wb") as dst:
                                dst.write(src.read())
                            final_abs = stable
                    except Exception:
                        pass

                    rel_final = os.path.relpath(final_abs, carpeta_plan).replace("\\", "/")
                    meta["selected"] = {
                        "candidate_index": best.get("candidate_index"),
                        "score": int(best.get("score") or 1),
                        "url": best.get("url"),
                        "title": str(best.get("title") or "auto_selected_from_candidates").strip(),
                        "path": rel_final,
                    }
                    return meta

                rec = _recuperar_candidato_local(carpeta_plan, seg_tag)
                if rec and _es_imagen_valida(rec) and not _es_placeholder_generado(rec):
                    rel_rec = os.path.relpath(rec, carpeta_plan).replace("\\", "/")
                    meta["selected"] = {
                        "candidate_index": None,
                        "score": 1,
                        "url": None,
                        "title": "auto_recovered_local_candidate",
                        "path": rel_rec,
                    }
                    meta["fallback"] = "recovered_local_candidate"
                return meta

            for seg_idx_1, meta in zip(faltantes, metas):
                if not isinstance(meta, dict):
                    continue
                if not isinstance(segmentos[seg_idx_1 - 1], dict):
                    continue
                meta = _normalizar_meta_seleccion(seg_idx_1, meta)
                segmentos[seg_idx_1 - 1]["image_selection"] = meta
            plan["segments"] = segmentos
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            print("[CUSTOM] ✅ Autodescarga completada (revisa candidatos si quieres ajustar)")
        except Exception as e:
            print(f"[CUSTOM] ⚠️ Autodescarga falló: {e}")

                                                                                            
    yt_title = str(plan.get("youtube_title_es") or plan.get("title_es") or "").strip()
    if not yt_title:
        b = str(plan.get("brief") or "Video personalizado").strip()
        yt_title = (b[:77] + "...") if len(b) > 80 else b
        plan["youtube_title_es"] = yt_title
        try:
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

                                                                          
    imagenes: List[str] = []

    def _intentar_reparar_seleccion(i_1: int, seg: Dict[str, Any]) -> str | None:
        seg_tag = f"seg_{i_1:02d}"
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            cand = os.path.join(carpeta_plan, f"{seg_tag}_chosen{ext}")
            if os.path.exists(cand) and _es_imagen_valida(cand) and not _es_placeholder_generado(cand):
                rel = os.path.relpath(cand, carpeta_plan).replace("\\", "/")
                q = str(seg.get("image_query") or seg.get("image_prompt") or "").strip()
                note = str(seg.get("note") or "").strip()
                seg["image_selection"] = {
                    "query": q,
                    "note": note,
                    "candidates": [],
                    "selected": {
                        "candidate_index": None,
                        "score": 1,
                        "url": None,
                        "title": "recovered",
                        "path": rel,
                    },
                }
                return cand

                                                                                      
        try:
            cand_dir = os.path.join(carpeta_plan, "candidates")
            if os.path.isdir(cand_dir):
                names = sorted(os.listdir(cand_dir))
                for n in names:
                    if not n.startswith(seg_tag + "_"):
                        continue
                    absp = os.path.join(cand_dir, n)
                    if not _es_imagen_valida(absp):
                        continue
                    ext = os.path.splitext(absp)[1].lower() or ".jpg"
                    stable = os.path.join(carpeta_plan, f"{seg_tag}_chosen{ext}")
                    try:
                        with open(absp, "rb") as src, open(stable, "wb") as dst:
                            dst.write(src.read())
                    except Exception:
                        stable = absp

                    rel = os.path.relpath(stable, carpeta_plan).replace("\\", "/")
                    q = str(seg.get("image_query") or seg.get("image_prompt") or "").strip()
                    note = str(seg.get("note") or "").strip()
                    seg["image_selection"] = {
                        "query": q,
                        "note": note,
                        "candidates": [],
                        "selected": {
                            "candidate_index": None,
                            "score": 1,
                            "url": None,
                            "title": "recovered_from_candidates",
                            "path": rel,
                        },
                    }
                    return stable
        except Exception:
            pass

        rec = _recuperar_candidato_local(carpeta_plan, seg_tag)
        if rec:
            rel = os.path.relpath(rec, carpeta_plan).replace("\\", "/")
            q = str(seg.get("image_query") or seg.get("image_prompt") or "").strip()
            note = str(seg.get("note") or "").strip()
            seg["image_selection"] = {
                "query": q,
                "note": note,
                "candidates": [],
                "selected": {
                    "candidate_index": None,
                    "score": 1,
                    "url": None,
                    "title": "recovered_candidate",
                    "path": rel,
                },
            }
            return rec

        reused = _reusar_chosen_existente(carpeta_plan, seg_tag)
        if reused:
            rel = os.path.relpath(reused, carpeta_plan).replace("\\", "/")
            q = str(seg.get("image_query") or seg.get("image_prompt") or "").strip()
            note = str(seg.get("note") or "").strip()
            seg["image_selection"] = {
                "query": q,
                "note": note,
                "candidates": [],
                "selected": {
                    "candidate_index": None,
                    "score": 1,
                    "url": None,
                    "title": "reused_existing_chosen",
                    "path": rel,
                },
                "fallback": "reused",
            }
            return reused

        try:
            q_base = (
                str(seg.get("image_query") or "").strip()
                or str(seg.get("image_prompt") or "").strip()
                or str(plan.get("brief") or "").strip()
                or "photo"
            )
            note = str(seg.get("note") or "").strip()
            q_fix = _build_better_image_query(
                base_query=q_base,
                text_es=str(seg.get("text_es") or ""),
                note=note,
                brief=str(plan.get("brief") or "").strip(),
            )

            rutas_fix, metas_fix = descargar_mejores_imagenes_ddg(
                carpeta_plan,
                [q_fix],
                [note],
                max_per_query=max(6, int(settings.custom_img_max_per_query)),
                segment_numbers=[i_1],
            )

            if rutas_fix and _es_imagen_valida(rutas_fix[0]) and not _es_placeholder_generado(rutas_fix[0]):
                fixed = rutas_fix[0]
                rel = os.path.relpath(fixed, carpeta_plan).replace("\\", "/")
                meta0 = metas_fix[0] if metas_fix else {}
                selected0 = meta0.get("selected") if isinstance(meta0, dict) else None
                seg["image_selection"] = {
                    "query": str((meta0 or {}).get("query") or q_fix).strip() if isinstance(meta0, dict) else q_fix,
                    "note": str((meta0 or {}).get("note") or note).strip() if isinstance(meta0, dict) else note,
                    "candidates": (meta0.get("candidates") if isinstance(meta0, dict) else []) or [],
                    "selected": {
                        "candidate_index": (selected0 or {}).get("candidate_index") if isinstance(selected0, dict) else None,
                        "score": int((selected0 or {}).get("score") or 1) if isinstance(selected0, dict) else 1,
                        "url": (selected0 or {}).get("url") if isinstance(selected0, dict) else None,
                        "title": str((selected0 or {}).get("title") or "web_autofix").strip() if isinstance(selected0, dict) else "web_autofix",
                        "path": rel,
                    },
                    "fallback": "web_autofix",
                }
                return fixed

            meta0 = metas_fix[0] if metas_fix else {}
            cands0 = meta0.get("candidates") if isinstance(meta0, dict) else []
            if isinstance(cands0, list) and cands0:
                best = None
                best_score = -1
                for c in cands0:
                    if not isinstance(c, dict):
                        continue
                    rel_c = str(c.get("path") or "").strip()
                    abs_c = os.path.join(carpeta_plan, rel_c.replace("/", os.sep)) if rel_c else ""
                    if not rel_c or (not _es_imagen_valida(abs_c)) or _es_placeholder_generado(abs_c):
                        continue
                    sc = int(c.get("score") or 1)
                    if sc > best_score:
                        best_score = sc
                        best = (c, abs_c, rel_c)

                if best:
                    c_best, abs_best, rel_best = best
                    seg["image_selection"] = {
                        "query": str((meta0 or {}).get("query") or q_fix).strip(),
                        "note": str((meta0 or {}).get("note") or note).strip(),
                        "candidates": cands0,
                        "selected": {
                            "candidate_index": c_best.get("candidate_index"),
                            "score": int(c_best.get("score") or 1),
                            "url": c_best.get("url"),
                            "title": str(c_best.get("title") or "web_autofix_best_available").strip(),
                            "path": rel_best,
                        },
                        "fallback": "web_autofix_best_available",
                    }
                    return abs_best
        except Exception:
            pass

        q_hint = str(seg.get("image_query") or seg.get("image_prompt") or plan.get("brief") or "").strip()
        comfy = _intentar_fallback_comfy(carpeta_plan, seg_tag, q_hint)
        if comfy and _es_imagen_valida(comfy) and not _es_placeholder_generado(comfy):
            rel = os.path.relpath(comfy, carpeta_plan).replace("\\", "/")
            q = str(seg.get("image_query") or seg.get("image_prompt") or "").strip()
            note = str(seg.get("note") or "").strip()
            seg["image_selection"] = {
                "query": q,
                "note": note,
                "candidates": [],
                "selected": {
                    "candidate_index": None,
                    "score": 1,
                    "url": None,
                    "title": "fallback_comfy_generated",
                    "path": rel,
                },
                "fallback": "comfy",
            }
            return comfy
                                                                    
        if not _ALLOW_PLACEHOLDER_IMAGES:
            return None

        ph = _crear_placeholder_imagen(carpeta_plan, seg_tag)
        if ph:
            rel = os.path.relpath(ph, carpeta_plan).replace("\\", "/")
            q = str(seg.get("image_query") or seg.get("image_prompt") or "").strip()
            note = str(seg.get("note") or "").strip()
            seg["image_selection"] = {
                "query": q,
                "note": note,
                "candidates": [],
                "selected": {
                    "candidate_index": None,
                    "score": 1,
                    "url": None,
                    "title": "placeholder",
                    "path": rel,
                },
                "fallback": "placeholder",
            }
            return ph
        return None

    usados_fp_render: set[str] = set()

    def _registrar_fp(path: str) -> None:
        fp = _imagen_fingerprint(path)
        if fp:
            usados_fp_render.add(fp)

    for i, seg in enumerate(segmentos, start=1):
        sel = (seg.get("image_selection") or {}).get("selected") if isinstance(seg, dict) else None
        rel = (sel or {}).get("path") if isinstance(sel, dict) else None
        if not rel:
            if isinstance(seg, dict):
                _intentar_reparar_seleccion(i, seg)
                try:
                    plan["segments"] = segmentos
                    with open(plan_file, "w", encoding="utf-8") as f:
                        json.dump(plan, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                sel = (seg.get("image_selection") or {}).get("selected")
                rel = (sel or {}).get("path") if isinstance(sel, dict) else None
            if not rel:
                print(f"[CUSTOM] ❌ Falta imagen seleccionada para segmento {i}")
                return False
        abs_path = os.path.join(carpeta_plan, rel.replace("/", os.sep))
        if (not _es_imagen_valida(abs_path)) or _es_placeholder_generado(abs_path):
            if isinstance(seg, dict):
                rep = _intentar_reparar_seleccion(i, seg)
                if rep and _es_imagen_valida(rep) and not _es_placeholder_generado(rep):
                    try:
                        plan["segments"] = segmentos
                        with open(plan_file, "w", encoding="utf-8") as f:
                            json.dump(plan, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    abs_path = rep

        if (not _es_imagen_valida(abs_path)) or _es_placeholder_generado(abs_path):
            seg_tag = f"seg_{i:02d}"
            q_hint = str(seg.get("image_query") or seg.get("image_prompt") or plan.get("brief") or "").strip()
            comfy = _intentar_fallback_comfy(carpeta_plan, seg_tag, q_hint)
            if comfy and _es_imagen_valida(comfy) and not _es_placeholder_generado(comfy):
                rel_comfy = os.path.relpath(comfy, carpeta_plan).replace("\\", "/")
                seg["image_selection"] = {
                    "query": str(seg.get("image_query") or seg.get("image_prompt") or "").strip(),
                    "note": str(seg.get("note") or "").strip(),
                    "candidates": [],
                    "selected": {
                        "candidate_index": None,
                        "score": 1,
                        "url": None,
                        "title": "fallback_comfy_replacing_invalid",
                        "path": rel_comfy,
                    },
                    "fallback": "comfy",
                }
                try:
                    plan["segments"] = segmentos
                    with open(plan_file, "w", encoding="utf-8") as f:
                        json.dump(plan, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                abs_path = comfy
            elif _ALLOW_PLACEHOLDER_IMAGES:
                ph = _crear_placeholder_imagen(carpeta_plan, seg_tag)
                if not ph:
                    print(f"[CUSTOM] ❌ Imagen inválida/corrupta para segmento {i}: {abs_path}")
                    return False
                rel_ph = os.path.relpath(ph, carpeta_plan).replace("\\", "/")
                seg["image_selection"] = {
                    "query": str(seg.get("image_query") or seg.get("image_prompt") or "").strip(),
                    "note": str(seg.get("note") or "").strip(),
                    "candidates": [],
                    "selected": {
                        "candidate_index": None,
                        "score": 1,
                        "url": None,
                        "title": "placeholder_replacing_invalid",
                        "path": rel_ph,
                    },
                    "fallback": "placeholder",
                }
                try:
                    plan["segments"] = segmentos
                    with open(plan_file, "w", encoding="utf-8") as f:
                        json.dump(plan, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                abs_path = ph
            else:
                print(
                    f"[CUSTOM] ❌ Segmento {i} sin imagen válida. "
                    "Se aborta para evitar video con pantallas negras. "
                    "(activa CUSTOM_ALLOW_PLACEHOLDER_IMAGES=1 para permitir placeholders)"
                )
                return False

        fp_cur = _imagen_fingerprint(abs_path)
        if fp_cur and fp_cur in usados_fp_render:
            seg_tag = f"seg_{i:02d}"
            rep_unique = _recuperar_candidato_local_unico(carpeta_plan, seg_tag, usados_fp_render)
            if rep_unique and _es_imagen_valida(rep_unique) and not _es_placeholder_generado(rep_unique):
                rel_new = os.path.relpath(rep_unique, carpeta_plan).replace("\\", "/")
                prev_meta = seg.get("image_selection") if isinstance(seg.get("image_selection"), dict) else {}
                seg["image_selection"] = {
                    "query": str(seg.get("image_query") or seg.get("image_prompt") or "").strip(),
                    "note": str(seg.get("note") or "").strip(),
                    "candidates": prev_meta.get("candidates") if isinstance(prev_meta, dict) else [],
                    "selected": {
                        "candidate_index": None,
                        "score": 1,
                        "url": None,
                        "title": "recovered_unique_candidate",
                        "path": rel_new,
                    },
                    "fallback": "unique_repair",
                }
                abs_path = rep_unique
                try:
                    plan["segments"] = segmentos
                    with open(plan_file, "w", encoding="utf-8") as f:
                        json.dump(plan, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        _registrar_fp(abs_path)
        imagenes.append(abs_path)

                                                       
    textos = [str(s.get("text_es") or "") for s in segmentos]
    audios: List[str] = []
    force_regen = (os.environ.get("CUSTOM_FORCE_REGEN_TTS") or "").strip().lower() in {"1", "true", "yes", "si", "sí"}

    if force_regen:
                                                                                   
                                             
        borrados = 0
        try:
            for name in os.listdir(carpeta_plan):
                low = name.lower()
                if low.startswith("audio_") and (low.endswith(".mp3") or low.endswith(".wav")):
                    try:
                        os.remove(os.path.join(carpeta_plan, name))
                        borrados += 1
                    except Exception:
                        pass
                elif low.startswith("audio_norm_") and low.endswith(".wav"):
                    try:
                        os.remove(os.path.join(carpeta_plan, name))
                        borrados += 1
                    except Exception:
                        pass
                elif low == "audio_con_silencios.wav":
                    try:
                        os.remove(os.path.join(carpeta_plan, name))
                        borrados += 1
                    except Exception:
                        pass
        except Exception:
            pass
        if borrados:
            print(f"[CUSTOM] INFO: Borrados {borrados} audios cacheados para regenerar TTS")

    if force_regen:
        print("[CUSTOM] INFO: Regenerando TTS (CUSTOM_FORCE_REGEN_TTS=1)")
        audios = tts.generar_audios(textos, carpeta_plan, voz=voz, velocidad=velocidad)
        if len(audios) != len(textos):
            print("[CUSTOM] ❌ No se pudieron generar todos los audios")
            return False
    else:
        total = len(textos)
        faltan = 0
        for i, texto in enumerate(textos):
            p_mp3 = os.path.join(carpeta_plan, f"audio_{i}.mp3")
            p_wav = os.path.join(carpeta_plan, f"audio_{i}.wav")
            if os.path.exists(p_mp3) and os.path.getsize(p_mp3) > 0:
                audios.append(p_mp3)
                continue
            if os.path.exists(p_wav) and os.path.getsize(p_wav) > 0:
                audios.append(p_wav)
                continue

            faltan += 1
            print(f"[CUSTOM] ▶️ Reanudando TTS: falta audio {i+1}/{total} (generando solo este)")
            gen = tts.generar_audios([texto], carpeta_plan, voz=voz, velocidad=velocidad, start_index=i)
            if not gen:
                print(f"[CUSTOM] ❌ No se pudo generar audio {i+1}/{total}")
                return False
            audios.append(gen[0])

        if faltan:
            print(f"[CUSTOM] ✅ TTS reanudado: generados {faltan} audios faltantes")

    duraciones = [max(0.6, audio_duration_seconds(a)) for a in audios]

                                                                          
    if bool(plan.get("seleccionar_imagenes")) and bool(interactive):
        try:
            img_meta: List[Dict[str, Any]] = []
            for seg in segmentos:
                m = seg.get("image_selection") if isinstance(seg, dict) else None
                if isinstance(m, dict):
                    img_meta.append(m)
                else:
                    q = str((seg or {}).get("image_query") or (seg or {}).get("image_prompt") or "").strip()
                    note = str((seg or {}).get("note") or "").strip()
                    img_meta.append({"query": q, "note": note, "candidates": [], "selected": None})

            imagenes, img_meta = _seleccionar_candidatos_interactivo(
                carpeta_plan,
                segmentos,
                imagenes,
                img_meta,
                audios,
                duraciones,
            )

                                                  
            for i, seg in enumerate(segmentos):
                if isinstance(seg, dict) and i < len(img_meta):
                    seg["image_selection"] = img_meta[i]
            plan["segments"] = segmentos
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CUSTOM] ⚠️ No se pudo hacer selección manual de candidatos: {e}")

                                                                                   
    if bool(interactive):
        n = len(segmentos)
        while True:
            print("\n[CUSTOM] Segmentos y selección actual:")
            for i, seg in enumerate(segmentos, start=1):
                sel = (seg.get("image_selection") or {}).get("selected") if isinstance(seg, dict) else None
                p = (sel or {}).get("path") if isinstance(sel, dict) else None
                sc = (sel or {}).get("score") if isinstance(sel, dict) else None
                note = str((seg or {}).get("note") or "").strip()
                print(f"  {i}. score={sc} img={p or 'N/A'} | {note[:80]}")

            raw = input("\nReemplazar imagen (1-{}), o ENTER para renderizar: ".format(n)).strip()
            if not raw:
                break
            try:
                idx = int(raw)
            except Exception:
                continue
            if idx < 1 or idx > n:
                continue
            src = input("Ruta local o URL de la nueva imagen: ").strip()
            try:
                new_path = _copiar_imagen_manual_a_segmento(carpeta_plan, idx, src)
                q = str((segmentos[idx - 1] or {}).get("image_query") or "").strip() or str((segmentos[idx - 1] or {}).get("image_prompt") or "").strip()
                note = str((segmentos[idx - 1] or {}).get("note") or "").strip()
                score = _puntuar_con_moondream(new_path, q, note=note)
                rel = os.path.relpath(new_path, carpeta_plan).replace("\\", "/")
                segmentos[idx - 1]["image_selection"] = {
                    "query": q,
                    "note": note,
                    "candidates": [],
                    "selected": {
                        "candidate_index": None,
                        "score": int(score),
                        "url": None,
                        "title": "manual",
                        "path": rel,
                    },
                }
                plan["segments"] = segmentos
                with open(plan_file, "w", encoding="utf-8") as f:
                    json.dump(plan, f, ensure_ascii=False, indent=2)
                print(f"[CUSTOM] ✅ Imagen reemplazada para segmento {idx} (score={score})")
            except Exception as e:
                print(f"[CUSTOM] ❌ No se pudo reemplazar: {e}")
    timeline = []
    pos = 0.0
    for q, d in zip([str(s.get("image_query") or s.get("image_prompt") or "") for s in segmentos], duraciones):
        timeline.append({"prompt": q, "start": round(pos, 2), "end": round(pos + d, 2)})
        pos += d
    plan["timeline"] = timeline
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    debug_min_seconds = int(plan.get("target_seconds") or 0)
    if not (0 < debug_min_seconds <= 10):
        debug_min_seconds = None

    plan_min_seconds = plan.get("min_seconds") if isinstance(plan, dict) else None
    plan_max_seconds = plan.get("max_seconds") if isinstance(plan, dict) else None
    try:
        plan_min_seconds = int(plan_min_seconds) if plan_min_seconds is not None else None
    except Exception:
        plan_min_seconds = None
    try:
        plan_max_seconds = int(plan_max_seconds) if plan_max_seconds is not None else None
    except Exception:
        plan_max_seconds = None

    try:
        default_short_min = int(settings.custom_min_video_sec or 40)
    except Exception:
        default_short_min = 40

    # Compatibilidad: planes antiguos guardaban 60s por defecto en modo corto.
    if plan_min_seconds is None:
        plan_min_seconds = default_short_min
    elif int(plan_min_seconds) == 60 and default_short_min == 40:
        plan_min_seconds = 40

    # En modo corto, fija rango 40-180 por defecto, conservando modo largo/debug.
    if plan_min_seconds < 300:
        if plan_min_seconds > 10 and plan_min_seconds < 40:
            plan_min_seconds = 40
        if plan_max_seconds is None:
            plan_max_seconds = 180
        else:
            plan_max_seconds = min(int(plan_max_seconds), 180)

    preserve_full_text = False
    try:
        raw_plan = plan.get("raw_plan") if isinstance(plan, dict) else None
        preserve_full_text = bool((raw_plan or {}).get("preserved_source_text")) if isinstance(raw_plan, dict) else False
    except Exception:
        preserve_full_text = False

    audio_min = debug_min_seconds if debug_min_seconds is not None else plan_min_seconds
    audio_max = None if preserve_full_text else plan_max_seconds

    audio_final = combine_audios_with_silence(
        audios,
        carpeta_plan,
        gap_seconds=0,
        min_seconds=audio_min,
        max_seconds=audio_max,
    )
    video_final = render_video_ffmpeg(imagenes, audio_final, carpeta_plan, tiempo_img=None, durations=duraciones)

    video_con_intro: str | None = None
    try:
        video_con_intro = append_intro_to_video(video_final, title_text=plan.get("youtube_title_es") or plan.get("title_es"))
    except Exception as e:
        print(f"[CUSTOM] ⚠️ No se pudo agregar intro: {e}")

    # Exporta SOLO videos largos a la carpeta raíz `videos/`.
    export_video: str | None = None
    export_info: Dict[str, Any] = {}
    try:
        export_video, export_info = _export_long_videos_to_videos_dir(
            plan=plan,
            carpeta_plan=carpeta_plan,
            video_final=video_final,
            video_con_intro=video_con_intro,
        )
    except Exception:
        export_video, export_info = (video_con_intro or video_final), {}

    try:
        import time as _time

        plan["render_done"] = True
        plan["render_done_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
        def _rel_if_exists(p: str | None) -> str | None:
            if not p:
                return None
            try:
                if not os.path.exists(p):
                    return None
            except Exception:
                return None
            try:
                return os.path.relpath(p, carpeta_plan).replace("\\", "/")
            except Exception:
                return None

        plan["render_outputs"] = {
            # Estas rutas son relativas al plan SOLO si los archivos siguen ahí (en largos se mueven).
            "video_final": _rel_if_exists(video_final),
            "video_con_intro": _rel_if_exists(video_con_intro),
            # Export para largos (ruta relativa al cwd/proyecto).
            "export": export_info or None,
            "export_video": None,
        }

        try:
            if export_video and os.path.exists(export_video):
                plan["render_outputs"]["export_video"] = os.path.relpath(export_video, os.path.abspath(".")).replace("\\", "/")
        except Exception:
            pass
        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    if export_info and export_info.get("export_video"):
        print(f"[CUSTOM] ✅ Exportado (largo) a: {export_info.get('export_video')}")
    print("[CUSTOM] ✅ Video personalizado re-renderizado")
    return True
