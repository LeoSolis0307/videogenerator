import json
import math
import os
import random
import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote_plus
import imghdr
from typing import Tuple
import time
import base64
from urllib.parse import urlparse

                                                                                
try:                 
    from ddgs import DDGS as DDGS
    _DDG_BACKEND = "ddgs"
except Exception:
    try:                  
        from duckduckgo_search import DDGS as DDGS
        _DDG_BACKEND = "duckduckgo_search"
    except Exception:
        DDGS = None
        _DDG_BACKEND = None

_DEBUG_IMG_VALIDATION = (os.environ.get("DEBUG_IMG_VALIDATION") or "").strip() in {"1", "true", "True", "YES", "yes"}

import requests

                                                                                    
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import tts_engine as tts
from core import vision_llava_phi3
from core import ollama_vram
from core.video_renderer import (
    append_intro_to_video,
    audio_duration_seconds,
    combine_audios_with_silence,
    render_video_ffmpeg,
)
from utils.fs import crear_carpeta_proyecto

                                                                                       
_VISION_AVAILABLE: bool | None = None
                              
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate").strip()
                           
                                                                                                
                                                              
                                                                                  
OLLAMA_TEXT_MODEL_SHORT = (os.environ.get("OLLAMA_TEXT_MODEL_SHORT") or "gemma2:9b").strip() or "gemma2:9b"
OLLAMA_TEXT_MODEL_LONG = (os.environ.get("OLLAMA_TEXT_MODEL_LONG") or "qwen2.5:7b").strip() or "qwen2.5:7b"


def _text_model_for_seconds(target_seconds: int | None) -> str:
    try:
        sec = int(target_seconds or 0)
    except Exception:
        sec = 0
    return OLLAMA_TEXT_MODEL_LONG if sec >= 300 else OLLAMA_TEXT_MODEL_SHORT
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "90") or "90")
                                                              
                                                                          
OLLAMA_OPTIONS_JSON = (os.environ.get("OLLAMA_OPTIONS_JSON") or "").strip()
                                                                                     
                                                                
OLLAMA_TEXT_NUM_CTX_DEFAULT = int(
    (os.environ.get("OLLAMA_TEXT_NUM_CTX") or os.environ.get("OLLAMA_NUM_CTX") or "2048").strip() or "2048"
)
                                                                  
                                                                      
VISION_TIMEOUT_SEC = int(os.environ.get("VISION_TIMEOUT_SEC", os.environ.get("MOONDREAM_TIMEOUT_SEC", "90")) or "90")
VISION_RETRIES = int(os.environ.get("VISION_RETRIES", os.environ.get("MOONDREAM_RETRIES", "3")) or "3")
VISION_MODEL = (os.environ.get("VISION_MODEL") or "minicpm-v:latest").strip() or "minicpm-v:latest"

                                                                 
DEFAULT_CUSTOM_MIN_VIDEO_SEC = int(os.environ.get("CUSTOM_MIN_VIDEO_SEC", "60") or "60")

                                                                                         
MIN_IMG_SCORE = int(os.environ.get("CUSTOM_MIN_IMG_SCORE", "1") or "1")

                                                                                           
CUSTOM_IMG_MAX_PER_QUERY = int(os.environ.get("CUSTOM_IMG_MAX_PER_QUERY", "8") or "8")

                                                   
CUSTOM_IMG_QUALITY = (os.environ.get("CUSTOM_IMG_QUALITY") or "").strip().lower()

                                                                                
                                                                  
CUSTOM_HOOK_SEGMENTS = int(os.environ.get("CUSTOM_HOOK_SEGMENTS", "2") or "2")
CUSTOM_HOOK_EXTRA_CANDIDATES = int(os.environ.get("CUSTOM_HOOK_EXTRA_CANDIDATES", "10") or "10")
CUSTOM_HOOK_MIN_IMG_SCORE = int(os.environ.get("CUSTOM_HOOK_MIN_IMG_SCORE", "3") or "3")

                                                     
WIKI_API = "https://commons.wikimedia.org/w/api.php"
                                                                                       
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

                                                                             
                                                                            
DDG_IMAGES_BACKEND = (os.environ.get("DDG_IMAGES_BACKEND") or "lite").strip().lower() or "lite"

                                                                                                 
                                                                                               
ENABLE_TEXT_RANK = (os.environ.get("CUSTOM_IMG_TEXT_RANK") or "1").strip().lower() in {"1", "true", "yes", "si", "sí"}

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
    return q or (base or "photo")


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
                                                    
    return float(hits) + (0.05 if (title or "").strip() else 0.0)

                                                                      
                                                                                                                    
_DEFAULT_BLOCKED_HOSTS = {
    "freepik.com",
    "img.freepik.com",
    "pinterest.com",
    "i.pinimg.com",
                                                                        
    "researchgate.net",
    "rgstatic.net",
}
IMG_BLOCKED_HOSTS = {
    h.strip().lower()
    for h in re.split(r"[\s,;]+", (os.environ.get("IMG_BLOCKED_HOSTS") or "").strip())
    if h.strip()
} or set(_DEFAULT_BLOCKED_HOSTS)

                                                                                       
ALLOW_AVIF = (os.environ.get("ALLOW_AVIF") or "").strip().lower() in {"1", "true", "yes", "si", "sí"}


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


_WIKIMEDIA_UA = (os.environ.get("WIKIMEDIA_USER_AGENT") or os.environ.get("IMG_WIKIMEDIA_UA") or "").strip()
_WARNED_WIKIMEDIA_UA = False


def _is_wikimedia_host(host: str) -> bool:
    host = (host or "").lower().strip()
    return host.endswith("wikimedia.org") or host.endswith("wikipedia.org")


def _maybe_warn_wikimedia_ua() -> None:
    \
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
    \
\
\
\
\
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
    seleccionar_imagenes: bool = False,
) -> str | None:
    \
\
\
\
\
    carpeta = crear_carpeta_proyecto(prefix="custom")
    try:
        plan = generar_plan_personalizado(brief, min_seconds=min_seconds)
    except Exception as e:
        print(f"[CUSTOM] ❌ No se pudo crear el plan: {e}")
        return None

                                                                       
    plan.setdefault("youtube_title_es", plan.get("title_es") or "Video personalizado")

                                                                                 
                                                                              
    try:
        if int(plan.get("target_seconds") or (min_seconds or DEFAULT_CUSTOM_MIN_VIDEO_SEC)) == 60:
            base_title = str(plan.get("youtube_title_es") or "").strip()
            script_es = str(plan.get("script_es") or "").strip()
            plan["youtube_title_es"] = _append_shorts_hashtags_to_title(
                base_title,
                brief=str(plan.get("brief") or brief or "").strip(),
                script_es=script_es,
            )
    except Exception:
        pass

                                                                                          
    try:
        yt_title_txt = os.path.join(carpeta, "youtube_title.txt")
        with open(yt_title_txt, "w", encoding="utf-8") as f:
            f.write(str(plan.get("youtube_title_es") or "").strip() + "\n")
    except Exception:
        pass
    plan["seleccionar_imagenes"] = bool(seleccionar_imagenes)

    plan_path = os.path.join(carpeta, "custom_plan.json")
    try:
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"[CUSTOM] ✅ Plan (solo guion) guardado en {plan_path}")
    except Exception as e:
        print(f"[CUSTOM] ⚠️ No se pudo guardar el plan: {e}")
        return None

                                                                      
                                                                  
    try:
        if (os.environ.get("UNLOAD_TEXT_MODEL") or "1").strip().lower() in {"1", "true", "yes", "si", "sí"}:
                                                                                 
            ok_any = False
            for m in {OLLAMA_TEXT_MODEL_SHORT, OLLAMA_TEXT_MODEL_LONG}:
                if m and ollama_vram.try_unload_model(m):
                    ok_any = True
                    print(f"[CUSTOM] ℹ️ Modelo de texto descargado: {m}")
            if not ok_any:
                pass
    except Exception:
        pass

    return carpeta


def intentar_descargar_modelo_texto() -> bool:
    \
\
\
\
    try:
        ok_any = False
        for m in {OLLAMA_TEXT_MODEL_SHORT, OLLAMA_TEXT_MODEL_LONG}:
            if m and ollama_vram.try_unload_model(m):
                ok_any = True
        return ok_any
    except Exception:
        return False


def check_text_llm_ready() -> bool:
    \
\
\
\
\
    try:
        _ = _ollama_generate("Reply only with: OK", temperature=0.0, max_tokens=8, model=OLLAMA_TEXT_MODEL_SHORT)
        return True
    except Exception as e:
        print(f"[CUSTOM] ❌ Ollama no está listo para generar guiones: {e}")
        return False


def _sanitize_brief_for_duration(brief: str) -> str:
    \
    b = (brief or "").strip()
    if not b:
        return ""

    b = re.sub(r"\b(shorts?|yt\s*shorts?)\b", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\b\d{1,4}\s*(segundos|segundo|s)\b", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\b\d{1,3}\s*(minutos|minuto|min)\b", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\s+", " ", b).strip()
    return b


def _extract_hook_from_brief(brief: str) -> str:
    \
    b = (brief or "").strip()
    if not b:
        return ""
    m = re.search(r"(?is)\bgancho\s*:\s*\"([^\"]{6,220})\"", b)
    if not m:
        return ""
    hook = (m.group(1) or "").strip()
    hook = re.sub(r"\s+", " ", hook)
    return hook


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

    return (
        "Eres guionista de YouTube Shorts en español especializado en RETENCIÓN. "
        "Reescribe SOLO el PRIMER segmento para que sea un HOOK agresivo y curioso. "
        "Debe enganchar en 1 segundo, sin saludos ni introducciones flojas.\n"
        f"BRIEF: {brief}\n"
        f"DATOS RAPIDOS: {contexto_line}\n"
        f"HOOK_OBLIGATORIO: {hook_line}\n"
        f"SEGMENTO_ACTUAL_JSON: {seg_json}\n\n"
        "REGLAS ESTRICTAS:\n"
        "- text_es DEBE EMPEZAR EXACTAMENTE con HOOK_OBLIGATORIO (primera oración).\n"
        "- PROHIBIDO: 'hola', 'bienvenidos', 'hoy', 'en este video', 'suscríbete', 'dale like'.\n"
        "- Abre un 'open loop': deja algo pendiente que se resuelve en el siguiente segmento.\n"
        "- Frases cortas, ritmo rápido, cero relleno.\n"
        "- Mantén el mismo tema; no inventes afirmaciones dudosas.\n\n"
        "Devuelve SOLO un objeto JSON con claves exactas: text_es, image_query, image_prompt, note. "
        f"text_es debe tener {wmin}-{wmax} palabras. "
        "JSON válido, nada más."
    )


def _words(text: str) -> int:
    return len((text or "").split())


def _estimar_segundos(texto: str) -> float:
    \
\
\
\
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
    return (
        "Eres guionista de YouTube en español. Reescribe SOLO el ÚLTIMO segmento para que sea un cierre con un DATO CURIOSO FINAL. "
        "Tono: energético y humano (no plano), con sarcasmo ligero y, si encaja, un chiste corto (sin groserías ni humor ofensivo). "
        "No cierres con frases vagas tipo 'exploramos el mundo mágico'. Debe quedar una idea concreta memorable.\n"
        f"BRIEF: {brief}\n"
        f"DATOS RAPIDOS: {contexto_line}\n"
        f"SEGMENTO_ACTUAL_JSON: {last_json}\n\n"
        "Devuelve SOLO un objeto JSON con claves exactas: text_es, image_query, image_prompt, note. "
        f"text_es debe tener {wmin}-{wmax} palabras. "
        "Incluye explícitamente una frase tipo 'Dato curioso final:' o '¿Sabías que...?' PERO con el dato completo en la misma oración. "
        "PROHIBIDO usar puntos suspensivos '...'. No dejes preguntas incompletas. "
        "El dato debe ser específico (nombres propios, obra/película/libro, o detalle verificable). "
        "JSON válido, nada más."
    )


def _infer_target_seconds_from_brief(brief: str) -> int:
    \
    _ = brief
    return int(DEFAULT_CUSTOM_MIN_VIDEO_SEC)


def _target_word_range(min_seconds: int) -> tuple[int, int]:
\
                                                        
    sec = max(30, int(min_seconds))
    base = int(140 * (sec / 60.0))
    return max(90, int(base * 0.95)), max(120, int(base * 1.35))


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
    \
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Respuesta vacía del LLM")

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    if not raw:
        raise ValueError("Respuesta vacía del LLM (tras remover markdown)")

    try:
        return json.loads(raw)
    except Exception:
        pass

    o_s = raw.find("{")
    o_e = raw.rfind("}")
    if o_s != -1 and o_e != -1 and o_e > o_s:
        chunk = raw[o_s : o_e + 1]
        try:
                                                                                      
            chunk = re.sub(r",\s*([\]}])", r"\1", chunk)
            return json.loads(chunk)
        except Exception:
            pass

    a_s = raw.find("[")
    a_e = raw.rfind("]")
    if a_s != -1 and a_e != -1 and a_e > a_s:
        chunk = raw[a_s : a_e + 1]
        try:
            chunk = re.sub(r",\s*([\]}])", r"\1", chunk)
            return json.loads(chunk)
        except Exception:
            pass

    raise ValueError("No se encontró JSON parseable en la respuesta del LLM")


def _extract_json_object(raw: str) -> Dict[str, Any]:
    obj = _extract_json_value(raw)
    if isinstance(obj, dict):
        return obj
                                        
    preview = (raw or "")[:500]
    print(f"[CUSTOM] ❌ La respuesta del LLM no fue un objeto JSON. Preview: {preview}...")
    raise ValueError("La respuesta del LLM no fue un objeto JSON")


def _extract_json_array(raw: str) -> List[Any]:
    obj = _extract_json_value(raw)
    if isinstance(obj, list):
        return obj
    raise ValueError("La respuesta del LLM no fue un array JSON")


def _raise_ollama_http_error(resp: requests.Response, *, model: str) -> None:
    body = (resp.text or "").strip()
                                      
    if len(body) > 1500:
        body = body[:1500] + "..."

    hint = ""
    low = body.lower()
    if "model" in low and ("not found" in low or "no such" in low or "does not exist" in low):
        hint = (
            "\n[CUSTOM] 💡 Hint: el modelo no está disponible en Ollama. "
            f"Prueba: `ollama pull {model}` o setea `OLLAMA_TEXT_MODEL` a un modelo que tengas en `ollama list`."
        )
    elif "out of memory" in low or "oom" in low or "cuda" in low or "vram" in low or "requires more system memory" in low:
        hint = (
            "\n[CUSTOM] 💡 Hint: parece falta de RAM/VRAM. "
            "Manteniendo Gemma 2, prueba `gemma2:2b` (recomendado) o `gemma2:9b` vía `OLLAMA_TEXT_MODEL`."
        )
    elif resp.status_code == 404:
        hint = "\n[CUSTOM] 💡 Hint: revisa `OLLAMA_URL` (debe apuntar a `http://localhost:11434/api/generate`)."

    msg = f"Ollama HTTP {resp.status_code} al generar con modelo '{model}'."
    if body:
        msg += f"\nOllama dice: {body}"
    if hint:
        msg += hint
    raise RuntimeError(msg)


def _ollama_extra_options() -> dict:
    if not OLLAMA_OPTIONS_JSON:
        return {}
    try:
        obj = json.loads(OLLAMA_OPTIONS_JSON)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        print("[CUSTOM] ⚠️ OLLAMA_OPTIONS_JSON no es JSON válido; ignorando")
        return {}


def _ollama_generate(prompt: str, *, temperature: float = 0.65, max_tokens: int = 900, model: str | None = None) -> str:
    model_name = (model or OLLAMA_TEXT_MODEL_SHORT).strip() or OLLAMA_TEXT_MODEL_SHORT
    extra = _ollama_extra_options()
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
                                                                                      
    if "num_ctx" not in extra:
        options["num_ctx"] = max(256, int(OLLAMA_TEXT_NUM_CTX_DEFAULT))
    options.update(extra)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            try:
                resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"No se pudo conectar a Ollama en {OLLAMA_URL}. "
                    "¿Está corriendo `ollama serve`/la app de Ollama?"
                ) from e

            if resp.status_code >= 400:
                _raise_ollama_http_error(resp, model=model_name)
            data = resp.json()
            text = (data.get("response") or "").strip()
            return text
        except Exception as e:
            last_err = e
            if attempt == 0:
                                                                          
                                                                                         
                payload["options"]["num_ctx"] = max(256, int(payload["options"].get("num_ctx") or 2048) // 2)
                payload["options"]["num_predict"] = max(128, int(payload["options"].get("num_predict") or max_tokens) // 2)
                print(
                    f"[CUSTOM] ⚠️ Reintentando Ollama con menos contexto/tokens "
                    f"(num_ctx={payload['options']['num_ctx']}, num_predict={payload['options']['num_predict']})"
                )
                continue
            break
    raise last_err


def _ollama_generate_with_timeout(
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    min_ctx: int = None,
    model: str | None = None,
) -> str:
    \
    model_name = (model or OLLAMA_TEXT_MODEL_SHORT).strip() or OLLAMA_TEXT_MODEL_SHORT
    extra = _ollama_extra_options()
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    if "num_ctx" not in extra:
                                                                            
        default_ctx = min_ctx if min_ctx else int(OLLAMA_TEXT_NUM_CTX_DEFAULT)
        options["num_ctx"] = max(256, default_ctx)
    options.update(extra)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout_sec)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"No se pudo conectar a Ollama en {OLLAMA_URL}. "
            "¿Está corriendo `ollama serve`/la app de Ollama?"
        ) from e

    if resp.status_code >= 400:
        _raise_ollama_http_error(resp, model=model_name)
    data = resp.json()
    text = (data.get("response") or "").strip()
    if not text:
        raise RuntimeError("Ollama devolvió una respuesta vacía")
    return text


def _ollama_generate_json(
    prompt: str,
    *,
    temperature: float = 0.65,
    max_tokens: int = 900,
    model: str | None = None,
) -> Dict[str, Any]:
    raw = _ollama_generate(prompt, temperature=temperature, max_tokens=max_tokens, model=model)
    return _extract_json_object(raw)


def _ollama_generate_json_with_timeout(
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    min_ctx: int = None,
    model: str | None = None,
) -> Dict[str, Any]:
    def _prompt_fix_invalid_json(bad_raw: str, err: Exception) -> str:
        bad = (bad_raw or "").strip()
                                
        if len(bad) > 8000:
            bad = bad[:8000]
        return (
            "You returned INVALID JSON. Fix it and return ONLY valid JSON. "
            "Do not add commentary, markdown, or extra keys. Preserve meaning.\n"
            f"PARSER_ERROR: {err}\n"
            "INVALID_JSON_START\n"
            f"{bad}\n"
            "INVALID_JSON_END\n"
        )
    
    def _is_memory_error(err: Exception) -> bool:
        \
        err_str = str(err).lower()
        return any(keyword in err_str for keyword in [
            "cublas", "cuda", "out of memory", "vram", 
            "llama runner process has terminated", "internal_error"
        ])

    last_err: Exception | None = None
    raw_last = ""
    current_max_tokens = max_tokens
    current_min_ctx = min_ctx
    model_name = (model or OLLAMA_TEXT_MODEL_SHORT).strip() or OLLAMA_TEXT_MODEL_SHORT

                                                                
    for attempt in range(3):
        try:
            raw_last = _ollama_generate_with_timeout(
                prompt,
                temperature=temperature,
                max_tokens=current_max_tokens,
                timeout_sec=timeout_sec,
                min_ctx=current_min_ctx,
                model=model_name,
            )
            return _extract_json_object(raw_last)
        except Exception as e:
            last_err = e
            
                                                                      
            if _is_memory_error(e):
                if attempt == 0:
                                                                   
                    current_min_ctx = int((current_min_ctx or 4096) * 0.4) if current_min_ctx else 1536
                    current_max_tokens = int(current_max_tokens * 0.4)
                    print(f"[CUSTOM] ⚠️ Error de VRAM detectado. Reintentando con ctx={current_min_ctx}, tokens={current_max_tokens}")
                    continue
                elif attempt == 1:
                                                                                      
                    current_min_ctx = int((min_ctx or 4096) * 0.15) if min_ctx else 768
                    current_max_tokens = max(500, int(max_tokens * 0.15))
                    print(f"[CUSTOM] ⚠️ Error de VRAM persiste. Último intento con ctx={current_min_ctx}, tokens={current_max_tokens}")
                    continue
            else:
                                                         
                temperature = min(0.8, max(0.2, temperature + 0.05))

                                                                               
    if raw_last:
        for _ in range(2):
            try:
                fix_prompt = _prompt_fix_invalid_json(raw_last, last_err or Exception("invalid json"))
                raw_last = _ollama_generate_with_timeout(
                    fix_prompt,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    timeout_sec=timeout_sec,
                    min_ctx=min_ctx,
                    model=model_name,
                )
                return _extract_json_object(raw_last)
            except Exception as e:
                last_err = e

    raise last_err or RuntimeError("No se pudo generar JSON con Ollama")


def _query_matches(query: str, title: str, url: str) -> bool:
    q_words = [w.lower() for w in re.split(r"[^\w]+", query) if len(w) >= 3]
    if not q_words:
        return False
    hay = 0
    text = f"{title} {url}".lower()
    for w in q_words:
        if w in text:
            hay += 1
    return hay >= max(1, len(q_words) // 2)


def _simplify_query(query: str) -> str:
    query = query.strip()
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
        return url
    return None


def _buscar_url_imagen(query: str) -> str | None:
    query_main = _simplify_query(query)
    return _wikimedia_image_url(query_main)


def _descargar_imagen(url: str, carpeta: str, idx: int) -> str | None:
    accept_avif = _can_convert_avif() or ALLOW_AVIF
    accept_header = "image/*,*/*;q=0.8"
    if accept_avif:
        accept_header = "image/avif," + accept_header
    headers = {"User-Agent": USER_AGENT}

                                                                        
    try:
        resolved = _resolve_wikimedia_thumb_via_api(url)
        if resolved:
            url = resolved
            headers = {"User-Agent": (_WIKIMEDIA_UA or USER_AGENT)}
    except Exception:
        pass

    resp = None
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers={**headers, "Accept": accept_header}, timeout=30)
                                                                     
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
            break
        except Exception as e:
            last_err = e
                                                     
            if attempt < 3:
                time.sleep(min(8.0, 1.2 * attempt + random.random()))
            resp = None
    if resp is None:
        print(f"[IMG-WEB] Falló descarga: {last_err}")
        return None

    ct = (resp.headers.get("Content-Type") or "").lower()
    if ct and ("text/html" in ct or "application/pdf" in ct or "image/svg" in ct):
        print(f"[IMG-WEB] Contenido no-imagen ({ct}), descartado: {url}")
        return None

    ext = "jpg"
    for cand in [".jpg", ".jpeg", ".png", ".webp"]:
        if cand in url.lower():
            ext = cand.replace(".", "")
            break
    os.makedirs(carpeta, exist_ok=True)
    path = os.path.join(carpeta, f"img_{idx}.{ext}")
    try:
        with open(path, "wb") as f:
            f.write(resp.content)
                                                                                 
        real_fmt = imghdr.what(path)
        if real_fmt == "jpeg":
            real_ext = "jpg"
        else:
            real_ext = real_fmt

        if real_ext and real_ext != ext:
            new_path = os.path.join(carpeta, f"img_{idx}.{real_ext}")
            try:
                os.replace(path, new_path)
                path = new_path
            except Exception:
                pass

                                                                          
        if not _es_imagen_valida(path):
            try:
                os.remove(path)
            except Exception:
                pass
            print(f"[IMG-WEB] Imagen corrupta, descartada: {url}")
            return None
                                                                           
        path = _convert_webp_to_png_if_needed(path)
        path = _convert_avif_to_png_if_needed(path)
        return path
    except Exception as e:
        print(f"[IMG-WEB] No se pudo guardar imagen: {e}")
        return None


def _descargar_imagen_a_archivo(url: str, dst_path: str) -> str | None:
\
                                                                   
                                                                                               
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
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": accept_header,
        "Accept-Language": "en-US,en;q=0.8,es-ES,es;q=0.7",
    }
    if referer:
        headers["Referer"] = referer

                                                                                     
    try:
        resolved = _resolve_wikimedia_thumb_via_api(url)
        if resolved:
            url = resolved
            headers["User-Agent"] = (_WIKIMEDIA_UA or headers.get("User-Agent") or USER_AGENT)
    except Exception:
        pass

    resp = None
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
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
            break
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(min(8.0, 1.2 * attempt + random.random()))
            resp = None
    if resp is None:
        print(f"[IMG-WEB] Falló descarga: {last_err}")
        return None

    ct = (resp.headers.get("Content-Type") or "").lower()
    if ct and ("text/html" in ct or "application/pdf" in ct or "image/svg" in ct):
        print(f"[IMG-WEB] Contenido no-imagen ({ct}), descartado: {url}")
        return None

    if ("image/avif" in ct) and not (ALLOW_AVIF or _can_convert_avif()):
        print(f"[IMG-WEB] AVIF no soportado (Content-Type: {ct}), descartado: {url}")
        return None

                                                                                   
    try:
        head = (resp.content or b"")[:512].lstrip().lower()
        if head.startswith(b"<") and (b"<html" in head or b"doctype" in head or b"captcha" in head):
            print(f"[IMG-WEB] Respuesta parece HTML/captcha, descartado: {url}")
            return None
    except Exception:
        pass

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(resp.content)

        real_fmt = imghdr.what(tmp_path)
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
            print(f"[IMG-WEB] Imagen corrupta, descartada: {url}")
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
    \
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
    \
\
\
\
\
    try:
        if not path or not os.path.exists(path):
            return path

        ext = os.path.splitext(path)[1].lower()
        fmt = (imghdr.what(path) or "").lower()
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
                                                
            return bool(imghdr.what(path))
        except Exception:
                                                                           
                                                                                                          
            fmt = (imghdr.what(path) or "").lower()
            return fmt in {"jpeg", "png", "gif", "bmp", "webp"}
    except Exception:
        return False


def _crear_placeholder_imagen(carpeta: str, seg_tag: str) -> str | None:
    \
\
\
\
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


def _buscar_ddg_imagenes(query: str, *, max_results: int = 8) -> list[Tuple[str, str]]:
    if not DDGS:
        return []

    try:
        with DDGS() as ddgs:
            kwargs = {
                "max_results": max_results,
                "safesearch": "off",
            }
            if _DDG_BACKEND == "ddgs":
                                                                             
                                                 
                kwargs["backend"] = DDG_IMAGES_BACKEND
            res = list(ddgs.images(query, **kwargs))
    except Exception as e:
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


def _puntuar_con_moondream(path: str, query: str, *, note: str = "") -> int:
    \
    global _VISION_AVAILABLE
    if _VISION_AVAILABLE is False:
        return 1

    try:
        score = vision_llava_phi3.score_image(
            path,
            query,
            note=note,
            model=VISION_MODEL,
            timeout_sec=VISION_TIMEOUT_SEC,
            retries=VISION_RETRIES,
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
    \
\
\
\
    rutas: List[str] = []
    meta_all: List[Dict[str, Any]] = []
    notes = notes or [""] * len(queries)

    cand_dir = os.path.join(carpeta, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

                                                                                       
    if max_per_query == 8 and CUSTOM_IMG_MAX_PER_QUERY != 8:
        max_per_query = max(1, int(CUSTOM_IMG_MAX_PER_QUERY))

                                                                    
    if CUSTOM_IMG_QUALITY in {"high", "alta"}:
        max_per_query = max(max_per_query, 14)
    elif CUSTOM_IMG_QUALITY in {"best", "max", "maxima", "máxima"}:
        max_per_query = max(max_per_query, 24)

    if segment_numbers is not None and len(segment_numbers) != len(queries):
        raise ValueError("segment_numbers debe tener la misma longitud que queries")

    def _uniq_candidates(cands: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        seen_u = set()
        out_u: List[Tuple[str, str]] = []
        for u, t in cands:
            if not u:
                continue
            if u in seen_u:
                continue
            seen_u.add(u)
            out_u.append((u, t))
        return out_u

    used_urls: set[str] = set()

    for idx, q in enumerate(queries):
        note = notes[idx] if idx < len(notes) else ""

        seg_n = (segment_numbers[idx] if segment_numbers is not None else (idx + 1))
        is_hook = int(seg_n) <= int(CUSTOM_HOOK_SEGMENTS)

                                                                                                                  
        local_max = int(max_per_query)
        if is_hook:
            local_max = max(local_max, int(max_per_query) + max(0, int(CUSTOM_HOOK_EXTRA_CANDIDATES)))

        candidatos = _buscar_ddg_imagenes(q, max_results=local_max)
        if is_hook:
            variants = [
                q,
                f"{q} close up photo",
                f"{q} dramatic lighting photo",
            ]
            merged: List[Tuple[str, str]] = []
            for v in variants:
                merged.extend(_buscar_ddg_imagenes(v, max_results=max(8, local_max // 2)))
            candidatos = _uniq_candidates(merged)[: max(local_max, len(candidatos))]

                                                                                   
        if ENABLE_TEXT_RANK and candidatos:
            kw = _extract_keywords(f"{q} {note}", max_keywords=10)
            if kw:
                candidatos = sorted(
                    candidatos,
                    key=lambda ut: _candidate_text_relevance(ut[0], ut[1], keywords=kw),
                    reverse=True,
                )
                candidatos = candidatos[: int(local_max)]
                                                                     
        if candidatos:
            candidatos = [(u, t) for (u, t) in candidatos if u not in used_urls]

        if not candidatos:
                                                                                  
            wiki_url = _wikimedia_image_url(_simplify_query(q))
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
                            "score": int(max(1, MIN_IMG_SCORE)),
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

        for k, (url, title) in enumerate(candidatos, start=1):
                                                  
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
                print(f"[IMG-DDG] ❌ No se pudo puntuar candidato {k} de '{q}': {e}")
                score = 1

            cand_meta.append({
                "candidate_index": k,
                "url": url,
                "title": title,
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
            wiki_url = _wikimedia_image_url(_simplify_query(q))
            if wiki_url:
                dst = os.path.join(cand_dir, f"{seg_tag}_wiki_01.jpg")
                saved = _descargar_imagen_a_archivo(wiki_url, dst)
                if saved:
                    best_path = saved
                    best_url = wiki_url
                    best_title = "Wikimedia Commons"
                    best_score = max(1, int(MIN_IMG_SCORE))
                    best_k = 1

                                                                                                         
        if is_hook and best_score < int(CUSTOM_HOOK_MIN_IMG_SCORE):
            extra_max = max(local_max, int(max_per_query) + int(CUSTOM_HOOK_EXTRA_CANDIDATES) + 12)
            variants2 = [
                q,
                f"{q} close up",
                f"{q} high contrast photo",
                f"{q} cinematic still photo",
            ]
            seen = {m.get("url") for m in cand_meta if isinstance(m, dict)}
            merged2: List[Tuple[str, str]] = []
            for v in variants2:
                merged2.extend(_buscar_ddg_imagenes(v, max_results=max(10, extra_max // 2)))
            nuevos2 = [(u, t) for (u, t) in _uniq_candidates(merged2) if u not in seen]
            for k2, (url, title) in enumerate(nuevos2, start=len(cand_meta) + 1):
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
                try:
                    score_note = (note or "").strip()
                    score_note = (
                        "HOOK (primeros segundos): elige la imagen MÁS llamativa para retener audiencia. "
                        "Prefiere close-up, alto contraste, emoción/acción, objeto icónico real; evita texto/diagramas/AI. "
                        + (score_note if score_note else "")
                    ).strip()
                    score = _puntuar_con_moondream(saved, q, note=score_note)
                except Exception as e:
                    print(f"[IMG-DDG] ❌ No se pudo puntuar candidato {k2} de '{q}': {e}")
                    score = 1
                cand_meta.append({
                    "candidate_index": k2,
                    "url": url,
                    "title": title,
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

                                                                                                        
        if (best_score < int(MIN_IMG_SCORE)) and (CUSTOM_IMG_QUALITY in {"high", "alta", "best", "max", "maxima", "máxima"}):
            extra = 32 if CUSTOM_IMG_QUALITY in {"best", "max", "maxima", "máxima"} else 20
            candidatos2 = _buscar_ddg_imagenes(q, max_results=max(max_per_query, extra))
                                             
            seen = {m.get("url") for m in cand_meta if isinstance(m, dict)}
            nuevos = [(u, t) for (u, t) in candidatos2 if u not in seen]
            for k2, (url, title) in enumerate(nuevos, start=len(cand_meta) + 1):
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
                try:
                    score = _puntuar_con_moondream(saved, q, note=note)
                except Exception as e:
                    print(f"[IMG-DDG] ❌ No se pudo puntuar candidato {k2} de '{q}': {e}")
                    score = 1
                cand_meta.append({
                    "candidate_index": k2,
                    "url": url,
                    "title": title,
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
        if best_path and best_score >= int(MIN_IMG_SCORE):
                                                           
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
    \
\
\
\
\
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
        timeout_sec=max(OLLAMA_TIMEOUT, 60),
        model=OLLAMA_TEXT_MODEL_SHORT,
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
        timeout_sec=max(OLLAMA_TIMEOUT, 60),
        model=OLLAMA_TEXT_MODEL_SHORT,
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
) -> str:
    contexto_line = contexto if contexto else "Sin contexto web disponible, apóyate en conocimiento general y datos comprobables."
    target_seconds = int(max(15, target_seconds))
    target_minutes = max(1, int(round(target_seconds / 60.0)))

    if target_seconds >= 300:
                                                                              
                                                                                   
        seg_min, seg_max = 12, 16
        total_words_min, total_words_max = 650, 950
        per_seg_min, per_seg_max = 45, 80
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

    return (
        "Eres productor de YouTube Shorts en español. Diseña un video informativo, claro y carismático para retener audiencia. "
        "Usarás imágenes REALES de internet (no IA). Necesito que alinees guion, segmentos y qué debe verse en cada momento.\n"
        f"BRIEF: {brief}\n"
        f"DATOS RAPIDOS: {contexto_line}\n\n"
        + hook_line
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
        "3) Cierre: termina con un DATO CURIOSO FINAL memorable (incluye 'Dato curioso final:' o '¿Sabías que...?').\n"
        "PROHIBIDO: terminar con frases vagas tipo 'en este video exploramos el mundo mágico' sin dar un dato final.\n\n"
        "Evita repeticiones: cero saludos; cero muletillas tipo 'hoy exploraremos'/'en este video'. Cada segmento debe aportar un dato nuevo.\n"
        "Evita redundancia: no repitas el mismo ejemplo/dato en más de un segmento. No rellenes con frases vacías. No repitas la misma pregunta retórica en varios segmentos.\n"
        "Hook único: hook_es debe ser UNA sola frase corta, no un array.\n"
        "Cierre obligatorio: el último segmento debe incluir 'Dato curioso final:' o '¿Sabías que...?' una sola vez.\n\n"
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
        "ESTRUCTURA OBLIGATORIA: intro concreta → datos curiosos específicos → cierre con DATO CURIOSO FINAL (incluye 'Dato curioso final:' o '¿Sabías que...?').\n"
        "Entrega SOLO JSON con las mismas claves: title_es, hook_es, segments (12-24), script_es. "
        "Cada segment.text_es 80-140 palabras. "
        f"script_es total ~{wmin}-{wmax} palabras. "
        "JSON valido, nada mas."
    )


def generar_plan_personalizado(brief: str, *, min_seconds: int | None = None, max_prompts: int = 12) -> Dict[str, Any]:
    brief_in = (brief or "").strip()
    if not brief_in:
        raise ValueError("Brief vacio")

                                                               
    target_seconds = int(min_seconds or DEFAULT_CUSTOM_MIN_VIDEO_SEC)
    if target_seconds not in (60, 300):
        target_seconds = max(60, target_seconds)
    brief = _sanitize_brief_for_duration(brief_in) or brief_in
    hook_hint = _extract_hook_from_brief(brief_in)

    contexto = _buscar_contexto_web(brief)

                                                                          
    plan: Dict[str, Any] | None = None
    segmentos: List[Dict[str, Any]] = []
    titulo = ""
    hook = ""
    script = ""

    model_text = _text_model_for_seconds(target_seconds)
    print(f"[CUSTOM] 🧠 Modelo texto por duración: {model_text} (target_seconds={target_seconds})")

                                                           
                                                                                      
                                                                       
                                                                            
    if target_seconds >= 300:
                                                                                  
        tokens_limit = 3600 if "qwen" in model_text.lower() else 3400
        timeout = 240
        min_ctx = 6144 if "qwen" in model_text.lower() else 5632
        print(f"[CUSTOM] 📝 Generando plan largo (~{target_seconds}s) (ctx={min_ctx}, tokens={tokens_limit})")
    else:
        tokens_limit = 2200
        timeout = 160
        min_ctx = None

    timeout = max(OLLAMA_TIMEOUT, timeout)

    for _ in range(3):
        prompt = _prompt_plan(
            brief,
            contexto,
            target_seconds=target_seconds,
            max_prompts=max_prompts,
            hook_hint=hook_hint,
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
            if seg_count < 6:
                continue
        if target_seconds >= 300 and minw < 25:
            continue
        if target_seconds < 300 and minw < 40:
            continue

        quality_issues = _plan_quality_issues(segmentos, target_seconds=target_seconds)
        if quality_issues:
            print(f"[CUSTOM] ⚠️ Plan rechazado por calidad: {', '.join(quality_issues)}")
            continue
        wmin, _wmax = _target_word_range(target_seconds)
        if total_words < int(wmin * 0.70):
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
                timeout_sec=max(OLLAMA_TIMEOUT, 120),
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
                timeout_sec=max(OLLAMA_TIMEOUT, 160),
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
                    timeout_sec=max(OLLAMA_TIMEOUT, 160),
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
            raise RuntimeError(
                f"El guion estimado ({int(est)}s) no cumple el mínimo ({target_seconds}s)."
            )

                                                                               
    try:
        if segmentos and not _es_cierre_valido(str(segmentos[-1].get("text_es") or "")):
            for attempt in range(2):
                cierre_prompt = _prompt_rewrite_closing_segment(brief, contexto, segmentos[-1], target_seconds=target_seconds)
                cierre_obj = _ollama_generate_json_with_timeout(
                    cierre_prompt,
                    temperature=0.25 if attempt == 1 else 0.35,
                    max_tokens=520,
                    timeout_sec=max(OLLAMA_TIMEOUT, 120),
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

    timeline = _generar_timeline(prompts_final, _estimar_segundos(script))

    return {
        "brief": brief_in,
        "target_seconds": int(target_seconds),
        "title_es": titulo or "Video personalizado",
        "hook_es": hook,
        "script_es": script,
        "segments": segmentos,
        "prompts": prompts_final,
        "timeline": timeline,
        "contexto_web": contexto,
        "raw_plan": plan,
    }


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
        plan = generar_plan_personalizado(brief, min_seconds=min_seconds)
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
                                                  
        if int(plan.get("target_seconds") or (min_seconds or DEFAULT_CUSTOM_MIN_VIDEO_SEC)) == 60:
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

    min_items = min(len(audios), len(imagenes), len(textos))
    audios = audios[:min_items]
    imagenes = imagenes[:min_items]
    textos = textos[:min_items]

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

    audio_final = combine_audios_with_silence(
        audios,
        carpeta,
        gap_seconds=0,
        min_seconds=None,
        max_seconds=None,
    )

    video_final = render_video_ffmpeg(imagenes, audio_final, carpeta, tiempo_img=None, durations=duraciones)

    try:
        append_intro_to_video(video_final, title_text=plan.get("youtube_title_es") or plan.get("title_es"))
    except Exception as e:
        print(f"[CUSTOM] ⚠️ No se pudo agregar intro: {e}")

    print("[CUSTOM] ✅ Video personalizado generado")
    return True


def renderizar_video_personalizado_desde_plan(
    carpeta_plan: str,
    *,
    voz: str,
    velocidad: str,
    interactive: bool = True,
) -> bool:
    \
\
\
\
    carpeta_plan = os.path.abspath(carpeta_plan)
    plan_file = carpeta_plan
    if os.path.isdir(plan_file):
        plan_file = os.path.join(carpeta_plan, "custom_plan.json")
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

                                                                                                                   
                                                                                               
    faltantes: List[int] = []
    q_falt: List[str] = []
    n_falt: List[str] = []
    for i, seg in enumerate(segmentos, start=1):
        if not isinstance(seg, dict):
            continue
        sel = (seg.get("image_selection") or {}).get("selected") if isinstance(seg.get("image_selection"), dict) else None
        rel = (sel or {}).get("path") if isinstance(sel, dict) else None
        abs_path = os.path.join(carpeta_plan, rel.replace("/", os.sep)) if rel else ""
        if (not rel) or (not _es_imagen_valida(abs_path)):
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
            for seg_idx_1, meta in zip(faltantes, metas):
                if not isinstance(meta, dict):
                    continue
                if not isinstance(segmentos[seg_idx_1 - 1], dict):
                    continue
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
        \
        seg_tag = f"seg_{i_1:02d}"
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            cand = os.path.join(carpeta_plan, f"{seg_tag}_chosen{ext}")
            if os.path.exists(cand) and _es_imagen_valida(cand):
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

        try:
            nombres = os.listdir(carpeta_plan)
        except Exception:
            nombres = []
        prefix = f"{seg_tag}_cand_"
        cand_files = sorted([n for n in nombres if n.startswith(prefix)])
        for n in cand_files:
            absp = os.path.join(carpeta_plan, n)
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
                    "title": "recovered_candidate",
                    "path": rel,
                },
            }
            return stable
                                                                    
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
        if not _es_imagen_valida(abs_path):
                                                                
            seg_tag = f"seg_{i:02d}"
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

    reuse_ok = not force_regen
    for i in range(len(textos)):
        p1 = os.path.join(carpeta_plan, f"audio_{i}.mp3")
        p2 = os.path.join(carpeta_plan, f"audio_{i}.wav")
        if os.path.exists(p1) and os.path.getsize(p1) > 0:
            audios.append(p1)
        elif os.path.exists(p2) and os.path.getsize(p2) > 0:
            audios.append(p2)
        else:
            reuse_ok = False
            break

    if not reuse_ok:
        if force_regen:
            print("[CUSTOM] INFO: Regenerando TTS (CUSTOM_FORCE_REGEN_TTS=1)")
        audios = tts.generar_audios(textos, carpeta_plan, voz=voz, velocidad=velocidad)
        if len(audios) != len(textos):
            print("[CUSTOM] ❌ No se pudieron generar todos los audios")
            return False

    duraciones = [max(0.6, audio_duration_seconds(a)) for a in audios]

                                                                          
    if bool(plan.get("seleccionar_imagenes")):
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

    audio_final = combine_audios_with_silence(audios, carpeta_plan, gap_seconds=0, min_seconds=None, max_seconds=None)
    video_final = render_video_ffmpeg(imagenes, audio_final, carpeta_plan, tiempo_img=None, durations=duraciones)

    try:
        append_intro_to_video(video_final, title_text=plan.get("youtube_title_es") or plan.get("title_es"))
    except Exception as e:
        print(f"[CUSTOM] ⚠️ No se pudo agregar intro: {e}")

    print("[CUSTOM] ✅ Video personalizado re-renderizado")
    return True
