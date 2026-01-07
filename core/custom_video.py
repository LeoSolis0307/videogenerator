import json
import math
import os
import re
import sys
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
                           
                                                                             
                                                                     
OLLAMA_TEXT_MODEL = (os.environ.get("OLLAMA_TEXT_MODEL") or "gemma2:9b").strip() or "gemma2:9b"
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
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

                                                                             
                                                                            
DDG_IMAGES_BACKEND = (os.environ.get("DDG_IMAGES_BACKEND") or "lite").strip().lower() or "lite"

                                                                      
                                                                                                                    
_DEFAULT_BLOCKED_HOSTS = {
    "freepik.com",
    "img.freepik.com",
    "pinterest.com",
    "i.pinimg.com",
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
                max_total_len=100,
            )
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
            if ollama_vram.try_unload_model(OLLAMA_TEXT_MODEL):
                print(f"[CUSTOM] ℹ️ Modelo de texto descargado: {OLLAMA_TEXT_MODEL}")
    except Exception:
        pass

    return carpeta


def intentar_descargar_modelo_texto() -> bool:
    \
\
\
\
    try:
        return bool(ollama_vram.try_unload_model(OLLAMA_TEXT_MODEL))
    except Exception:
        return False


def check_text_llm_ready() -> bool:
    \
\
\
\
\
    try:
        _ = _ollama_generate("Reply only with: OK", temperature=0.0, max_tokens=8)
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


def _prompt_add_segments(brief: str, contexto: str, *, need_words: int, n_segments: int, last_note: str = "") -> str:
    contexto_line = contexto if contexto else "Sin contexto web disponible."
    last_line = (last_note or "").strip()
    parts = [
        "You are continuing a Spanish YouTube video script plan. ",
        "Add NEW segments that continue the narrative and add real information (no filler). ",
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
            return json.loads(chunk)
        except Exception:
            pass

    a_s = raw.find("[")
    a_e = raw.rfind("]")
    if a_s != -1 and a_e != -1 and a_e > a_s:
        chunk = raw[a_s : a_e + 1]
        return json.loads(chunk)

    raise ValueError("No se encontró JSON parseable en la respuesta del LLM")


def _extract_json_object(raw: str) -> Dict[str, Any]:
    obj = _extract_json_value(raw)
    if isinstance(obj, dict):
        return obj
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


def _ollama_generate(prompt: str, *, temperature: float = 0.65, max_tokens: int = 900) -> str:
    extra = _ollama_extra_options()
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
                                                                                      
    if "num_ctx" not in extra:
        options["num_ctx"] = max(256, int(OLLAMA_TEXT_NUM_CTX_DEFAULT))
    options.update(extra)

    payload = {
        "model": OLLAMA_TEXT_MODEL,
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
                _raise_ollama_http_error(resp, model=OLLAMA_TEXT_MODEL)
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


def _ollama_generate_with_timeout(prompt: str, *, temperature: float, max_tokens: int, timeout_sec: int) -> str:
    \
    extra = _ollama_extra_options()
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    if "num_ctx" not in extra:
        options["num_ctx"] = max(256, int(OLLAMA_TEXT_NUM_CTX_DEFAULT))
    options.update(extra)
    payload = {
        "model": OLLAMA_TEXT_MODEL,
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
        _raise_ollama_http_error(resp, model=OLLAMA_TEXT_MODEL)
    data = resp.json()
    text = (data.get("response") or "").strip()
    if not text:
        raise RuntimeError("Ollama devolvió una respuesta vacía")
    return text


def _ollama_generate_json(prompt: str, *, temperature: float = 0.65, max_tokens: int = 900) -> Dict[str, Any]:
    raw = _ollama_generate(prompt, temperature=temperature, max_tokens=max_tokens)
    return _extract_json_object(raw)


def _ollama_generate_json_with_timeout(prompt: str, *, temperature: float, max_tokens: int, timeout_sec: int) -> Dict[str, Any]:
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

    last_err: Exception | None = None
    raw_last = ""

                           
    for attempt in range(2):
        try:
            raw_last = _ollama_generate_with_timeout(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
            return _extract_json_object(raw_last)
        except Exception as e:
            last_err = e
                                                                                                          
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
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers={**headers, "Accept": "image/*,*/*;q=0.8"}, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[IMG-WEB] Falló descarga: {e}")
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

    headers = {
        "User-Agent": USER_AGENT,
                                                                                    
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,es-ES,es;q=0.7",
    }
    if referer:
        headers["Referer"] = referer
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[IMG-WEB] Falló descarga: {e}")
        return None

    ct = (resp.headers.get("Content-Type") or "").lower()
    if ct and ("text/html" in ct or "application/pdf" in ct or "image/svg" in ct):
        print(f"[IMG-WEB] Contenido no-imagen ({ct}), descartado: {url}")
        return None

    if ("image/avif" in ct) and not ALLOW_AVIF:
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
        return final_path
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"[IMG-WEB] No se pudo guardar imagen: {e}")
        return None


def _es_imagen_valida(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < 1024:                                     
            return False
        if not imghdr.what(path):
            return False
        try:
            from PIL import Image                              

            with Image.open(path) as img:
                img.verify()
        except ImportError:
            pass
        except Exception:
            return False
        return True
    except Exception:
        return False


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
        "Reglas: máximo 70 caracteres, sin comillas, sin hashtags, sin emojis. "
        "Debe ser apto para subir tal cual.\n"
        f"TEMA/BRIEF: {brief}\n"
        f"GUIÓN (resumen): {script_snip}\n\n"
        "Devuelve SOLO el título."
    )


def generar_titulo_youtube(brief: str, script_es: str) -> str:
    prompt = _prompt_titulo_youtube(brief, script_es)
    titulo = _ollama_generate_with_timeout(prompt, temperature=0.55, max_tokens=80, timeout_sec=max(OLLAMA_TIMEOUT, 60))
    titulo = (titulo or "").strip().strip('"').strip("'")
    titulo = re.sub(r"\s+", " ", titulo).strip()
    if not titulo:
        raise RuntimeError("Ollama no devolvió un título")
    if len(titulo) > 70:
        titulo = titulo[:70].rstrip()
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
    raw = _ollama_generate_with_timeout(prompt, temperature=0.35, max_tokens=60, timeout_sec=max(OLLAMA_TIMEOUT, 60))
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


def _append_shorts_hashtags_to_title(title_es: str, *, brief: str, script_es: str, max_total_len: int = 100) -> str:
    base = re.sub(r"\s+", " ", (title_es or "").strip())
    if not base:
        base = "Video personalizado"

                                                       
    if "#" in base:
        return base

    tags = _generar_hashtags_shorts(brief, base, script_es)
    tags_str = " ".join(tags)
    if not tags_str:
        return base

                                                                 
    max_total_len = int(max_total_len or 100)
    allowed_title_len = max(10, max_total_len - 1 - len(tags_str))
    if len(base) > allowed_title_len:
        base = base[:allowed_title_len].rstrip()
                                                                       
        base = base.rstrip("-:|,.;")

    return f"{base} {tags_str}".strip()


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


def _prompt_plan(brief: str, contexto: str, *, target_seconds: int, max_prompts: int = 12) -> str:
    contexto_line = contexto if contexto else "Sin contexto web disponible, apóyate en conocimiento general y datos comprobables."
    target_seconds = int(max(15, target_seconds))
    target_minutes = max(1, int(round(target_seconds / 60.0)))

    if target_seconds >= 300:
        seg_min, seg_max = 12, 24
        total_words_min, total_words_max = 850, 1200
        per_seg_min, per_seg_max = 80, 140
    else:
                                   
        seg_min, seg_max = 6, 10
        total_words_min, total_words_max = _target_word_range(target_seconds)
        per_seg_min, per_seg_max = 22, 55

    return (
        "Eres productor de YouTube en español. Diseña un video informativo, claro y carismático para retener audiencia. "
        "Usarás imágenes REALES de internet (no IA). Necesito que alinees guion, segmentos y qué debe verse en cada momento.\n"
        f"BRIEF: {brief}\n"
        f"DATOS RAPIDOS: {contexto_line}\n\n"
        f"OBJETIVO: El guion narrado debe durar ~{target_seconds} segundos (~{target_minutes} min). "
        "NO rellenes con silencio: escribe contenido real.\n\n"
        "ESTRUCTURA OBLIGATORIA (en este orden):\n"
        "1) Introducción: presenta el tema elegido de forma concreta (qué es y por qué importa).\n"
        "2) Datos curiosos: entrega datos específicos, ejemplos, mini-historia o contexto; evita generalidades.\n"
        "3) Cierre: termina con un DATO CURIOSO FINAL memorable (incluye 'Dato curioso final:' o '¿Sabías que...?').\n"
        "PROHIBIDO: terminar con frases vagas tipo 'en este video exploramos el mundo mágico' sin dar un dato final.\n\n"
        "Entrega SOLO JSON con estas claves:\n"
        "- title_es: titulo atractivo (<=80 chars).\n"
        "- hook_es: 1-2 frases de gancho inmediato.\n"
        f"- segments: lista de {seg_min}-{seg_max} objetos, estricto orden cronologico. Cada objeto: {{\n"
        f"    text_es: parte del guion en español ({per_seg_min}-{per_seg_max} palabras) para narrar en TTS;\n"
        "    image_query: frase corta en ingles para buscar foto real exacta (ej: 'elder wand prop from deathly hallows movie');\n"
        "    image_prompt: descripcion en ingles de la escena para contexto visual;\n"
        "    note: detalle breve de lo que debe verse para validar que coincide.\n"
        "  }.\n"
        "- script_es: concatenación de todos los text_es en orden, como guion completo.\n"
        "Reglas: prioriza objetos/lugares/personas reales, evita conceptos abstractos. Para nombres concretos (ej. 'varita de saúco') usa queries precisas del item real. "
        f"Verificación interna: el script_es final debe tener aprox {total_words_min}-{total_words_max} palabras. "
        "JSON valido, nada mas."
    )


def _prompt_expand_to_min_duration(brief: str, contexto: str, plan_raw: Dict[str, Any], *, min_seconds: int) -> str:
    contexto_line = contexto if contexto else "Sin contexto web disponible."
    raw_str = json.dumps(plan_raw, ensure_ascii=False)[:4000]
    wmin, wmax = _target_word_range(min_seconds)
    return (
        "Reescribe y EXPANDE el plan para cumplir duración mínima sin añadir relleno vacío. "
        "Mantén el mismo tema y estructura, pero agrega más segmentos y más detalle narrativo.\n"
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

    contexto = _buscar_contexto_web(brief)

                                                                          
    plan: Dict[str, Any] | None = None
    segmentos: List[Dict[str, Any]] = []
    titulo = ""
    hook = ""
    script = ""

    for _ in range(3):
        prompt = _prompt_plan(brief, contexto, target_seconds=target_seconds, max_prompts=max_prompts)
        plan = _ollama_generate_json_with_timeout(prompt, temperature=0.62, max_tokens=2200, timeout_sec=max(OLLAMA_TIMEOUT, 160))

        titulo = str(plan.get("title_es") or "").strip()
        hook = str(plan.get("hook_es") or "").strip()
        script = str(plan.get("script_es") or "").strip()

        raw_segments = plan.get("segments") or []
        segmentos = []
        if isinstance(raw_segments, list):
            for s in raw_segments:
                if not isinstance(s, dict):
                    continue
                texto = str(s.get("text_es") or "").strip()
                if not texto:
                    continue
                query = str(s.get("image_query") or "").strip()
                iprompt = str(s.get("image_prompt") or "").strip() or query
                note = str(s.get("note") or "").strip()
                segmentos.append({
                    "text_es": texto,
                    "image_query": query,
                    "image_prompt": iprompt,
                    "note": note,
                })

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
        if minw < 40:
            continue
        wmin, _wmax = _target_word_range(target_seconds)
        if total_words < int(wmin * 0.70):
            continue

        break

    if not plan or not segmentos:
        raise RuntimeError("El plan no devolvió 'segments' válidos")

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
            raw = _ollama_generate_with_timeout(add_prompt, temperature=0.55, max_tokens=1800, timeout_sec=max(OLLAMA_TIMEOUT, 160))
            try:
                arr = _extract_json_array(raw)
            except Exception:
                                                                      
                fix = (
                    "You returned INVALID JSON array. Fix it and return ONLY a valid JSON array.\n"
                    "INVALID_JSON_START\n" + (raw or "")[:8000] + "\nINVALID_JSON_END\n"
                )
                raw2 = _ollama_generate_with_timeout(fix, temperature=0.2, max_tokens=1800, timeout_sec=max(OLLAMA_TIMEOUT, 160))
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
                                                                                     
    queries = [
        (str(s.get("image_query") or "").strip() or str(s.get("image_prompt") or "").strip() or brief)
        for s in segmentos
    ]

    notes = [str(s.get("note") or "").strip() for s in segmentos]

                                                                                   
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
                max_total_len=100,
            )
        plan["youtube_title_es"] = yt_title
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
            q = (str(seg.get("image_query") or "").strip() or str(seg.get("image_prompt") or "").strip() or str(plan.get("brief") or "").strip())
            q_falt.append(q or "photo")
            n_falt.append(str(seg.get("note") or "").strip())

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
            print(f"[CUSTOM] ❌ Imagen inválida/corrupta para segmento {i}: {abs_path}")
            return False
        imagenes.append(abs_path)

                                                       
    textos = [str(s.get("text_es") or "") for s in segmentos]
    audios: List[str] = []
    reuse_ok = True
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
