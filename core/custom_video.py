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

# DDG cliente: preferir paquete renombrado 'ddgs', si no, usar duckduckgo_search
try:  # nuevo paquete
    from ddgs import DDGS as DDGS
    _DDG_BACKEND = "ddgs"
except Exception:
    try:  # legacy paquete
        from duckduckgo_search import DDGS as DDGS
        _DDG_BACKEND = "duckduckgo_search"
    except Exception:
        DDGS = None
        _DDG_BACKEND = None

# Cache para saber si moondream via ollama está disponible; None = sin probar
_MOONDREAM_AVAILABLE: bool | None = None
_DEBUG_IMG_VALIDATION = (os.environ.get("DEBUG_IMG_VALIDATION") or "").strip() in {"1", "true", "True", "YES", "yes"}

import requests

# Permite ejecutar como script standalone (ajusta sys.path si no hay paquete parent)
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import tts_engine as tts
from core.video_renderer import (
    append_intro_to_video,
    audio_duration_seconds,
    combine_audios_with_silence,
    render_video_ffmpeg,
)
from utils.fs import crear_carpeta_proyecto


# Config LLM (llama3.1 en Ollama)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate").strip()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1").strip() or "llama3.1"
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "90") or "90")


def _ollama_host() -> str:
    """Devuelve el host base de Ollama (ej: http://localhost:11434).

    Soporta dos configuraciones:
    - OLLAMA_HOST=http://localhost:11434
    - OLLAMA_URL=http://localhost:11434/api/generate (legacy en este proyecto)
    """
    host = (os.environ.get("OLLAMA_HOST") or "").strip()
    if host:
        return host.rstrip("/")

    raw = (OLLAMA_URL or "").strip()
    if not raw:
        return "http://localhost:11434"

    # Si viene como /api/generate, extraemos scheme://netloc
    try:
        p = urlparse(raw)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}".rstrip("/")
    except Exception:
        pass
    # Compatibilidad con OLLAMA_URL apuntando a /api/*
    return raw.split("/api/")[0].rstrip("/")


def _ollama_api_url(path: str) -> str:
    return _ollama_host().rstrip("/") + "/" + path.lstrip("/")

# Descarga de imágenes web (Wikimedia Commons)
WIKI_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# Duración mínima por defecto para video personalizado (segundos)
# El flujo interactivo permite elegir 60s o 300s; este valor es el default.
DEFAULT_CUSTOM_MIN_VIDEO_SEC = int(os.environ.get("CUSTOM_MIN_VIDEO_SEC", "60") or "60")

# Puntaje mínimo aceptable para seleccionar una imagen (1-5). Default 1 para no bloquear.
MIN_IMG_SCORE = int(os.environ.get("CUSTOM_MIN_IMG_SCORE", "1") or "1")


def _estimar_segundos(texto: str) -> float:
    palabras = len((texto or "").split())
    if palabras <= 0:
        return 0.0
    wpm = 140.0
    estimado = (palabras / wpm) * 60.0
    estimado = estimado * 1.35 + 1.5
    return max(3.0, estimado)


def _extract_json_object(raw: str) -> Dict[str, Any]:
    """Intenta extraer un objeto JSON válido de una respuesta de LLM.

    Maneja casos comunes: texto extra alrededor, markdown fences, etc.
    """
    obj = _extract_json_value(raw)
    if isinstance(obj, dict):
        return obj
    raise ValueError("La respuesta del LLM no fue un objeto JSON")


def _extract_json_value(raw: str) -> Any:
    """Extrae un valor JSON (dict o list) tolerando fences y texto extra."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Respuesta vacía del LLM")

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    if not raw:
        raise ValueError("Respuesta vacía del LLM (tras remover markdown)")

    # Directo
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Intento: primer objeto { ... }
    o_s = raw.find("{")
    o_e = raw.rfind("}")
    if o_s != -1 and o_e != -1 and o_e > o_s:
        chunk = raw[o_s : o_e + 1]
        try:
            return json.loads(chunk)
        except Exception:
            pass

    # Intento: primer array [ ... ]
    a_s = raw.find("[")
    a_e = raw.rfind("]")
    if a_s != -1 and a_e != -1 and a_e > a_s:
        chunk = raw[a_s : a_e + 1]
        return json.loads(chunk)

    raise ValueError("No se encontró JSON parseable en la respuesta del LLM")


def _extract_json_array(raw: str) -> List[Any]:
    obj = _extract_json_value(raw)
    if isinstance(obj, list):
        return obj
    raise ValueError("La respuesta del LLM no fue un array JSON")


def _sanitize_brief_for_duration(brief: str) -> str:
    """Remueve pistas de 'Shorts/60s' para no sesgar al modelo contra el mínimo 5 min."""
    b = (brief or "").strip()
    if not b:
        return ""

    # Quitar menciones explícitas comunes
    b = re.sub(r"\b(shorts?|yt\s*shorts?)\b", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\b\d{1,4}\s*(segundos|segundo|s)\b", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\b\d{1,3}\s*(minutos|minuto|min)\b", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\s+", " ", b).strip()
    return b


def _words(text: str) -> int:
    return len((text or "").split())


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
    # Evitar cierres incompletos.
    if t.strip().endswith("..."):
        return False
    if "¿sabías que..." in t or "sabías que..." in t or "sabias que..." in t:
        return False
    # Debe cerrar con un dato curioso/impactante, no con una frase vaga.
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
    # Para shorts: 25-60 palabras; para largo: 70-140 palabras.
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
    """Compat: duración viene del input del usuario; el brief no manda."""
    _ = brief
    return int(DEFAULT_CUSTOM_MIN_VIDEO_SEC)


def _target_word_range(min_seconds: int) -> tuple[int, int]:
    """Rango de palabras recomendado para cumplir duración (aprox)."""
    # Base ~140 wpm y un colchón para pausas/entonación.
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


def _ollama_generate(prompt: str, *, temperature: float = 0.65, max_tokens: int = 900) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("response") or "").strip()
    return text


def _ollama_generate_with_timeout(prompt: str, *, temperature: float, max_tokens: int, timeout_sec: int) -> str:
    """Igual que _ollama_generate pero con timeout configurable por llamada."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
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
        # Evitar prompts enormes
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

    # 1) Generación inicial
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
            # Segundo intento: cambia levemente temperatura para destrabar salidas vacías o texto no-JSON.
            temperature = min(0.8, max(0.2, temperature + 0.05))

    # 2) Si hubo salida pero fue JSON inválido, pedir corrección del mismo JSON
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
        # Solo aceptar imágenes reales, no PDF u otros mimetypes.
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
        # Detectar formato real (a veces el URL no coincide con el mimetype real)
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

        # Validar imagen para evitar archivos corruptos que rompan ffmpeg.
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
    """Descarga una imagen a un path específico y valida que sea decodificable."""
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
        if os.path.getsize(path) < 1024:  # demasiado pequeña; probable error
            return False
        if not imghdr.what(path):
            return False
        try:
            from PIL import Image  # opcional si está instalado

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
                kwargs["backend"] = "lite"
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
        ext = url.split("?")[0].lower()
        if not any(ext.endswith(suf) for suf in (".jpg", ".jpeg", ".png", ".webp")):
            continue
        urls.append((url, title))
    return urls


def _puntuar_con_moondream(path: str, query: str, *, note: str = "") -> int:
    global _MOONDREAM_AVAILABLE
    if _MOONDREAM_AVAILABLE is False:
        raise RuntimeError("moondream no está disponible")

    # Validación via HTTP directo a Ollama (no depende del paquete `ollama` de Python)
    api_chat = _ollama_api_url("/api/chat")
    try:
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        print(f"[IMG-DDG] No se pudo leer imagen para validar: {e}")
        return 1

    note_line = (note or "").strip()
    prompt = (
        "Rate how coherent this image is with the requirement for a YouTube video segment. "
        "Return ONLY a single integer from 1 to 5. "
        "1 = completely unrelated, 2 = weak match, 3 = acceptable, 4 = good, 5 = perfect match. "
        f"Requirement: {query}. "
        + (f"Extra context: {note_line}. " if note_line else "")
        + "Do not add any other text."
    )
    try:
        payload = {
            "model": "moondream",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "options": {"temperature": 0.1},
        }
        r = requests.post(api_chat, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json() if r.content else {}
        _MOONDREAM_AVAILABLE = True
        ans = str((data.get("message") or {}).get("content") or "").strip().upper()
        if _DEBUG_IMG_VALIDATION:
            print(f"[IMG-DDG] moondream => {ans[:80]}")
        m = re.search(r"\b([1-5])\b", ans)
        if not m:
            return 1
        return int(m.group(1))
    except Exception as e:
        _MOONDREAM_AVAILABLE = False
        raise RuntimeError(f"Validación moondream no disponible ({api_chat}): {e}")


def descargar_mejores_imagenes_ddg(
    carpeta: str,
    queries: List[str],
    notes: List[str] | None = None,
    *,
    max_per_query: int = 8,
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Descarga múltiples candidatos por query, los puntúa (1-5) y elige el mejor.

    Devuelve (rutas_elegidas, meta_por_segmento).
    """
    rutas: List[str] = []
    meta_all: List[Dict[str, Any]] = []
    notes = notes or [""] * len(queries)

    cand_dir = os.path.join(carpeta, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

    for idx, q in enumerate(queries):
        note = notes[idx] if idx < len(notes) else ""
        candidatos = _buscar_ddg_imagenes(q, max_results=max_per_query)
        if not candidatos:
            meta_all.append({"query": q, "note": note, "candidates": [], "selected": None})
            continue

        seg_tag = f"seg_{idx+1:02d}"
        cand_meta: List[Dict[str, Any]] = []
        best_score = -1
        best_path = None
        best_url = ""
        best_title = ""
        best_k = None

        for k, (url, title) in enumerate(candidatos, start=1):
            # Guardar candidato con nombre estable
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
                score = _puntuar_con_moondream(saved, q, note=note)
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

        selected = None
        if best_path and best_score >= int(MIN_IMG_SCORE):
            # Copiar/normalizar a ruta estable por segmento
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
    """Permite elegir manualmente qué candidato usar por segmento.

    - No cambia audios ni TTS.
    - Solo reemplaza el archivo estable seg_XX_chosen.* antes del render.
    """
    n = min(len(segmentos), len(imagenes), len(img_meta), len(audios), len(duraciones))
    if n <= 0:
        return imagenes, img_meta

    print("\n[CUSTOM] Selección manual de imágenes (opcional)")
    print("- ENTER: mantener la imagen actual")
    print("- Número: elegir ese candidato")
    print("- s: saltar selección y continuar")

    for i in range(n):
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

        ans = input("Elegir candidato (ENTER para mantener, s para saltar): ").strip().lower()
        if ans == "s":
            print("[CUSTOM] Selección manual omitida.")
            break
        if not ans:
            continue

        try:
            chosen_k = int(ans)
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

        # Actualiza lista final de imágenes
        imagenes[i] = stable

        # Actualiza metadata seleccionada
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


def _copiar_imagen_manual_a_segmento(carpeta: str, seg_index_1: int, src: str) -> str:
    seg_tag = f"seg_{seg_index_1:02d}"
    dst = os.path.join(carpeta, f"{seg_tag}_chosen")

    src = (src or "").strip().strip('"').strip("'")
    if not src:
        raise ValueError("Ruta/URL vacía")

    if src.lower().startswith("http://") or src.lower().startswith("https://"):
        # Descargar desde URL
        url = src
        # Ext tentativa
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

    # Copiar desde archivo local
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
        # Formato corto pero >= 60s
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

    # Duración mínima seleccionada por el usuario (default 60s)
    target_seconds = int(min_seconds or DEFAULT_CUSTOM_MIN_VIDEO_SEC)
    if target_seconds not in (60, 300):
        target_seconds = max(60, target_seconds)
    brief = _sanitize_brief_for_duration(brief_in) or brief_in

    contexto = _buscar_contexto_web(brief)

    # Generación base del plan (reintentos si no cumple estructura mínima)
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

        # Validación mínima: cantidad de segmentos y palabras razonables
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

    # Enforce mínimo: extender en tandas agregando segmentos (no reescribir todo).
    est = _estimar_segundos(script)
    if est < float(target_seconds):
        for _ in range(6):
            # Cuántas palabras faltan aproximadamente
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
                # Intento de auto-fix usando el reparador ya existente
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

    # Asegurar cierre con dato curioso final (evita finales vacíos o genéricos)
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
        # Si no se puede asegurar el cierre por Ollama, se deja como está.
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
    # Queries: usar las que genera Llama por segmento (evita que se vuelva genérico).
    queries = [
        (str(s.get("image_query") or "").strip() or str(s.get("image_prompt") or "").strip() or brief)
        for s in segmentos
    ]

    notes = [str(s.get("note") or "").strip() for s in segmentos]

    # Selección robusta: descargar varios candidatos y elegir el mejor (score 1-5).
    imagenes, img_meta = descargar_mejores_imagenes_ddg(carpeta, queries, notes, max_per_query=8)
    if len(imagenes) != len(textos):
        print(f"[CUSTOM] ❌ Imágenes insuficientes: {len(imagenes)}/{len(textos)}. Se aborta.")
        return False

    # Enriquecer plan con selección de imagen por segmento
    for i, seg in enumerate(segmentos):
        meta = img_meta[i] if i < len(img_meta) else None
        if not isinstance(seg, dict) or not meta:
            continue
        seg["image_selection"] = meta

    # Generar título YouTube (corto y llamativo)
    try:
        yt_title = generar_titulo_youtube(brief, str(plan.get("script_es") or ""))
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

    # Permitir selección manual justo antes del render (sin tocar audios)
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

    # Guardar plan actualizado (con selección y título)
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


def renderizar_video_personalizado_desde_plan(carpeta_plan: str, *, voz: str, velocidad: str) -> bool:
    """Re-renderiza un video personalizado desde un custom_plan.json existente.

    Permite reemplazar imágenes por índice antes de renderizar.
    """
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
        print(f"