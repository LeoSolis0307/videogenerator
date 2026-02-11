import hashlib
import json
import os
import time
from typing import List, Tuple

import requests

from utils.fs import asegurar_directorio, guardar_texto
from core.ollama_metrics import maybe_print_ollama_speed

DB_PATH = "storage/historias_gpt.json"
                                                                
MODEL_NAME_DEFAULT = "gemini-2.0-flash"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
                                                         
PROVIDER_ENV = "STORY_PROVIDER"
OLLAMA_URL_ENV = "OLLAMA_URL"
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"
OLLAMA_URL_DEFAULT = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT_ENV = "OLLAMA_TIMEOUT"
                                                                             
STORY_QUALITY_ENV = "STORY_QUALITY"
                                                              
                                                             
OLLAMA_OPTIONS_JSON_ENV = "OLLAMA_OPTIONS_JSON"
                                                                            
                                              
OLLAMA_TEXT_NUM_CTX_ENV = "OLLAMA_TEXT_NUM_CTX"
OLLAMA_NUM_CTX_ENV = "OLLAMA_NUM_CTX"
                                                                               
MAX_TOKENS_GEN = 12000

                                                                    
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _now_ts() -> int:
    return int(time.time())


def _hash_text(texto: str) -> str:
    norm = " ".join((texto or "").strip().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _get_env(nombre: str, default: str | None = None) -> str | None:
    val = os.environ.get(nombre)
    if val is not None:
        return val.strip()

    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.split("=", 1)[0].strip() == nombre:
                        return line.split("=", 1)[1].strip()
        except Exception:
            pass
    return default


def _load_db() -> dict:
    asegurar_directorio(os.path.dirname(DB_PATH))
    if not os.path.exists(DB_PATH):
        return {"items": []}
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
            if "items" not in data or not isinstance(data["items"], list):
                return {"items": []}
            return data
    except Exception:
        return {"items": []}


def _save_db(data: dict) -> None:
    asegurar_directorio(os.path.dirname(DB_PATH))
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _db_hashes(data: dict) -> set:
    return {item.get("hash", "") for item in data.get("items", []) if item.get("hash")}


def _load_api_key() -> str:
    key = _get_env("GEMINI_API_KEY")
    if key:
        return key
    raise RuntimeError("GEMINI_API_KEY no está configurada (entorno o .env)")


def _provider() -> str:
    val = (_get_env(PROVIDER_ENV, "gemini") or "gemini").strip().lower()
    if val not in {"gemini", "ollama"}:
        val = "gemini"
    return val


def _story_quality() -> str:
    return (_get_env(STORY_QUALITY_ENV, "") or "").strip().lower()


def _ollama_options_from_env() -> dict:
    raw = (_get_env(OLLAMA_OPTIONS_JSON_ENV, "") or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        print(f"[GPT] ⚠️ {OLLAMA_OPTIONS_JSON_ENV} no es JSON válido; ignorando")
        return {}


def _model_name() -> str:
    raw = _get_env("GEMINI_MODEL", MODEL_NAME_DEFAULT)
    name = (raw or "").strip()
    low = name.lower()
                            
    low = low.replace("lattest", "latest")
    low = low.replace("latestt", "latest")
    low = low.replace("fflash", "flash")

                                                                       
    allowed = {
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-2.0-pro",
    }
    if low not in allowed:
        print(f"[GPT] Modelo no permitido '{low}', usando default {MODEL_NAME_DEFAULT}")
        low = MODEL_NAME_DEFAULT

    if low != name:
        print(f"[GPT] Modelo normalizado de '{name}' a '{low}'")
    return low


def _call_genai(prompt: str, *, temperature: float = 0.85, max_tokens: int = 500) -> str:
    api_key = _load_api_key()
    model_name = _model_name()
    print(f"[GPT] Usando modelo Gemini: {model_name}")
    url = f"{API_BASE}/{model_name}:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    for intento in range(3):
        resp = requests.post(url, params={"key": api_key}, headers=headers, json=payload, timeout=45)
        if resp.status_code == 429 and intento < 2:
            espera = 2 * (intento + 1)
            print(f"[GPT] 429 Too Many Requests, reintentando en {espera}s...")
            time.sleep(espera)
            continue
        resp.raise_for_status()
        break
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Respuesta sin candidatos de Gemini")
    content = candidates[0].get("content", {})
    parts = content.get("parts") or []
    if not parts:
        raise RuntimeError("Respuesta de Gemini sin partes")
    text = parts[0].get("text", "")
    return text.strip()


def _call_ollama(prompt: str, *, max_tokens: int = 800, temperature: float = 0.85) -> str:
    url = (_get_env(OLLAMA_URL_ENV, OLLAMA_URL_DEFAULT) or OLLAMA_URL_DEFAULT).strip()
                                                                                                 
                                                                           
    model = (
        _get_env("OLLAMA_TEXT_MODEL")
        or _get_env(OLLAMA_MODEL_ENV)
        or "llama3.1:latest"
    ).strip() or "llama3.1:latest"
    timeout_s = int((_get_env(OLLAMA_TIMEOUT_ENV, "180") or "180").strip() or "180")
    print(f"[GPT] Usando Ollama modelo: {model} @ {url} (timeout {timeout_s}s)")
    def _default_num_ctx() -> int:
        raw = (
            _get_env(OLLAMA_TEXT_NUM_CTX_ENV)
            or _get_env(OLLAMA_NUM_CTX_ENV)
            or os.environ.get(OLLAMA_TEXT_NUM_CTX_ENV)
            or os.environ.get(OLLAMA_NUM_CTX_ENV)
            or "2048"
        )
        try:
            return max(256, int(str(raw).strip()))
        except Exception:
            return 2048

                                                    
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }

                                                                 
    q = _story_quality()
    if q in {"high", "alta"}:
        options.setdefault("num_ctx", 8192)
        options.setdefault("top_p", 0.9)
        options.setdefault("repeat_penalty", 1.12)
    elif q in {"best", "max", "maxima", "máxima"}:
        options.setdefault("num_ctx", 16384)
        options.setdefault("top_p", 0.9)
        options.setdefault("repeat_penalty", 1.15)

                                                                          
    options.setdefault("num_ctx", _default_num_ctx())

                                                       
    options.update(_ollama_options_from_env())

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    def _http_error_message(resp: requests.Response) -> str:
        body = (resp.text or "").strip()
        if len(body) > 1500:
            body = body[:1500] + "..."
        hint = ""
        low = body.lower()
        if "model" in low and ("not found" in low or "no such" in low or "does not exist" in low):
            hint = (
                "\n[GPT] 💡 Hint: el modelo no está disponible en Ollama. "
                f"Prueba: `ollama pull {model}` o setea `OLLAMA_TEXT_MODEL`/`OLLAMA_MODEL` a uno disponible."
            )
        elif "out of memory" in low or "oom" in low or "cuda" in low or "vram" in low or "requires more system memory" in low:
            hint = (
                "\n[GPT] 💡 Hint: parece falta de RAM/VRAM. "
                "Manteniendo Gemma 2, prueba `gemma2:2b` (recomendado) o `gemma2:9b` vía `OLLAMA_TEXT_MODEL`."
            )
        msg = f"Ollama HTTP {resp.status_code} al generar con modelo '{model}'."
        if body:
            msg += f"\nOllama dice: {body}"
        if hint:
            msg += hint
        return msg

    last_err = None
    for intento in range(2):
        try:
            try:
                resp = requests.post(url, json=payload, timeout=timeout_s)
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"No se pudo conectar a Ollama en {url}. ¿Está corriendo `ollama serve`/la app de Ollama?"
                ) from e
            if resp.status_code >= 400:
                raise RuntimeError(_http_error_message(resp))
            data = resp.json()
            maybe_print_ollama_speed(data, tag="GPT")
            if "response" not in data:
                raise RuntimeError("Respuesta de Ollama sin 'response'")
            return (data.get("response") or "").strip()
        except Exception as e:
            last_err = e
            if intento == 0:
                                                                                                 
                payload["options"]["num_predict"] = max(256, max_tokens // 2)
                try:
                    payload["options"]["num_ctx"] = max(256, int(payload["options"].get("num_ctx") or 2048) // 2)
                except Exception:
                    payload["options"]["num_ctx"] = 1024
                print(
                    "[GPT] ⚠️ Ollama timeout/fallo, reintento con menos contexto/tokens "
                    f"(num_ctx={payload['options']['num_ctx']}, num_predict={payload['options']['num_predict']})"
                )
                continue
            break
    raise last_err


def _refinar_historia(texto: str, genero: str) -> str:
    texto = (texto or "").strip()
    if not texto:
        return texto

    prompt = (
        "Eres un editor profesional de guiones/historias en español para narración. "
        f"Tu tarea: mejorar la historia del género '{genero}' manteniendo la trama, pero: "
        "(1) reduce repetición y muletillas, (2) mejora coherencia y continuidad, "
        "(3) refuerza tensión y giros sin agregar relleno, (4) mejora ritmo en párrafos. "
        "NO cambies el final a uno genérico; NO agregues listas, títulos ni encabezados. "
        "Devuelve SOLO el texto final (no JSON, no comentarios).\n\n"
        "HISTORIA:\n" + texto
    )

    prov = _provider()
                                                     
    if prov == "ollama":
        return _call_ollama(prompt, temperature=0.25, max_tokens=min(MAX_TOKENS_GEN, 9000))
    return _call_genai(prompt, temperature=0.25, max_tokens=min(MAX_TOKENS_GEN, 9000))


def _generar_historia_genero(genero: str, *, inventar: bool = False) -> str:
    prompt = (
        "Eres un escritor que crea historias largas, en español, con tono narrativo y giros que enganchen. "
        f"Genera una historia original para el género '{genero}'. "
        "Extensión objetivo: 5200-6500 palabras (mínimo 4800) para lograr ~30-40 minutos narrados. "
        "Incluye morbo, chisme, infidelidades y secretos, pero siempre con coherencia y sentido para la trama. "
        "Construye personajes, conflicto, tensión sostenida, giros, clímax y cierre claro. "
        "No agregues títulos ni bullets, solo la historia en texto corrido. Devuelve JSON con la clave 'historia'."
    )
    if inventar:
        prompt += " Si falta contexto o datos, inventa detalles morbosos y de chisme sin perder coherencia."

    prov = _provider()
    q = _story_quality()
                                                                                  
    temp = 0.85 if q in {"high", "best", "alta", "max", "maxima", "máxima"} else 0.9

    if prov == "ollama":
        raw = _call_ollama(prompt, temperature=temp, max_tokens=MAX_TOKENS_GEN)
    else:
        raw = _call_genai(prompt, temperature=temp, max_tokens=MAX_TOKENS_GEN)
    try:
        data = json.loads(raw)
        texto = data.get("historia") if isinstance(data, dict) else None
        if isinstance(texto, str) and texto.strip():
            return texto.strip()
    except Exception:
        pass
                                                                    
    out = raw
    if q in {"high", "best", "alta", "max", "maxima", "máxima"}:
        try:
            out = _refinar_historia(out, genero)
        except Exception as e:
            print(f"[GPT] ⚠️ No se pudo refinar historia: {e}")
    return out


def _validar_historia(texto: str) -> tuple[bool, str]:
    prompt = (
        "Evalúa si el siguiente texto es una historia coherente, original y completa en español. "
        "Debe tener al menos 120 palabras y no ser una lista ni un resumen. "
        "Responde JSON: {\"accept\": true/false, \"reason\": \"...\"}.\n\n" + texto
    )
    try:
        prov = _provider()
        if prov == "ollama":
            raw = _call_ollama(prompt, temperature=0.2, max_tokens=400)
        else:
            raw = _call_genai(prompt, temperature=0.2, max_tokens=400)
        data = json.loads(raw)
        ok = bool(data.get("accept"))
        reason = data.get("reason", "") if isinstance(data, dict) else ""
        return ok, reason
    except Exception as e:
        return False, f"Error de validación: {e}"


def _guardar_historia(texto: str, genero: str, idx: int) -> str:
    carpeta = os.path.join("historias", genero)
    asegurar_directorio(carpeta)
    nombre = f"historia_{int(time.time())}_{idx}.txt"
    ruta = os.path.join(carpeta, nombre)
    guardar_texto(ruta, texto)
    return ruta


def generar_historias(genero: str, cantidad: int) -> List[Tuple[str, str]]:
    data = _load_db()
    hashes = _db_hashes(data)
    aceptadas: List[Tuple[str, str]] = []

    for i in range(cantidad):
        for intento in range(4):
            texto = _generar_historia_genero(genero)
            h = _hash_text(texto)
            if h in hashes:
                print(f"[GPT] Historia repetida detectada (hash ya usado), reintentando...")
                continue

            ok, reason = _validar_historia(texto)
            if not ok:
                print(f"[GPT] Historia rechazada: {reason} -> inventando detalles")
                texto = _generar_historia_genero(genero, inventar=True)
                h = _hash_text(texto)
                ok2, reason2 = _validar_historia(texto)
                if not ok2:
                    print(f"[GPT] Historia rechazada nuevamente: {reason2} -> reintentando...")
                    continue
                ok, reason = True, reason2

            ruta = _guardar_historia(texto, genero, i)
            data.setdefault("items", []).append({
                "hash": h,
                "genero": genero,
                "path": ruta,
                "created": _now_ts(),
            })
            hashes.add(h)
            aceptadas.append((ruta, h))
            print(f"[GPT] ✅ Historia {i + 1}/{cantidad} guardada en {ruta}")
            break
        else:
            print(f"[GPT] ❌ No se pudo generar historia única para el índice {i}")

    _save_db(data)
    return aceptadas
