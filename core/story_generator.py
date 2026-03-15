import hashlib
import json
import os
import re
import sys
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
                                                                               
MAX_TOKENS_GEN = 1800
STORY_MAX_ATTEMPTS_ENV = "STORY_MAX_ATTEMPTS"

                                                                    
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _now_ts() -> int:
    return int(time.time())


def _hash_text(texto: str) -> str:
    norm = " ".join((texto or "").strip().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _word_count(texto: str) -> int:
    return len(re.findall(r"\b\w+\b", (texto or ""), flags=re.UNICODE))


def _log(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
        safe = str(message).encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe)


def _has_first_person(texto: str) -> bool:
    low = (texto or "").lower()
    markers = (
        " yo ", " me ", " mi ", " mis ", " conmigo ", "mío", "mía", "mías", "míos",
        " estaba ", " fui ", " tenía ", " sentí ", " pensé ", " decidí ",
    )
    padded = f" {low} "
    hits = sum(1 for m in markers if m in padded)
    return hits >= 3


def _has_climax_marker(texto: str) -> bool:
    low = (texto or "").lower()
    markers = (
        "de repente",
        "en ese momento",
        "entonces",
        "cuando lo vi",
        "cuando me dijo",
        "todo explotó",
        "me enfrenté",
        "lo confronté",
        "fue ahí cuando",
        "ese día",
        "esa noche",
        "ahí me di cuenta",
        "descubrí",
        "me confesó",
        "vi los mensajes",
        "lo vi con",
        "la encontré con",
    )
    if any(m in low for m in markers):
        return True

    turning_point_patterns = (
        r"\bdescubr\w*\b",
        r"\bconfront\w*\b",
        r"\bencar\w*\b",
        r"\bconfes\w*\b",
        r"\bexplot\w*\b",
        r"\bse acab[oó]\b",
        r"\blo dej[eé]\b",
        r"\ble dije que no\b",
    )
    return any(re.search(p, low) for p in turning_point_patterns)


def _has_resolution_in_tail(texto: str) -> bool:
    low = (texto or "").lower().strip()
    if not low:
        return False
    tail = low[-max(220, int(len(low) * 0.35)):]
    markers = (
        "al final",
        "esa noche",
        "desde entonces",
        "terminé",
        "decidí",
        "corté",
        "me fui",
        "renuncié",
        "lo bloqueé",
        "nunca volvió",
        "y ahí entendí",
        "desde ese día",
        "le dije que no",
        "no volví",
        "cerré esa puerta",
    )
    return any(m in tail for m in markers)


def _has_soft_closure_in_tail(texto: str) -> bool:
    low = (texto or "").lower().strip()
    if not low:
        return False
    tail = low[-max(220, int(len(low) * 0.35)):]
    soft_markers = (
        "amistad",
        "amigos",
        "en buenos términos",
        "con respeto",
        "límites claros",
        "límites en la amistad",
        "seguimos hablando",
        "quedamos bien",
    )
    return any(m in tail for m in soft_markers)


def _has_hard_rejection_tail(texto: str) -> bool:
    low = (texto or "").lower().strip()
    if not low:
        return False
    tail = low[-max(260, int(len(low) * 0.4)):]

    reject_patterns = (
        r"\ble dije que no\b",
        r"\blo? rechac\w*\b",
        r"\bno vuelvo contigo\b",
        r"\bno quier[oa] volver\b",
        r"\bse acab[oó]\b",
        r"\bno regres[ée]\b",
    )
    drama_patterns = (
        r"\bhizo un drama\b",
        r"\bllor\w*\b",
        r"\bgrit\w*\b",
        r"\bsuplic\w*\b",
        r"\binsist\w*\b",
        r"\breclam\w*\b",
    )
    walkaway_patterns = (
        r"\bme fui\b",
        r"\bme di la vuelta\b",
        r"\blo? dej[eé] hablando sol[oa]\b",
        r"\bcerr[eé] la puerta\b",
        r"\blo? bloque[eé]\b",
    )

    has_reject = any(re.search(p, tail) for p in reject_patterns)
    has_drama = any(re.search(p, tail) for p in drama_patterns)
    has_walkaway = any(re.search(p, tail) for p in walkaway_patterns)
    return has_reject and (has_drama or has_walkaway)


def _has_generic_ending(texto: str) -> bool:
    low = (texto or "").lower().strip()
    if not low:
        return True
    tail = low[-220:]
    generic = (
        "aprendí que",
        "la moraleja",
        "la lección",
        "en conclusión",
        "todo pasa por algo",
        "hoy soy más fuerte",
    )
    return any(g in tail for g in generic)


def _genre_pref_infidelity(genero: str) -> bool:
    g = (genero or "").strip().lower()
    return "drama" in g or "relaciones" in g


def _has_infidelity_markers(texto: str) -> bool:
    low = (texto or "").lower()
    markers = (
        "infidel",
        "me engañ",
        "me fue infiel",
        "me cambió por",
        "otro hombre",
        "otra mujer",
        "mensaje oculto",
        "chat borrado",
        "descubrí",
        "me dejó por",
        "se veía con",
        "tenía otra",
        "tenía otro",
        "me traicion",
    )
    if any(m in low for m in markers):
        return True
    patterns = (
        r"\binfidel\w*\b",
        r"\bengañ\w*\b",
        r"\btraicion\w*\b",
        r"\botra (persona|mujer|chica|hombre|relaci[oó]n)\b",
        r"\bme dej[oó] por\b",
    )
    return any(re.search(p, low) for p in patterns)


def _has_breakdown_markers(texto: str) -> bool:
    low = (texto or "").lower()
    markers = (
        "me derrumb",
        "toqué fondo",
        "no podía",
        "no dormía",
        "lloré",
        "ansiedad",
        "depres",
        "me quedé solo",
    )
    if any(m in low for m in markers):
        return True
    patterns = (
        r"\bderrumb\w*\b",
        r"\bfondo\b",
        r"\bansiedad\b",
        r"\bdepres\w*\b",
        r"\bllor[ée]\b",
        r"\bno dorm[ií]a\b",
        r"\bsolo\b",
    )
    return any(re.search(p, low) for p in patterns)


def _has_rebuild_markers(texto: str) -> bool:
    low = (texto or "").lower()
    markers = (
        "empecé terapia",
        "fui al gimnasio",
        "cambié mi rutina",
        "ahorré",
        "subí de puesto",
        "aprendí",
        "emprendí",
        "me enfoqué",
        "volví a estudiar",
        "conseguí trabajo",
        "me discipliné",
        "dejé de buscarla",
    )
    if any(m in low for m in markers):
        return True
    patterns = (
        r"\bterapia\b",
        r"\bgimnasio\b",
        r"\bestudi\w*\b",
        r"\btrabaj\w*\b",
        r"\bahorr\w*\b",
        r"\bme enfoqu\w*\b",
        r"\bdisciplin\w*\b",
        r"\bdej[eé] de buscarla\b",
    )
    return any(re.search(p, low) for p in patterns)


def _has_ex_comeback_marker(texto: str) -> bool:
    low = (texto or "").lower()
    markers = (
        "quiso volver",
        "me buscó",
        "me escribió otra vez",
        "me pidió volver",
        "me llamó llorando",
        "volvió a escribirme",
        "apareció otra vez",
    )
    if any(m in low for m in markers):
        return True
    patterns = (
        r"\b(volvi[oó]|volver)\b",
        r"\bme busc\w*\b",
        r"\bme escribi[oó] otra vez\b",
        r"\bme llam[oó] llorando\b",
        r"\bintent[oó] regresar\b",
    )
    return any(re.search(p, low) for p in patterns)


def _count_true(flags: list[bool]) -> int:
    return sum(1 for f in flags if f)


def _normalizar_drama_infidelidad(texto: str, genero: str) -> str:
    t = (texto or "").strip()
    if not t or not _genre_pref_infidelity(genero):
        return t

    chunks: list[str] = [t]

    if not _has_infidelity_markers(t):
        chunks.append("La verdad es que me fue infiel y ahí entendí que todo lo demás era una mentira.")

    if not _has_breakdown_markers(t):
        chunks.append("Ese golpe me dejó mal por semanas: no dormía bien y me quebré por dentro.")

    if not _has_rebuild_markers(t):
        chunks.append("Después me enfoqué en mí, cambié mi rutina y reconstruí mi vida paso a paso.")

    if not _has_ex_comeback_marker(t):
        chunks.append("Meses después apareció otra vez para pedirme que regresáramos.")

    joined = "\n\n".join(chunks)

    if _has_soft_closure_in_tail(joined) or not _has_hard_rejection_tail(joined):
        joined += (
            "\n\nLe dije que no, sin rodeos. Ella hizo un drama, lloró y me reclamó en la puerta. "
            "No discutí más: me di la vuelta, la dejé hablando sola y me fui."
        )

    return joined.strip()


def _local_story_gate(texto: str, genero: str = "") -> tuple[bool, str]:
    t = (texto or "").strip()
    if not t:
        return False, "texto vacío"

    low = t.lower()
    if low.startswith("{") or '"historia"' in low or "```" in low:
        return False, "formato inválido (json/código)"

    wc = _word_count(t)
    if wc < 90 or wc > 380:
        return False, f"longitud fuera de rango ({wc} palabras)"

    if not _has_first_person(t):
        return False, "no está en primera persona creíble"

    if _has_generic_ending(t):
        return False, "final genérico"

    if _genre_pref_infidelity(genero):
        has_infidelity = _has_infidelity_markers(t)
        has_breakdown = _has_breakdown_markers(t)
        has_rebuild = _has_rebuild_markers(t)
        has_comeback = _has_ex_comeback_marker(t)

        if not has_infidelity:
            return False, "falta eje de infidelidad"

        arc_score = _count_true([has_breakdown, has_rebuild, has_comeback])
        if arc_score < 1:
            return False, "falta arco de caída/superación/regreso"

        if _has_soft_closure_in_tail(t):
            return False, "cierre blando/amistoso"

        if not _has_hard_rejection_tail(t):
            return False, "cierre sin rechazo firme y drama"

    return True, "ok"


def _extract_historia_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            historia = data.get("historia")
            if isinstance(historia, str) and historia.strip():
                return historia.strip()
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidate = m.group(0)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                historia = data.get("historia")
                if isinstance(historia, str) and historia.strip():
                    return historia.strip()
        except Exception:
            pass

    return text


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


def _max_attempts() -> int:
    raw = (_get_env(STORY_MAX_ATTEMPTS_ENV, "5") or "5").strip()
    try:
        return max(1, min(12, int(raw)))
    except Exception:
        return 5


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
        _log(f"[GPT] ⚠️ {OLLAMA_OPTIONS_JSON_ENV} no es JSON válido; ignorando")
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
        _log(f"[GPT] Modelo no permitido '{low}', usando default {MODEL_NAME_DEFAULT}")
        low = MODEL_NAME_DEFAULT

    if low != name:
        _log(f"[GPT] Modelo normalizado de '{name}' a '{low}'")
    return low


def _call_genai(prompt: str, *, temperature: float = 0.85, max_tokens: int = 500) -> str:
    api_key = _load_api_key()
    model_name = _model_name()
    _log(f"[GPT] Usando modelo Gemini: {model_name}")
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
            _log(f"[GPT] 429 Too Many Requests, reintentando en {espera}s...")
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
    _log(f"[GPT] Usando Ollama modelo: {model} @ {url} (timeout {timeout_s}s)")
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
                _log(
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
        "(5) deja el texto entre 170 y 240 palabras. "
        "NO cambies el final a uno genérico; NO agregues listas, títulos ni encabezados. "
        "Devuelve SOLO el texto final (no JSON, no comentarios).\n\n"
        "HISTORIA:\n" + texto
    )

    prov = _provider()
                                                     
    if prov == "ollama":
        return _call_ollama(prompt, temperature=0.25, max_tokens=min(MAX_TOKENS_GEN, 1600))
    return _call_genai(prompt, temperature=0.25, max_tokens=min(MAX_TOKENS_GEN, 1600))


def _reparar_historia_para_gate(texto: str, genero: str, motivo: str) -> str:
    t = (texto or "").strip()
    if not t:
        return t

    prompt = (
        "Reescribe esta historia para cumplir formato viral corto y realista. "
        "Obligatorio: 170-240 palabras, primera persona, gancho fuerte, clímax claro y cierre contundente sin moralina. "
        "No JSON, no markdown, no títulos. Solo texto final. "
        f"Motivo del rechazo: {motivo}. "
        f"Género: {genero}. "
    )
    if _genre_pref_infidelity(genero):
        prompt += (
            "Debe incluir infidelidad, caída emocional, reconstrucción personal con acciones concretas, "
            "intento de regreso de la ex/pareja y decisión final firme. "
            "Cierre obligatorio: el narrador rechaza de forma directa, la ex hace drama (llora, insiste o reclama) "
            "y el narrador se va o la deja hablando sola. "
            "Prohibido terminar en amistad, cordialidad o límites amistosos. "
        )
    prompt += "\n\nHISTORIA A REESCRIBIR:\n" + t

    prov = _provider()
    if prov == "ollama":
        raw = _call_ollama(prompt, temperature=0.25, max_tokens=min(MAX_TOKENS_GEN, 1200))
    else:
        raw = _call_genai(prompt, temperature=0.25, max_tokens=min(MAX_TOKENS_GEN, 1200))
    repaired = _extract_historia_text(raw)
    return _normalizar_drama_infidelidad(repaired, genero)


def _generar_historia_genero(genero: str, *, inventar: bool = False) -> str:
    prompt = (
        "Eres editor de historias virales para narración corta en español. "
        f"Genera una historia original del género '{genero}' con estilo de testimonio real (primera persona). "
        "Debe sonar creíble: detalles concretos (fechas aproximadas, lugares cotidianos, acciones verificables), "
        "sin metáforas literarias, sin tono de novela, sin frases grandilocuentes. "
        "Estructura obligatoria: gancho inicial (1-2 frases), contexto breve, conflicto central, clímax fuerte y cierre contundente. "
        "El clímax debe ocurrir en el último tercio del texto y el cierre debe explicar consecuencia concreta e irreversible. "
        "Extensión obligatoria: 170 a 240 palabras (aprox. 900-1300 caracteres). "
        "No agregues títulos, bullets ni encabezados. "
        "Devuelve SOLO texto final, sin JSON, sin markdown, sin comillas envolventes."
    )
    if _genre_pref_infidelity(genero):
        prompt += (
            " Tema prioritario: infidelidad y superación personal. "
            "Arco obligatorio en este orden: "
            "(1) descubro engaño o cambio por otra persona, "
            "(2) me derrumbo de forma realista, "
            "(3) reconstruyo mi vida con acciones concretas, "
            "(4) la ex/pareja intenta regresar, "
            "(5) cierre duro: la rechazo de frente, ella hace drama y yo me voy. "
            "Debe sentirse contado por gente normal, con frases simples y directas, "
            "sin poesía ni moralina. "
            "No cierres con amistad, perdón suave ni términos cordiales. "
            "Incluye micro-ganchos para retención cada 2-3 frases (revelación, consecuencia o giro)."
        )
    if inventar:
        prompt += (
            " Si falta contexto, inventa solo detalles plausibles y cotidianos "
            "(sin elementos fantasiosos ni melodrama exagerado)."
        )

    prov = _provider()
    q = _story_quality()
                                                                                  
    temp = 0.85 if q in {"high", "best", "alta", "max", "maxima", "máxima"} else 0.9

    if prov == "ollama":
        raw = _call_ollama(prompt, temperature=temp, max_tokens=MAX_TOKENS_GEN)
    else:
        raw = _call_genai(prompt, temperature=temp, max_tokens=MAX_TOKENS_GEN)
    extracted = _extract_historia_text(raw)
    if extracted:
        raw = extracted

    raw = _normalizar_drama_infidelidad(raw, genero)
                                                                    
    out = raw
    if q in {"high", "best", "alta", "max", "maxima", "máxima"}:
        try:
            out = _refinar_historia(out, genero)
        except Exception as e:
            _log(f"[GPT] ⚠️ No se pudo refinar historia: {e}")
    return out


def _validar_historia(texto: str) -> tuple[bool, str]:
    prompt = (
        "Evalúa si el texto cumple formato viral corto y realista en español. "
        "Criterios obligatorios: "
        "(1) 130-280 palabras, "
        "(2) primera persona, "
        "(3) conflicto + clímax + desenlace claro, "
        "(3.1) el clímax aparece antes del cierre y el cierre deja consecuencia concreta, "
        "(4) tono creíble/cotidiano (no literario/novelesco), "
        "(5) sin listas ni encabezados. "
        "(6) rechazar finales genéricos tipo moraleja/autoayuda. "
        "(7) rechazar si el texto viene en JSON, markdown o formato de código. "
        "(8) para drama/relaciones, exigir infidelidad + caída + superación + intento de regreso + cierre duro: "
        "rechazo directo, drama del ex y salida del narrador. "
        "(9) rechazar cierres amistosos/cordiales. "
        "Responde SOLO JSON: {\"accept\": true/false, \"reason\": \"...\"}.\n\n" + texto
    )
    try:
        prov = _provider()
        if prov == "ollama":
            raw = _call_ollama(prompt, temperature=0.2, max_tokens=260)
        else:
            raw = _call_genai(prompt, temperature=0.2, max_tokens=260)
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
    max_attempts = _max_attempts()

    for i in range(cantidad):
        for intento in range(max_attempts):
            texto = _generar_historia_genero(genero)
            ok_local, reason_local = _local_story_gate(texto, genero)
            if not ok_local:
                _log(f"[GPT] Historia rechazada por gate local: {reason_local} -> intentando reparar...")
                try:
                    texto = _reparar_historia_para_gate(texto, genero, reason_local)
                except Exception as e:
                    _log(f"[GPT] Reparación falló: {e}")
                ok_local, reason_local = _local_story_gate(texto, genero)
                if not ok_local:
                    _log(f"[GPT] Historia sigue fuera de gate: {reason_local} -> reintentando...")
                    continue

            h = _hash_text(texto)
            if h in hashes:
                _log(f"[GPT] Historia repetida detectada (hash ya usado), reintentando...")
                continue

            ok, reason = _validar_historia(texto)
            if not ok:
                _log(f"[GPT] Historia rechazada por validación IA: {reason} -> intentando reparar...")
                try:
                    texto_reparado = _reparar_historia_para_gate(texto, genero, reason or "validación IA")
                except Exception as e:
                    _log(f"[GPT] Reparación tras validación falló: {e}")
                    continue

                ok_local2, reason_local2 = _local_story_gate(texto_reparado, genero)
                if not ok_local2:
                    _log(f"[GPT] Historia reparada sigue fuera de gate: {reason_local2} -> reintentando...")
                    continue

                h2 = _hash_text(texto_reparado)
                if h2 in hashes:
                    _log("[GPT] Historia reparada repetida detectada, reintentando...")
                    continue

                ok2, reason2 = _validar_historia(texto_reparado)
                if not ok2:
                    _log(f"[GPT] Historia reparada rechazada nuevamente: {reason2} -> reintentando...")
                    continue
                texto = texto_reparado
                h = h2

            ruta = _guardar_historia(texto, genero, i)
            data.setdefault("items", []).append({
                "hash": h,
                "genero": genero,
                "path": ruta,
                "created": _now_ts(),
            })
            hashes.add(h)
            aceptadas.append((ruta, h))
            _log(f"[GPT] ✅ Historia {i + 1}/{cantidad} guardada en {ruta}")
            break
        else:
            _log(f"[GPT] ❌ No se pudo generar historia única para el índice {i} (intentos={max_attempts})")

    _save_db(data)
    return aceptadas
