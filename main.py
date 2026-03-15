import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from core import custom_video, image_downloader, local_tts, reddit_scraper, reddit_story_importer, story_generator, text_processor, tts, voice_clone
from core.cli_actions import CliActionContext, run_action_4, run_action_5, run_action_7, run_action_8, run_action_11
from utils.fs import crear_carpeta_proyecto
from core import topic_db
from core.video_renderer import (
    append_intro_to_video,
    audio_duration_seconds,
    combine_audios_with_silence,
    render_story_clip,
    render_video_base_con_audio,
    render_video_ffmpeg,
    select_video_base,
)
from utils.fs import guardar_historial
from utils import topic_file
from utils import topic_importer


from core.config import settings


def _configure_console_encoding() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Constantes locales (podrían moverse a config si se desea)
VOZ = (settings.custom_voice or os.environ.get("CUSTOM_VOICE") or "es-MX-JorgeNeural").strip()
VELOCIDAD = "-10%"

HISTORIAS_BASE = "historias"
HISTORIAS_GENEROS = {
    "1": "Drama y Relaciones",
    "2": "Terror y Paranormal",
    "3": "Venganza y Karens",
    "4": "Preguntas y Curiosidades",
    "5": "Reddit Virales",
}


def _pedir_entero(mensaje: str, *, minimo: int = 1, default: int = 1) -> int:
    try:
        val = int(input(mensaje).strip())
        if val < minimo:
            return default
        return val
    except Exception:
        return default


def _pedir_texto_multiline(etiqueta: str, *, max_empty_attempts: int = 3) -> str | None:
    print(etiqueta)
    print("[MAIN] Pega tu texto. Para terminar escribe 'fin' en una línea sola.")
    lineas: list[str] = []
    empty_attempts = 0
    while True:
        try:
            ln = input()
        except EOFError:
            if not lineas:
                return None
            break
        lns = ln.strip()
        if lns.lower() == "fin":
            break

        if not lns:
            if lineas:
                lineas.append("")
                continue
            empty_attempts += 1
            if empty_attempts >= max(1, int(max_empty_attempts)):
                print("[MAIN] ❌ Demasiados intentos vacíos. Abortando captura del prompt.")
                return None
            print(f"[MAIN] ⚠️ El prompt no puede estar vacío. Intento {empty_attempts}/{max_empty_attempts}.")
            continue
        empty_attempts = 0
        lineas.append(ln.rstrip())
    texto = "\n".join(lineas).rstrip()
    if not texto.strip():
        return None
    return texto


def _parse_indices_csv(raw: str, *, max_index: int) -> list[int]:
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[int] = []
    seen = set()
    for p in parts:
                        
        if "-" in p:
            a_raw, b_raw = [x.strip() for x in p.split("-", 1)]
            try:
                a = int(a_raw)
                b = int(b_raw)
            except Exception:
                continue
            if a <= 0 or b <= 0:
                continue
            start = min(a, b)
            end = max(a, b)
            for i in range(start, end + 1):
                if i < 1 or i > int(max_index):
                    continue
                if i in seen:
                    continue
                seen.add(i)
                out.append(i)
            continue

                           
        try:
            i = int(p)
        except Exception:
            continue
        if i < 1 or i > int(max_index):
            continue
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def _estimar_segundos(texto: str) -> float:
    palabras = len((texto or "").split())
    if palabras <= 0:
        return 0.0
                                                                                        
    wpm = 140.0
    estimado = (palabras / wpm) * 60.0                   
    estimado = estimado * 1.50 + 2.0                           
    return max(3.0, estimado)


def _crear_silencio(segundos: float, carpeta: str) -> str:
    import wave

    seg = max(0.0, segundos)
    if seg <= 0:
        return ""
    rate = 48000
    channels = 1
    sampwidth = 2
    frames = int(seg * rate)
    path = os.path.join(carpeta, f"silencio_{seg:.2f}s.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(b"\x00" * frames * channels * sampwidth)
    return path


def _filtrar_comentarios(comentarios, limite=200):
    filtrados = []
    for c in comentarios:
        if c.get("kind") != "t1":
            continue
        body = c.get("data", {}).get("body", "")
        cid = c.get("data", {}).get("id", "")
        if not body or "[deleted]" in body:
            continue
        if len(body) <= 80:
            continue
        if not reddit_scraper.es_historia_narrativa(body, min_chars=900):
            continue
        filtrados.append((body, cid))
        if len(filtrados) >= limite:
            break
    return filtrados


def _leer_historias_locales(carpeta_genero: str):
    os.makedirs(carpeta_genero, exist_ok=True)
    historias = []
    for nombre in sorted(os.listdir(carpeta_genero)):
        if nombre.startswith("0"):
            continue            
        if not nombre.lower().endswith(".txt"):
            continue
        ruta = os.path.join(carpeta_genero, nombre)
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                texto = f.read().strip()
            if texto:
                historias.append((texto, ruta))
        except Exception as e:
            print(f"[MAIN] ⚠️ No se pudo leer {ruta}: {e}")
    return historias


def _marcar_historias_usadas(rutas):
    for ruta in rutas:
        try:
            base = os.path.basename(ruta)
            dirname = os.path.dirname(ruta)
            if base.startswith("0"):
                continue
            nuevo = os.path.join(dirname, "0" + base)
                                                              
            while os.path.exists(nuevo):
                nuevo = os.path.join(dirname, "0" + os.path.basename(nuevo))
            os.rename(ruta, nuevo)
        except Exception as e:
            print(f"[MAIN] ⚠️ No se pudo marcar historia como usada ({ruta}): {e}")


def _seleccionar_genero() -> str:
    print("Elige género:")
    for clave, nombre in HISTORIAS_GENEROS.items():
        print(f"  {clave}. {nombre}")
    seleccion = input("Opción: ").strip()
    genero = HISTORIAS_GENEROS.get(seleccion, HISTORIAS_GENEROS["1"])
                                                                                                
    for nombre in HISTORIAS_GENEROS.values():
        os.makedirs(os.path.join(HISTORIAS_BASE, nombre), exist_ok=True)
    return genero


def _es_si(raw: str) -> bool:
    return (raw or "").strip().lower() in {"s", "si", "sí", "y", "yes", "1", "true"}


def _es_error_gpu_bloqueante(exc: Exception) -> bool:
    low = str(exc or "").strip().lower()
    if not low:
        return False
    claves = (
        "require_gpu",
        "encoder gpu",
        "h264_amf",
        "gpu detectada pero el encode de prueba falló",
        "ffmpeg no soporta el encoder gpu requerido",
    )
    return any(k in low for k in claves)


def _separar_texto_max_chars(texto: str, *, max_chars: int = 498) -> str:
    raw = (texto or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""

    max_chars = max(1, int(max_chars))
    salida: list[str] = []

    def _partir_texto_plano(s: str) -> list[str]:
        s = (s or "").strip()
        if not s:
            return []
        partes: list[str] = []
        actual = ""
        for palabra in s.split():
            if len(palabra) > max_chars:
                if actual:
                    partes.append(actual)
                    actual = ""
                for i in range(0, len(palabra), max_chars):
                    partes.append(palabra[i : i + max_chars])
                continue

            candidato = palabra if not actual else f"{actual} {palabra}"
            if len(candidato) <= max_chars:
                actual = candidato
            else:
                if actual:
                    partes.append(actual)
                actual = palabra

        if actual:
            partes.append(actual)
        return partes

    def _partir_parrafo(parrafo: str) -> list[str]:
        p = (parrafo or "").strip()
        if not p:
            return []

        oraciones = re.split(r'(?<=[.!?…])(?:["»”’)\]]+)?\s+', p)
        oraciones = [o.strip() for o in oraciones if o.strip()]
        if not oraciones:
            return _partir_texto_plano(p)

        partes: list[str] = []
        actual = ""

        for oracion in oraciones:
            if len(oracion) > max_chars:
                if actual:
                    partes.append(actual)
                    actual = ""
                partes.extend(_partir_texto_plano(oracion))
                continue

            candidato = oracion if not actual else f"{actual} {oracion}"
            if len(candidato) <= max_chars:
                actual = candidato
            else:
                if actual:
                    partes.append(actual)
                actual = oracion

        if actual:
            partes.append(actual)
        return partes

    parrafos = [p for p in re.split(r"\n\s*\n", raw) if p.strip()]
    for idx, parrafo in enumerate(parrafos):
        lineas_parrafo = [ln.strip() for ln in parrafo.split("\n") if ln.strip()]
        for linea in lineas_parrafo:
            salida.extend(_partir_parrafo(linea))
        if idx < len(parrafos) - 1 and salida and salida[-1] != "":
            salida.append("")

    return "\n".join(salida)


def _accion_generar_voz_local() -> None:
    print("\n[MAIN] Generar voz local (texto -> audio)")

    modo = input("Fuente (1=pegar texto, 2=archivo .txt) [1]: ").strip()
    texto = ""
    if modo == "2":
        ruta = input("Ruta del archivo .txt: ").strip().strip('"')
        if not ruta or not os.path.exists(ruta):
            print("[MAIN] ❌ Ruta inválida")
            raise SystemExit(0)
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                texto = f.read()
        except Exception as e:
            print(f"[MAIN] ❌ No se pudo leer archivo: {e}")
            raise SystemExit(0)
    else:
        texto_in = _pedir_texto_multiline("[MAIN] Pega el texto para voz local:")
        if texto_in is None:
            print("[MAIN] ❌ No se recibió texto")
            raise SystemExit(0)
        texto = texto_in

    if not (texto or "").strip():
        print("[MAIN] ❌ El texto está vacío")
        raise SystemExit(0)

    out_path = os.path.join("output", f"local_tts_{int(time.time())}.wav")

    blocked_voices = set(getattr(local_tts, "KOKORO_BLOCKED_VOICES", set()) or set())
    preferred_voices = [
        v
        for v in (getattr(local_tts, "KOKORO_PREFERRED_VOICES", []) or [])
        if v and v not in blocked_voices
    ]
    default_voice = preferred_voices[0] if preferred_voices else "em_santa"

    voice_options: list[tuple[str, str]] = [
        ("santa", "em_santa"),
        ("titan", "pm_santa"),
        ("bella", "af_bella"),
        ("heart", "af_heart"),
        ("emma", "bf_emma"),
        ("nicola", "im_nicola"),
    ]
    voice_options = [(alias, voice) for alias, voice in voice_options if voice in preferred_voices]
    if not voice_options:
        voice_options = [("santa", default_voice)]

    print("[MAIN] Elige voz:")
    for idx, (alias, _) in enumerate(voice_options, start=1):
        print(f"  {idx}. {alias}")

    default_index = 1
    for idx, (_, voice) in enumerate(voice_options, start=1):
        if voice == default_voice:
            default_index = idx
            break

    voice_raw = input(f"Voz [{default_index}]: ").strip().lower()
    selected_voice = default_voice
    if voice_raw:
        if voice_raw.isdigit():
            sel_idx = int(voice_raw)
            if 1 <= sel_idx <= len(voice_options):
                selected_voice = voice_options[sel_idx - 1][1]
        else:
            by_alias = {alias.lower(): voice for alias, voice in voice_options}
            if voice_raw in by_alias:
                selected_voice = by_alias[voice_raw]
            elif voice_raw in preferred_voices and voice_raw not in blocked_voices:
                selected_voice = voice_raw

    if selected_voice in blocked_voices:
        print(f"[MAIN] ⚠️ Voz bloqueada ({selected_voice}). Usando '{default_voice}'.")
        selected_voice = default_voice

    kwargs = {
        "text": texto,
        "output_path": out_path,
        "engine": "kokoro",
        "kokoro_voice": selected_voice,
        "timeout_s": 600,
    }

    try:
        result = local_tts.synthesize_local_voice(**kwargs)
        print(f"[MAIN] ✅ Audio generado: {result.output_path} (engine={result.engine})")
    except Exception as e:
        print(f"[MAIN] ❌ Error generando voz local: {e}")

    raise SystemExit(0)


def _accion_clonar_voz_local() -> None:
    def _buscar_referencias_voz(*, max_items: int = 20) -> list[str]:
        exts = {".wav", ".mp3", ".m4a"}
        roots = [".", "storage", "output", "historias"]
        skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "site-packages", "dist-packages"}
        cwd = os.path.abspath(os.getcwd())
        keyword_tokens = {
            "voz",
            "voice",
            "clone",
            "clon",
            "ref",
            "referencia",
            "sample",
            "speaker",
            "capcut",
        }

        def _should_skip_dir(name: str) -> bool:
            low = (name or "").strip().lower()
            if not low:
                return True
            if low in skip_dirs:
                return True
            # Exclude virtualenv folders with custom names like .venv-voiceclone.
            if low.startswith(".venv") or low.endswith(".venv"):
                return True
            return False

        def _is_render_audio(rel_path: str, stem: str) -> bool:
            if rel_path.startswith("output/custom_"):
                if stem == "audio_con_silencios":
                    return True
                if re.fullmatch(r"audio_\d+", stem):
                    return True
            if re.fullmatch(r"audio_\d+", stem):
                return True
            if "con_silencios" in stem:
                return True
            return False

        ranked: list[tuple[int, float, str]] = []

        for root in roots:
            if not os.path.isdir(root):
                continue
            for current_root, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if not _should_skip_dir(d)]
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in exts:
                        continue
                    p = os.path.abspath(os.path.join(current_root, fn))
                    rel = os.path.relpath(p, cwd).replace("\\", "/").lower()
                    stem = os.path.splitext(fn)[0].strip().lower()
                    if _is_render_audio(rel, stem):
                        continue

                    score = 0
                    if any(tok in stem for tok in keyword_tokens):
                        score += 4
                    if any(tok in rel for tok in keyword_tokens):
                        score += 2
                    if rel.count("/") <= 1:
                        score += 1
                    if rel.startswith("output/"):
                        score -= 1
                    if score <= 0:
                        continue

                    try:
                        mtime = os.path.getmtime(p)
                    except Exception:
                        mtime = 0.0
                    ranked.append((score, mtime, p))

        ranked.sort(key=lambda it: (it[0], it[1]), reverse=True)
        unique: list[str] = []
        seen: set[str] = set()
        for _, _, p in ranked:
            if p in seen:
                continue
            seen.add(p)
            unique.append(p)
            if len(unique) >= max(1, int(max_items)):
                break
        return unique

    print("\n[MAIN] Texto a voz clonada (MODO CALIDAD MAXIMA)")
    print("[MAIN] Paso 1/3: elige el audio de voz a clonar")

    referencias = _buscar_referencias_voz(max_items=20)
    ref_path = ""

    if referencias:
        print("[MAIN] Voces de referencia detectadas:")
        for i, p in enumerate(referencias, start=1):
            nombre = os.path.basename(p)
            print(f"  {i}. {nombre}  ({p})")
        raw_ref = input("Voz a clonar (número o ruta) [1]: ").strip().strip('"')
        if not raw_ref:
            ref_path = referencias[0]
        elif raw_ref.isdigit() and 1 <= int(raw_ref) <= len(referencias):
            ref_path = referencias[int(raw_ref) - 1]
        else:
            ref_path = raw_ref
    else:
        print("[MAIN] No encontré audios de voz en el proyecto. Puedes pegar la ruta manualmente.")
        ref_path = input("Ruta del audio de voz (.wav/.mp3/.m4a): ").strip().strip('"')

    if not ref_path or not os.path.exists(ref_path):
        print("[MAIN] ❌ Ruta de referencia inválida")
        raise SystemExit(0)

    ref_ext = os.path.splitext(ref_path)[1].lower()
    if ref_ext not in {".wav", ".mp3", ".m4a"}:
        print("[MAIN] ❌ La referencia debe ser .wav, .mp3 o .m4a")
        raise SystemExit(0)

    print("[MAIN] Paso 2/3: escribe el texto que quieres escuchar con esa voz")
    modo = input("Fuente de texto (1=pegar texto, 2=archivo .txt) [1]: ").strip()
    texto = ""
    if modo == "2":
        ruta = input("Ruta del archivo .txt: ").strip().strip('"')
        if not ruta or not os.path.exists(ruta):
            print("[MAIN] ❌ Ruta inválida")
            raise SystemExit(0)
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                texto = f.read()
        except Exception as e:
            print(f"[MAIN] ❌ No se pudo leer archivo: {e}")
            raise SystemExit(0)
    else:
        texto_in = _pedir_texto_multiline("[MAIN] Pega el texto para texto-a-voz clonada:")
        if texto_in is None:
            print("[MAIN] ❌ No se recibió texto")
            raise SystemExit(0)
        texto = texto_in

    if not (texto or "").strip():
        print("[MAIN] ❌ El texto está vacío")
        raise SystemExit(0)

    out_path = os.path.join("output", f"voice_clone_hq_{int(time.time())}.wav")

    print("[MAIN] Paso 3/3: opciones de generación (calidad máxima)")
    idioma = (input("Idioma para TTS [es]: ").strip() or "es").lower()
    force_cpu = True
    print("[MAIN] Perfil aplicado: WAV + CPU + limpieza de silencios + referencia 30s + mejor tramo vocal.")

    fallback_py = os.path.join(".venv-voiceclone", "Scripts", "python.exe")
    try:
        if os.path.exists(fallback_py):
            tmp_text_path = ""
            try:
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as tf:
                    tf.write(texto)
                    tmp_text_path = tf.name

                cmd = [
                    os.path.abspath(fallback_py),
                    "-m",
                    "core.voice_clone",
                    "--ref",
                    os.path.abspath(ref_path),
                    "--text-file",
                    tmp_text_path,
                    "--out",
                    os.path.abspath(out_path),
                    "--lang",
                    idioma,
                    "--timeout",
                    "1800",
                    "--ref-max-sec",
                    "30",
                    "--ref-trim-silence",
                    "--ref-pick-best",
                    "--cpu",
                ]

                run = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1900,
                )
                out_msg = (run.stdout or "").strip()
                err_msg = (run.stderr or "").strip()
                if run.returncode == 0:
                    if out_msg:
                        print(out_msg)
                    print("[MAIN] ✅ Texto-a-voz clonada completado usando .venv-voiceclone (MODO CALIDAD MAXIMA)")
                    raise SystemExit(0)

                if out_msg:
                    print(out_msg)
                if err_msg:
                    print(f"[MAIN] ❌ STDERR: {err_msg[:500]}")
                print("[MAIN] ⚠️ Falló .venv-voiceclone. Intentando fallback con entorno actual...")
            finally:
                if tmp_text_path and os.path.exists(tmp_text_path):
                    try:
                        os.remove(tmp_text_path)
                    except Exception:
                        pass

        result = voice_clone.clone_voice_to_audio(
            text=texto,
            reference_audio_path=ref_path,
            output_path=out_path,
            language=idioma,
            prefer_gpu=False,
            timeout_s=1800,
            reference_max_seconds=30,
            reference_trim_silence=True,
            reference_pick_best_segment=True,
        )
        print(
            f"[MAIN] ✅ Texto-a-voz clonada completado: {result.output_path} "
            f"(backend={result.backend}, model={result.model_name})"
        )
    except Exception as e:
        print(f"[MAIN] ❌ Error en clonación de voz (modo calidad máxima): {e}")

    raise SystemExit(0)


def _preguntar_tts_render(*, voz_default: str, velocidad_default: str) -> tuple[str, str]:
    render_velocidad = velocidad_default

    default_kokoro_voice = "em_santa"
    try:
        default_kokoro_voice = str(local_tts._default_kokoro_voice()).strip() or "em_santa"
    except Exception:
        default_kokoro_voice = "em_santa"

    kokoro_voices: list[str] = []
    try:
        if getattr(local_tts, "Kokoro", None) is not None:
            model_path, voices_path = local_tts._ensure_kokoro_assets(timeout_s=300)
            kk = local_tts.Kokoro(model_path, voices_path)
            blocked = set(getattr(local_tts, "KOKORO_BLOCKED_VOICES", set()) or set())
            raw = [str(v).strip() for v in (kk.get_voices() or []) if str(v).strip()]
            filtered = [v for v in raw if v not in blocked]
            preferred = list(getattr(local_tts, "KOKORO_PREFERRED_VOICES", []) or [])
            pref_set = set(preferred)
            kokoro_voices = sorted(filtered, key=lambda v: (v not in pref_set, preferred.index(v) if v in pref_set else 9999, v))
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudieron listar voces de Kokoro: {e}")
        kokoro_voices = []

    if kokoro_voices:
        print("[MAIN] Voces Kokoro disponibles:")
        for i, v in enumerate(kokoro_voices, start=1):
            print(f"  {i}. {v}")

    raw = input(
        f"Voz Kokoro para render (número o nombre) [{default_kokoro_voice}]: "
    ).strip()

    if not raw:
        render_voz = default_kokoro_voice
    elif raw.isdigit() and kokoro_voices:
        idx = int(raw)
        if 1 <= idx <= len(kokoro_voices):
            render_voz = kokoro_voices[idx - 1]
        else:
            render_voz = default_kokoro_voice
    else:
        render_voz = raw
        low = raw.lower()
        if kokoro_voices:
            for v_name in kokoro_voices:
                if low in v_name.lower():
                    render_voz = v_name
                    break

    print(f"[MAIN] TTS elegido: voz='{render_voz}' | velocidad='{render_velocidad}'")
    return render_voz, render_velocidad


class _WindowsKeepAwake:
    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._ok = False
        self._kernel32 = None

    def __enter__(self):
        if (not self._enabled) or os.name != "nt":
            return self
        try:
            import ctypes

            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED

            self._kernel32 = ctypes.windll.kernel32
            self._kernel32.SetThreadExecutionState(flags)
            self._ok = True
        except Exception:
            self._ok = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if (not self._enabled) or os.name != "nt":
            return False
        if not self._ok:
            return False
        try:
            import ctypes

            ES_CONTINUOUS = 0x80000000
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            pass
        return False


def _start_cancel_suspend_listener(flag: dict) -> None:
    return


def _maybe_suspend_with_grace(cancel_flag: dict, *, grace_seconds: int = 60) -> None:
    return


def _planes_pendientes() -> list[str]:
    pendientes = []
    base = os.path.abspath("output")
    if not os.path.isdir(base):
        return pendientes
    for entry in os.scandir(base):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("0"):
            continue
        plan_file = os.path.join(entry.path, "plan.json")
        if os.path.exists(plan_file):
            pendientes.append(entry.path)
                                                                     
    pendientes.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pendientes


def _custom_plans_pendientes() -> list[str]:
    pendientes: list[str] = []
    base = os.path.abspath("output")
    if not os.path.isdir(base):
        return pendientes
    for entry in os.scandir(base):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("0"):
            continue
        plan_file = os.path.join(entry.path, "custom_plan.json")
        if not os.path.exists(plan_file):
            continue


        done = False
        finalized = False
        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                plan = json.load(f) or {}
            done = bool(plan.get("render_done"))
            finalized = bool(plan.get("topic_finalized"))
        except Exception:
            done = False
            finalized = False

        if not done:
            try:
                vf = os.path.join(entry.path, "Video_Final.mp4")
                if os.path.exists(vf) and os.path.getsize(vf) > 250_000:
                    done = True
            except Exception:
                pass

        
        # Pendiente si:
        # - falta render, o
        # - ya renderizó pero falta finalizar (DB/archivo)
        if (not done) or (done and not finalized):
            pendientes.append(entry.path)

    pendientes.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pendientes


def _custom_plans_todos() -> list[str]:
    planes: list[str] = []
    base = os.path.abspath("output")
    if not os.path.isdir(base):
        return planes
    for entry in os.scandir(base):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("0"):
            continue
        plan_file = os.path.join(entry.path, "custom_plan.json")
        if os.path.exists(plan_file):
            planes.append(entry.path)

    planes.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return planes


def _custom_plan_flags(ruta: str) -> dict:
    plan_path = os.path.join(ruta, "custom_plan.json")
    rendered = False
    finalized = False
    try:
        if os.path.exists(plan_path):
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f) or {}
            rendered = bool(plan.get("render_done"))
            finalized = bool(plan.get("topic_finalized"))
    except Exception:
        pass
    if not rendered:
        try:
            vf = os.path.join(ruta, "Video_Final.mp4")
            if os.path.exists(vf) and os.path.getsize(vf) > 250_000:
                rendered = True
        except Exception:
            pass
    return {"rendered": rendered, "finalized": finalized, "plan_path": plan_path}


def _finalizar_tema_custom_renderizado(ruta: str) -> None:
    try:
        plan_path = os.path.join(ruta, "custom_plan.json")
        if not os.path.exists(plan_path):
            return
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f) or {}
        brief = str(plan.get("brief") or "").strip()
        topic_source = str(plan.get("topic_source") or "").strip().lower()
        topic_file_path = str(plan.get("topic_file") or "").strip()

        if brief:
            topic_db.register_topic_if_new(
                brief,
                kind="custom",
                plan_dir=ruta,
                threshold=0.98,
            )

        try:
            topic_db.delete_by_plan_dir(ruta, kind="custom_pending")
        except Exception:
            pass

        if topic_source == "file" and topic_file_path and brief:
            marcado = topic_file.mark_topic_used(brief, topic_file_path)
            if marcado:
                print("[MAIN] ✅ Tema marcado como usado en archivo.")

        try:
            import time as _time

            plan["topic_finalized"] = True
            plan["topic_finalized_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudo finalizar tema (DB/archivo): {e}")


def _marcar_plan_completado(path: str):
    if not path:
        return
    base = os.path.abspath(path)
    parent = os.path.dirname(base)
    name = os.path.basename(base)
    if name.startswith("0"):
        return
    nuevo = os.path.join(parent, "0" + name)
    while os.path.exists(nuevo):
        nuevo = os.path.join(parent, "0" + os.path.basename(nuevo))
    shutil.move(base, nuevo)
    plan_json = os.path.join(nuevo, "plan.json")
    if os.path.exists(plan_json):
        try:
            os.remove(plan_json)
        except Exception:
            print(f"[MAIN] ⚠️ No se pudo borrar plan.json en {nuevo}")
    print(f"[MAIN] Plan marcado como completado: {nuevo}")


def _generar_timeline(prompts: list[str], dur_est: float) -> list[dict]:
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


def _segmentar_historia_en_prompts(historia: str, prompts: list[str]) -> list[str]:
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
    segmentos = []
    for i in range(0, len(oraciones), chunk):
        segmentos.append(" ".join(oraciones[i : i + chunk]).strip())

                                                                             
    segmentos = [s for s in segmentos if s]
    if len(segmentos) > objetivo:
        segmentos = segmentos[:objetivo]

    return segmentos


def _generar_video(usar_video_base: bool, indice: int, total: int, *, usar_historias_locales: bool, carpeta_genero: str | None, fase: int, plan_path: str | None) -> bool:
    print(f"[MAIN] ===== Video {indice}/{total} (fase {fase}) =====")

    carpeta = crear_carpeta_proyecto(prefix="plan" if fase == 1 else None)

    video_base_path = None
    video_base_dur = 0.0
    plan_data = None

                         
    if fase == 2:
        if not plan_path:
            print("[MAIN] Debes indicar la ruta de plan (carpeta con plan.json)")
            return False
        plan_file = os.path.join(plan_path, "plan.json")
        if not os.path.exists(plan_file):
            print(f"[MAIN] No se encontró plan.json en {plan_path}")
            return False
        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
        except Exception as e:
            print(f"[MAIN] No se pudo leer plan: {e}")
            return False
        usar_video_base = bool(plan_data.get("usar_video_base", usar_video_base))
        texto_elegido = plan_data.get("texto_en", "")
        texto_es_plan = plan_data.get("texto_es", "")
        ids_seleccionados = plan_data.get("ids", []) or [""]
        prompts_plan = plan_data.get("prompts", [])
        timeline_plan = plan_data.get("timeline", [])
        if not texto_es_plan:
            print("[MAIN] El plan no tiene texto_es")
            return False
        textos_en = [texto_elegido or texto_es_plan]
        textos_es = [texto_es_plan]
        prompts = prompts_plan
        timeline = timeline_plan
        carpeta = plan_path                               
    else:
                                      
        if usar_video_base:
            video_base_path, video_base_dur = select_video_base(None)
            if not video_base_path:
                print("[MAIN] No se encontró video base utilizable")
                return False
            if video_base_dur <= 0:
                print("[MAIN] No se pudo detectar duración del video base. Abortando.")
                return False
            print(f"[MAIN] Video base seleccionado: {video_base_path} (~{video_base_dur:.1f}s)")

        if usar_historias_locales:
            if not carpeta_genero:
                print("[MAIN] No se especificó carpeta de historias")
                return False
            comentarios_filtrados = _leer_historias_locales(carpeta_genero)
            if not comentarios_filtrados:
                print(f"[MAIN] No se encontraron historias en {carpeta_genero}")
                return False
        else:
            post = reddit_scraper.obtener_post()
            if not post:
                print("[MAIN] No se pudo obtener post")
                return False

            comentarios = reddit_scraper.obtener_comentarios(post["permalink"])
            comentarios_filtrados = _filtrar_comentarios(comentarios, limite=200)

        if not comentarios_filtrados:
            print("[MAIN] No se encontraron historias")
            return False

                                                                    
        comentarios_filtrados = sorted(comentarios_filtrados, key=lambda t: len(t[0]), reverse=True)
        texto_elegido, cid_elegido = comentarios_filtrados[0]
        comentarios_filtrados = [(texto_elegido, cid_elegido)]

        textos_en = [texto_elegido]
        ids_seleccionados = [cid_elegido]

        if usar_historias_locales:
            _marcar_historias_usadas(ids_seleccionados)

    print(f"[MAIN] {len(textos_en)} textos obtenidos")

                                                                     
    if fase != 2:
        textos_es = text_processor.traducir_lista(textos_en)

    print("[DEBUG] Primer texto que irá al TTS:")
    print(textos_es[0][:200])

    if fase == 2:
        historia_es = textos_es[0]
        prompts = prompts or []
        segments_plan = plan_data.get("segments", []) if plan_data else []
        segments = segments_plan or _segmentar_historia_en_prompts(historia_es, prompts)
        if len(segments) < len(prompts):
            prompts = prompts[: len(segments)]
        timeline = plan_data.get("timeline", []) if plan_data else []
    else:
                                                    
        historia_es = textos_es[0]
        prompts = image_downloader.generar_prompts_historia(historia_es)
        if not prompts:
            print("[MAIN] No se generaron prompts de imagen (IA)")
            return False

        segments = _segmentar_historia_en_prompts(historia_es, prompts)
        if not segments:
            print("[MAIN] No se pudo segmentar la historia para sincronizar audio e imágenes")
            return False
        if len(segments) < len(prompts):
            prompts = prompts[: len(segments)]

    if not segments:
        print("[MAIN] No hay segmentos listos para sincronizar audio e imágenes")
        return False

    dur_est = _estimar_segundos(historia_es)
    timeline = _generar_timeline(prompts, dur_est)

    if fase == 1:
        plan = {
            "texto_en": textos_en[0],
            "texto_es": textos_es[0],
            "prompts": prompts,
            "timeline": timeline,
            "dur_est": dur_est,
            "ids": ids_seleccionados,
            "segments": segments,
            "usar_video_base": usar_video_base,
        }
        plan_file = os.path.join(carpeta, "plan.json")
        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"[MAIN] ✅ Plan guardado en {plan_file}")
        return True

                                                                           
    if fase == 2 and timeline:
        dur_est = timeline[-1].get("end", _estimar_segundos(textos_es[0]))
    else:
        dur_est = _estimar_segundos(textos_es[0])

    imagenes = image_downloader.descargar_imagenes_desde_prompts(carpeta, prompts, dur_audio=dur_est)
    if not imagenes:
        print("[MAIN] No se descargaron imágenes")
        return False

                                                    
    audios = tts.generar_audios(segments, carpeta, voz=VOZ, velocidad=VELOCIDAD)
    if not audios:
        print("[MAIN] No se generaron audios")
        return False

    if len(audios) != len(imagenes) or len(audios) != len(prompts):
        min_items = min(len(audios), len(imagenes), len(prompts))
        print(f"[MAIN] Ajustando a {min_items} segmentos por desbalance entre audios/prompts/imagenes")
        audios = audios[:min_items]
        imagenes = imagenes[:min_items]
        prompts = prompts[:min_items]
        segments = segments[:min_items]
    if not audios or not imagenes:
        print("[MAIN] No hay suficientes audios o imágenes para renderizar")
        return False

    duraciones_imgs = [max(0.6, audio_duration_seconds(a)) for a in audios]
    audio_final = combine_audios_with_silence(audios, carpeta, gap_seconds=0, min_seconds=None, max_seconds=None)

    if usar_video_base:
        try:
            video_final = render_video_base_con_audio(video_base_path, audio_final, carpeta, videos_dir=None)
        except Exception as e:
            if _es_error_gpu_bloqueante(e):
                raise
            print(f"[MAIN] Error renderizando con video base: {e}")
            return False
    else:
                                                                       
        timeline = []
        pos = 0.0
        for prompt, d in zip(prompts, duraciones_imgs):
            start = pos
            end = pos + d
            timeline.append({"prompt": prompt, "start": round(start, 2), "end": round(end, 2)})
            pos = end

        video_final = render_video_ffmpeg(imagenes, audio_final, carpeta, tiempo_img=None, durations=duraciones_imgs)

    append_intro_to_video(video_final, title_text=textos_es[0])

    try:
        claves = []
        for cid, texto in zip(ids_seleccionados, textos_es):
            if cid:
                claves.append(f"reddit_comment_used:{cid}")
            else:
                h = hashlib.sha1(texto.encode("utf-8")).hexdigest()
                claves.append(f"reddit_comment_hash:{h}")
        if claves:
            guardar_historial(claves)
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudo guardar historial de historias usadas: {e}")

    print("[MAIN] ✅ Video generado")

                                                    
    if fase == 2 and plan_path:
        try:
            _marcar_plan_completado(plan_path)
        except Exception as e:
            print(f"[MAIN] ⚠️ No se pudo marcar plan completado: {e}")
    return True


def _bootstrap_topics_file() -> None:
    try:
        topic_file.ensure_topics_file(topic_file.TOPICS_FILE_DEFAULT)
        disponibles = topic_file.load_topics_available_with_flags(topic_file.TOPICS_FILE_DEFAULT)
        print(
            f"[MAIN] Temas en archivo (sin prefijo 0): {len(disponibles)} -> {topic_file.TOPICS_FILE_DEFAULT}"
        )
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudo leer temas_custom.txt: {e}")


def _print_runtime_models_info() -> None:
    text_model = getattr(custom_video, "OLLAMA_TEXT_MODEL", "") or "(desconocido)"
    vision_model = (os.environ.get("VISION_MODEL") or "minicpm-v:latest").strip() or "minicpm-v:latest"
    ollama_url = (os.environ.get("OLLAMA_URL") or "http://localhost:11434/api/generate").strip()
    print(f"[MAIN] Modelo texto: {text_model} (OLLAMA_TEXT_MODEL/OLLAMA_MODEL o default)")
    print(f"[MAIN] Modelo visión: {vision_model} (env VISION_MODEL)")
    print(f"[MAIN] Ollama URL: {ollama_url} (env OLLAMA_URL)")


def _accion_separar_texto_498() -> None:
    texto = _pedir_texto_multiline("\n[MAIN] Separar texto en líneas de máximo 498 caracteres:")
    if texto is None:
        print("[MAIN] ❌ No se recibió texto.")
        raise SystemExit(0)

    resultado = _separar_texto_max_chars(texto, max_chars=498)
    lineas = resultado.splitlines() if resultado else []
    max_len = max((len(ln) for ln in lineas), default=0)
    print("\n[MAIN] ✅ Texto separado (máximo 498 por línea):\n")
    print(resultado)
    print(f"\n[MAIN] Verificación -> líneas: {len(lineas)} | largo máximo: {max_len}")
    raise SystemExit(0)


def _leer_blob_import_temas() -> str:
    modo = input("Fuente (1=pegar texto, 2=archivo .txt) [1]: ").strip()
    if modo == "2":
        ruta = input("Ruta del archivo .txt: ").strip().strip('"')
        if not ruta or not os.path.exists(ruta):
            print("[MAIN] Ruta inválida")
            raise SystemExit(0)
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[MAIN] No se pudo leer archivo: {e}")
            raise SystemExit(0)

    print("Pega el texto (multi-línea). Escribe END en una línea sola para terminar:")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _accion_importar_temas() -> None:
    print("\n[MAIN] Importar temas a storage/temas_custom.txt")
    blob = _leer_blob_import_temas()

    prompts = topic_importer.parse_prompts_from_blob(blob)
    if not prompts:
        print("[MAIN] No se encontraron prompts.")
        raise SystemExit(0)

    aceptados, descartados = topic_importer.dedupe_prompts(prompts, topics_path=topic_file.TOPICS_FILE_DEFAULT)

    if aceptados:
        topic_file.append_topics([(p, False) for p in aceptados], path=topic_file.TOPICS_FILE_DEFAULT)

    print(f"\n[MAIN] Agregados: {len(aceptados)}")
    for i, p in enumerate(aceptados, start=1):
        print(f"  + {i}. {p[:120]}")

    print(f"\n[MAIN] Descartados: {len(descartados)}")
    for i, (p, motivo) in enumerate(descartados, start=1):
        print(f"  - {i}. ({motivo}) {p[:120]}")

    if descartados:
        sel = input(
            "\n¿Quieres agregar alguno de los descartados de todos modos? (ej: 1,3,5 | Enter=skip): "
        ).strip()
        if sel:
            idxs = _parse_indices_csv(sel, max_index=len(descartados))
            if idxs:
                force = [(descartados[i - 1][0], True) for i in idxs]
                topic_file.append_topics(force, path=topic_file.TOPICS_FILE_DEFAULT)
                print(f"[MAIN] Forzados agregados: {len(force)} (con prefijo '!')")

    print(f"\n[MAIN] Listo. Archivo: {topic_file.TOPICS_FILE_DEFAULT}")
    raise SystemExit(0)


def _accion_importar_historias() -> None:
    print("\n[MAIN] Importar historias virales de Reddit (filtro IA + no repetidas)")
    total = _pedir_entero("¿Cuántas historias importar?: ", minimo=1, default=3)
    try:
        res = reddit_story_importer.importar_historias_reddit(total=total)
    except Exception as e:
        print(f"[MAIN] ❌ Error importando historias: {e}")
        raise SystemExit(1)

    print(
        "[MAIN] Resultado importación -> "
        f"solicitadas: {res.get('requested', 0)} | "
        f"evaluadas por IA: {res.get('evaluated', 0)} | "
        f"importadas: {res.get('imported', 0)}"
    )

    cache = res.get("cache") or {}
    if isinstance(cache, dict):
        print(
            "[MAIN] Cache Reddit -> "
            f"posts vistos: {cache.get('seen_posts', 0)} | "
            f"comentarios vistos: {cache.get('seen_comments', 0)}"
        )

    for i, ruta in enumerate(res.get("saved", []) or [], start=1):
        print(f"  {i}. {ruta}")

    if res.get("reason"):
        print(f"[MAIN] ℹ️ {res.get('reason')}")

    print("[MAIN] Las historias quedaron en: historias/Reddit Virales")
    raise SystemExit(0)


def _accion_generar_textos() -> None:
    total_textos = _pedir_entero("¿Cuántas historias generar?: ", minimo=1, default=1)
    genero = _seleccionar_genero()
    try:
        resultados = story_generator.generar_historias(genero, total_textos)
        print(f"[MAIN] Historias generadas: {len(resultados)}/{total_textos}")
    except Exception as e:
        print(f"[MAIN] ⚠️ Error generando historias: {e}")
    raise SystemExit(0)


def _accion_imagen_prueba() -> None:
    prompt = input("Prompt para la imagen de prueba: ").strip()
    carpeta_prueba = os.path.join("imagenes_prueba")
    os.makedirs(carpeta_prueba, exist_ok=True)
    try:
        ruta = image_downloader.generar_imagen_prueba(prompt, carpeta_prueba)
        if ruta:
            print(f"[MAIN] ✅ Imagen guardada en: {ruta}")
        else:
            print("[MAIN] ❌ No se pudo generar la imagen de prueba")
    except Exception as e:
        print(f"[MAIN] ⚠️ Error generando imagen de prueba: {e}")
    raise SystemExit(0)


def _build_cli_action_context() -> CliActionContext:
    return CliActionContext(
        voz=VOZ,
        velocidad=VELOCIDAD,
        pedir_entero=_pedir_entero,
        pedir_texto_multiline=_pedir_texto_multiline,
        parse_indices_csv=_parse_indices_csv,
        es_si=_es_si,
        preguntar_tts_render=_preguntar_tts_render,
        windows_keep_awake_cls=_WindowsKeepAwake,
        es_error_gpu_bloqueante=_es_error_gpu_bloqueante,
        custom_plans_pendientes=_custom_plans_pendientes,
        custom_plans_todos=_custom_plans_todos,
        custom_plan_flags=_custom_plan_flags,
        finalizar_tema_custom_renderizado=_finalizar_tema_custom_renderizado,
    )


if __name__ == "__main__":
    _configure_console_encoding()
    print("[MAIN] Iniciando proceso")

    _bootstrap_topics_file()
    _print_runtime_models_info()

    accion = input(
        "¿Qué deseas hacer? (1 = Videos, 2 = Textos, 3 = Imagen de prueba, 4 = Video personalizado [en 4.3: usa importadas de Reddit primero], 5 = Renderizar personalizado, 6 = Importar temas, 7 = Reanudar último personalizado, 8 = Reanudar TODOS personalizados pendientes, 9 = Separar texto en líneas de 498, 10 = Solo generar voz local, 11 = Mejorar planes personalizados, 12 = Importar historias, 13 = Texto a voz clonada desde audio): "
    ).strip()

    if accion == "10":
        _accion_generar_voz_local()

    if accion == "13":
        _accion_clonar_voz_local()

    if accion == "9":
        _accion_separar_texto_498()

    if accion == "6":
        _accion_importar_temas()

    if accion == "2":
        _accion_generar_textos()

    if accion == "3":
        _accion_imagen_prueba()

    ctx = _build_cli_action_context()

    if accion == "4":
        run_action_4(ctx)

    if accion == "5":
        run_action_5(ctx)

    if accion == "7":
        run_action_7(ctx)

    if accion == "8":
        run_action_8(ctx)

    if accion == "11":
        run_action_11(ctx)

    if accion == "12":
        _accion_importar_historias()

                    
    fase = _pedir_entero("Selecciona fase (1=plan, 2=render pendientes): ", minimo=1, default=2)

    plan_paths = []
    if fase == 2:
        pend = _planes_pendientes()
        if not pend:
            print("[MAIN] No hay planes pendientes (carpetas sin 0 con plan.json)")
            raise SystemExit(0)
        print(f"[MAIN] Planes pendientes detectados: {len(pend)}")
        for p in pend:
            print(f"   - {p}")
        plan_paths = pend
        usar_video_base = False                     
        usar_historias_locales = False
        carpeta_genero = None
        total_videos = len(plan_paths)
    else:
        total_videos = _pedir_entero("¿Cuántos videos (planes) generar?: ", minimo=1, default=1)
        modo = input("Selecciona modo (1 = IA/imagenes, 2 = Video base): ").strip()
        usar_video_base = modo == "2"

        origen = input("Origen de historias (1 = Reddit automático, 2 = Carpeta 'historias'): ").strip()
        usar_historias_locales = origen == "2"
        carpeta_genero = None
        plan_paths = [None] * total_videos

        if usar_historias_locales:
            genero = _seleccionar_genero()
            carpeta_genero = os.path.join(HISTORIAS_BASE, genero)
            print(f"[MAIN] Usando historias locales desde: {carpeta_genero}")

    exitos = 0
    for i in range(total_videos):
        plan_path = plan_paths[i] if i < len(plan_paths) else None
        try:
            if _generar_video(
                usar_video_base,
                i + 1,
                total_videos,
                usar_historias_locales=usar_historias_locales,
                carpeta_genero=carpeta_genero,
                fase=fase,
                plan_path=plan_path,
            ):
                exitos += 1
        except Exception as e:
            if _es_error_gpu_bloqueante(e):
                print(f"[MAIN] ❌ Render detenido por requisito GPU: {e}")
                raise SystemExit(1)
            print(f"[MAIN] ⚠️ Error inesperado en video {i+1}/{total_videos}: {e}")
    print(f"[MAIN] Finalizado: {exitos}/{total_videos} operaciones completadas con éxito")
