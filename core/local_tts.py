import argparse
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

try:
    from kokoro_onnx import Kokoro
except Exception:
    Kokoro = None


class LocalTTSError(RuntimeError):
    pass


ROOT_DIR = Path(__file__).resolve().parents[1]
KOKORO_DIR = ROOT_DIR / "storage" / "kokoro"
KOKORO_MODEL_PATH = KOKORO_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = KOKORO_DIR / "voices-v1.0.bin"
KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)

KOKORO_PREFERRED_VOICES = ["em_santa", "af_bella", "af_heart", "bf_emma", "im_nicola", "pm_santa"]
KOKORO_BLOCKED_VOICES = {"ef_dora", "em_alex", "pm_alex", "pf_dora", "if_sara"}


@dataclass(slots=True)
class LocalTTSResult:
    output_path: str
    engine: str
    sample_rate: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def _find_ffmpeg() -> str | None:
    candidates: list[str] = []
    if imageio_ffmpeg is not None:
        try:
            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            pass
    candidates.append("ffmpeg")

    for cmd in candidates:
        try:
            r = subprocess.run([cmd, "-version"], capture_output=True, text=True, timeout=6)
            if r.returncode == 0:
                return cmd
        except Exception:
            continue
    return None


def _wav_to_mp3(input_wav: str, output_mp3: str) -> None:
    ffmpeg_cmd = _find_ffmpeg()
    if not ffmpeg_cmd:
        raise LocalTTSError("No se encontró ffmpeg para convertir WAV->MP3.")

    os.makedirs(os.path.dirname(os.path.abspath(output_mp3)) or ".", exist_ok=True)
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_wav,
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "3",
        output_mp3,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if r.returncode != 0 or not os.path.exists(output_mp3) or os.path.getsize(output_mp3) == 0:
        raise LocalTTSError(f"Falló conversión a MP3: {(r.stderr or '').strip()[:300]}")


def _normalize_text(text: str) -> str:
    t = (text or "").replace("\u200b", " ").strip()
    if not t:
        raise LocalTTSError("El texto está vacío.")
    return t


def _download_file(url: str, dest: Path, *, timeout_s: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=max(30, int(timeout_s))) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _ensure_kokoro_assets(timeout_s: int = 1200) -> tuple[str, str]:
    if not KOKORO_MODEL_PATH.exists() or KOKORO_MODEL_PATH.stat().st_size == 0:
        _download_file(KOKORO_MODEL_URL, KOKORO_MODEL_PATH, timeout_s=timeout_s)
    if not KOKORO_VOICES_PATH.exists() or KOKORO_VOICES_PATH.stat().st_size == 0:
        _download_file(KOKORO_VOICES_URL, KOKORO_VOICES_PATH, timeout_s=timeout_s)
    return str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH)


def _default_kokoro_voice() -> str:
    val = (os.environ.get("LOCAL_TTS_KOKORO_VOICE") or "").strip()
    if val and val not in KOKORO_BLOCKED_VOICES:
        return val
    return "em_santa"


def _default_kokoro_speed() -> float:
    try:
        v = float((os.environ.get("LOCAL_TTS_KOKORO_SPEED") or "0.95").strip())
        return max(0.5, min(2.0, v))
    except Exception:
        return 0.95


def _render_with_kokoro(
    *,
    text: str,
    output_wav: str,
    voice: str,
    speed: float,
    lang: str,
    timeout_s: int,
) -> None:
    if Kokoro is None:
        raise LocalTTSError(
            "Falta 'kokoro-onnx'. Instala dependencias del entorno para usar voz local de alta calidad."
        )

    model_path, voices_path = _ensure_kokoro_assets(timeout_s=max(300, int(timeout_s)))
    kokoro = Kokoro(model_path, voices_path)
    available = set(kokoro.get_voices())
    selected = (voice or "").strip() or _default_kokoro_voice()
    if selected in KOKORO_BLOCKED_VOICES:
        selected = _default_kokoro_voice()
    if selected not in available:
        for candidate in KOKORO_PREFERRED_VOICES:
            if candidate in available and candidate not in KOKORO_BLOCKED_VOICES:
                selected = candidate
                break
        else:
            raise LocalTTSError("No hay voces permitidas disponibles en Kokoro.")

    samples, sample_rate = kokoro.create(
        text=text,
        voice=selected,
        speed=max(0.5, min(2.0, float(speed))),
        lang=(lang or "es").strip() or "es",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_wav)) or ".", exist_ok=True)
    sf.write(output_wav, np.asarray(samples), int(sample_rate))
    if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
        raise LocalTTSError("Kokoro no generó WAV válido.")


def synthesize_local_voice(
    text: str,
    output_path: str,
    *,
    engine: str = "kokoro",
    kokoro_voice: str | None = None,
    kokoro_speed: float | None = None,
    kokoro_lang: str = "es",
    piper_model: str | None = None,
    piper_executable: str | None = None,
    timeout_s: int = 120,
) -> LocalTTSResult:
    text_norm = _normalize_text(text)
    out_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)

    ext = Path(out_abs).suffix.lower()
    needs_mp3 = ext == ".mp3"
    if ext not in {".wav", ".mp3"}:
        raise LocalTTSError("La salida debe terminar en .wav o .mp3")

    tmp_wav = out_abs if not needs_mp3 else str(Path(out_abs).with_suffix(".tmp.wav"))

    eng = (engine or "kokoro").strip().lower()
    if eng != "kokoro":
        raise LocalTTSError("Motor no soportado. Este módulo usa solo 'kokoro'.")

    _render_with_kokoro(
        text=text_norm,
        output_wav=tmp_wav,
        voice=(kokoro_voice or "").strip() or _default_kokoro_voice(),
        speed=float(kokoro_speed if kokoro_speed is not None else _default_kokoro_speed()),
        lang=(kokoro_lang or "es").strip() or "es",
        timeout_s=timeout_s,
    )

    if needs_mp3:
        try:
            _wav_to_mp3(tmp_wav, out_abs)
        finally:
            try:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
            except Exception:
                pass

    return LocalTTSResult(
        output_path=out_abs,
        engine=eng,
        metadata={
            "voice": (kokoro_voice or _default_kokoro_voice()),
            "lang": (kokoro_lang or "es"),
        },
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTS local de alta calidad (Kokoro): texto -> audio (WAV/MP3)")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", help="Texto literal a sintetizar")
    source_group.add_argument("--text-file", help="Ruta a archivo de texto UTF-8")

    parser.add_argument("--out", required=True, help="Ruta de salida .wav o .mp3")
    parser.add_argument("--engine", default="kokoro", choices=["kokoro"], help="Motor TTS local")
    parser.add_argument("--voice", default=None, help="Voz Kokoro (es: ef_dora, em_alex, em_santa)")
    parser.add_argument("--speed", type=float, default=None, help="Velocidad Kokoro (0.5 - 2.0). Default 0.95")
    parser.add_argument("--lang", default="es", help="Idioma para g2p de Kokoro (recomendado: es)")

    parser.add_argument("--timeout", type=int, default=120, help="Timeout global por síntesis")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    text = ""
    if args.text_file:
        if not os.path.exists(args.text_file):
            print(f"[LOCAL-TTS] ERROR: No existe --text-file: {args.text_file}", file=sys.stderr)
            return 2
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text or ""

    try:
        result = synthesize_local_voice(
            text=text,
            output_path=args.out,
            engine=args.engine,
            kokoro_voice=args.voice,
            kokoro_speed=args.speed,
            kokoro_lang=args.lang,
            timeout_s=args.timeout,
        )
        voz_out = result.metadata.get("voice", "") if result.metadata else ""
        print(f"[LOCAL-TTS] OK -> {result.output_path} (engine={result.engine}, voice={voz_out})")
        return 0
    except Exception as exc:
        print(f"[LOCAL-TTS] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
