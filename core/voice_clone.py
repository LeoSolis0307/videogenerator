import argparse
import audioop
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


class VoiceCloneError(RuntimeError):
    pass


@dataclass(slots=True)
class VoiceCloneResult:
    output_path: str
    backend: str
    model_name: str


XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def _dbg(msg: str) -> None:
    if (os.environ.get("VOICE_CLONE_DEBUG") or "").strip().lower() in {"1", "true", "yes", "on", "si", "s"}:
        print(f"[VOICE-CLONE][DEBUG] {msg}")


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


def _ensure_audio_exists(path: str) -> str:
    p = os.path.abspath((path or "").strip().strip('"'))
    if not p or not os.path.exists(p):
        raise VoiceCloneError(f"No existe el archivo de referencia: {path}")
    ext = Path(p).suffix.lower()
    if ext not in {".wav", ".mp3", ".m4a"}:
        raise VoiceCloneError("La referencia debe ser .wav, .mp3 o .m4a")
    return p


def _prepare_reference_wav(input_audio: str, *, timeout_s: int = 120) -> str:
    ffmpeg_cmd = _find_ffmpeg()
    if not ffmpeg_cmd:
        raise VoiceCloneError("No se encontró ffmpeg para preparar audio de referencia.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    out_wav = tmp.name
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_audio,
        "-ac",
        "1",
        "-ar",
        "24000",
        out_wav,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(30, int(timeout_s)))
    if r.returncode != 0 or not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
        try:
            os.remove(out_wav)
        except Exception:
            pass
        raise VoiceCloneError(f"No se pudo preparar la referencia WAV: {(r.stderr or '').strip()[:300]}")
    return out_wav


def _optimize_reference_wav(
    input_wav: str,
    *,
    max_seconds: int | None = None,
    trim_silence: bool = False,
    timeout_s: int = 120,
) -> str:
    if (not trim_silence) and (not max_seconds or int(max_seconds) <= 0):
        return input_wav

    ffmpeg_cmd = _find_ffmpeg()
    if not ffmpeg_cmd:
        return input_wav

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    out_wav = tmp.name

    cmd = [
        ffmpeg_cmd,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_wav,
    ]

    if trim_silence:
        # Trim silence at start and end to improve speaker conditioning quality.
        cmd.extend(
            [
                "-af",
                "silenceremove=start_periods=1:start_duration=0.25:start_threshold=-40dB,"
                "areverse,"
                "silenceremove=start_periods=1:start_duration=0.25:start_threshold=-40dB,"
                "areverse",
            ]
        )

    if max_seconds and int(max_seconds) > 0:
        cmd.extend(["-t", str(int(max_seconds))])

    cmd.extend(["-ac", "1", "-ar", "24000", out_wav])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(30, int(timeout_s)))
    except Exception:
        r = None

    if (
        r is None
        or r.returncode != 0
        or not os.path.exists(out_wav)
        or os.path.getsize(out_wav) == 0
    ):
        try:
            if os.path.exists(out_wav):
                os.remove(out_wav)
        except Exception:
            pass
        return input_wav

    return out_wav


def _select_best_energy_window_wav(
    input_wav: str,
    *,
    target_seconds: int,
    timeout_s: int = 120,
) -> str:
    if int(target_seconds) <= 0:
        return input_wav

    ffmpeg_cmd = _find_ffmpeg()
    if not ffmpeg_cmd:
        return input_wav

    try:
        with wave.open(input_wav, "rb") as wf:
            frame_rate = int(wf.getframerate() or 0)
            n_frames = int(wf.getnframes() or 0)
            sample_width = int(wf.getsampwidth() or 0)
            channels = int(wf.getnchannels() or 0)
            if frame_rate <= 0 or n_frames <= 0 or sample_width <= 0 or channels <= 0:
                return input_wav

            total_seconds = n_frames / float(frame_rate)
            target_s = float(max(1, int(target_seconds)))
            if total_seconds <= target_s + 0.05:
                return input_wav

            chunk_s = 0.25
            chunk_frames = max(1, int(frame_rate * chunk_s))
            chunk_energies: list[float] = []
            while True:
                raw = wf.readframes(chunk_frames)
                if not raw:
                    break
                try:
                    rms = float(audioop.rms(raw, sample_width))
                except Exception:
                    rms = 0.0
                chunk_energies.append(rms * rms)
    except Exception:
        return input_wav

    if not chunk_energies:
        return input_wav

    window_chunks = max(1, int(round(target_s / 0.25)))
    if len(chunk_energies) <= window_chunks:
        return input_wav

    prefix = [0.0]
    for e in chunk_energies:
        prefix.append(prefix[-1] + e)

    best_start = 0
    best_score = -1.0
    limit = len(chunk_energies) - window_chunks + 1
    for start in range(limit):
        end = start + window_chunks
        score = prefix[end] - prefix[start]
        if score > best_score:
            best_score = score
            best_start = start

    start_s = best_start * 0.25
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    out_wav = tmp.name
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        str(int(target_seconds)),
        "-i",
        input_wav,
        "-ac",
        "1",
        "-ar",
        "24000",
        out_wav,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(30, int(timeout_s)))
    except Exception:
        r = None

    if (
        r is None
        or r.returncode != 0
        or not os.path.exists(out_wav)
        or os.path.getsize(out_wav) == 0
    ):
        try:
            if os.path.exists(out_wav):
                os.remove(out_wav)
        except Exception:
            pass
        return input_wav

    return out_wav


def _wav_to_mp3(input_wav: str, output_mp3: str, *, timeout_s: int = 120) -> None:
    ffmpeg_cmd = _find_ffmpeg()
    if not ffmpeg_cmd:
        raise VoiceCloneError("No se encontró ffmpeg para convertir WAV -> MP3.")

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
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(30, int(timeout_s)))
    if r.returncode != 0 or not os.path.exists(output_mp3) or os.path.getsize(output_mp3) == 0:
        raise VoiceCloneError(f"Falló conversión a MP3: {(r.stderr or '').strip()[:300]}")


def _select_backend(prefer_gpu: bool = True) -> tuple[str, object | None]:
    if not prefer_gpu:
        return "cpu", None

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", None
    except Exception:
        pass

    try:
        import torch_directml  # type: ignore

        dml_device = torch_directml.device()
        return "directml", dml_device
    except Exception:
        pass

    return "cpu", None


def _apply_transformers_compat_shim() -> None:
    """Expose legacy symbols at transformers top-level for Coqui TTS compatibility."""
    try:
        import transformers  # type: ignore
    except Exception:
        return

    try:
        has_beam = hasattr(transformers, "BeamSearchScorer")
    except Exception:
        has_beam = False

    if has_beam:
        return

    try:
        from transformers.generation import BeamSearchScorer as _BeamSearchScorer  # type: ignore

        setattr(transformers, "BeamSearchScorer", _BeamSearchScorer)
    except Exception:
        return


def clone_voice_to_audio(
    *,
    text: str,
    reference_audio_path: str,
    output_path: str,
    language: str = "es",
    model_name: str = XTTS_MODEL,
    prefer_gpu: bool = True,
    timeout_s: int = 300,
    reference_max_seconds: int | None = None,
    reference_trim_silence: bool = False,
    reference_pick_best_segment: bool = False,
) -> VoiceCloneResult:
    _dbg("Inicio clone_voice_to_audio")
    text_norm = (text or "").strip()
    if not text_norm:
        raise VoiceCloneError("El texto a sintetizar está vacío.")

    ref_abs = _ensure_audio_exists(reference_audio_path)
    out_abs = os.path.abspath((output_path or "").strip())
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    ext = Path(out_abs).suffix.lower()
    if ext not in {".wav", ".mp3"}:
        raise VoiceCloneError("La salida debe terminar en .wav o .mp3")

    backend, backend_device = _select_backend(prefer_gpu=prefer_gpu)
    _dbg(f"Backend seleccionado: {backend}")

    # Avoid interactive ToS prompt when downloading Coqui models in non-interactive runs.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    _apply_transformers_compat_shim()

    try:
        from TTS.api import TTS
    except Exception as exc:
        raise VoiceCloneError(
            "Falta Coqui TTS o no es compatible con este Python. "
            "Usa Python 3.10 e instala: pip install -r requirements.voice_clone.txt"
        ) from exc

    ref_wav = ""
    ref_wav_opt = ""
    ref_wav_best = ""
    tmp_out_wav = out_abs if ext == ".wav" else str(Path(out_abs).with_suffix(".tmp.wav"))
    try:
        _dbg("Preparando referencia WAV")
        ref_wav = _prepare_reference_wav(ref_abs, timeout_s=timeout_s)
        _dbg(f"Referencia preparada: {ref_wav}")
        ref_wav_opt = _optimize_reference_wav(
            ref_wav,
            max_seconds=reference_max_seconds,
            trim_silence=reference_trim_silence,
            timeout_s=timeout_s,
        )
        _dbg(f"Referencia optimizada: {ref_wav_opt}")
        if reference_pick_best_segment and reference_max_seconds and int(reference_max_seconds) > 0:
            ref_wav_best = _select_best_energy_window_wav(
                ref_wav_opt or ref_wav,
                target_seconds=int(reference_max_seconds),
                timeout_s=timeout_s,
            )
            _dbg(f"Mejor tramo seleccionado: {ref_wav_best}")
        speaker_ref = ref_wav_best or ref_wav_opt or ref_wav
        _dbg(f"Referencia final usada: {speaker_ref}")

        gpu_flag = backend == "cuda"
        _dbg("Inicializando modelo TTS")
        tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu_flag)
        _dbg("Modelo TTS cargado")
        if backend == "directml" and backend_device is not None:
            try:
                tts.to(backend_device)
            except Exception:
                backend = "cpu"

        _dbg("Iniciando sintesis")

        tts.tts_to_file(
            text=text_norm,
            speaker_wav=speaker_ref,
            language=(language or "es").strip() or "es",
            file_path=tmp_out_wav,
            split_sentences=True,
        )
        _dbg("Sintesis finalizada")
        if not os.path.exists(tmp_out_wav) or os.path.getsize(tmp_out_wav) == 0:
            raise VoiceCloneError("La clonación no generó un archivo de audio válido.")

        if ext == ".mp3":
            _wav_to_mp3(tmp_out_wav, out_abs, timeout_s=timeout_s)
            try:
                if os.path.exists(tmp_out_wav):
                    os.remove(tmp_out_wav)
            except Exception:
                pass

        return VoiceCloneResult(output_path=out_abs, backend=backend, model_name=model_name)
    finally:
        if ref_wav_best and ref_wav_best not in {ref_wav_opt, ref_wav}:
            try:
                if os.path.exists(ref_wav_best):
                    os.remove(ref_wav_best)
            except Exception:
                pass
        if ref_wav_opt and ref_wav_opt != ref_wav:
            try:
                if os.path.exists(ref_wav_opt):
                    os.remove(ref_wav_opt)
            except Exception:
                pass
        if ref_wav:
            try:
                if os.path.exists(ref_wav):
                    os.remove(ref_wav)
            except Exception:
                pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clonación de voz local con XTTS v2 (entrada WAV/MP3/M4A)")
    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", help="Texto literal a sintetizar")
    source_group.add_argument("--text-file", help="Ruta a archivo de texto UTF-8")

    p.add_argument("--ref", required=True, help="Audio de referencia de voz (.wav, .mp3 o .m4a)")
    p.add_argument("--out", required=True, help="Salida final (.wav o .mp3)")
    p.add_argument("--lang", default="es", help="Idioma del texto (por defecto: es)")
    p.add_argument("--model", default=XTTS_MODEL, help="Modelo Coqui TTS")
    p.add_argument("--cpu", action="store_true", help="Forzar CPU")
    p.add_argument("--timeout", type=int, default=300, help="Timeout de operaciones en segundos")
    p.add_argument("--ref-max-sec", type=int, default=0, help="Recorta referencia a N segundos (0 = desactivado)")
    p.add_argument("--ref-trim-silence", action="store_true", help="Recorta silencios al inicio/fin de la referencia")
    p.add_argument("--ref-pick-best", action="store_true", help="Selecciona el tramo de mayor energía vocal dentro de la referencia")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.text_file:
        if not os.path.exists(args.text_file):
            print(f"[VOICE-CLONE] ERROR: No existe --text-file: {args.text_file}")
            return 2
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text or ""

    try:
        res = clone_voice_to_audio(
            text=text,
            reference_audio_path=args.ref,
            output_path=args.out,
            language=args.lang,
            model_name=args.model,
            prefer_gpu=not bool(args.cpu),
            timeout_s=args.timeout,
            reference_max_seconds=(int(args.ref_max_sec) if int(args.ref_max_sec) > 0 else None),
            reference_trim_silence=bool(args.ref_trim_silence),
            reference_pick_best_segment=bool(args.ref_pick_best),
        )
        print(
            f"[VOICE-CLONE] OK -> {res.output_path} "
            f"(backend={res.backend}, model={res.model_name})"
        )
        return 0
    except Exception as exc:
        print(f"[VOICE-CLONE] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
