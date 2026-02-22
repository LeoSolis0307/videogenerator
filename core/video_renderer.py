import json
import math
import os
import random
import re
import subprocess
import wave
import shutil

import imageio_ffmpeg
from core.config import settings

_LEGACY_DEFAULT_INTRO_PATH = r"C:\Users\Leonardo\Downloads\video\intro.mp4"
_LOCAL_DEFAULT_INTRO_PATH = os.path.abspath(os.path.join(os.getcwd(), "intro.mp4"))
DEFAULT_INTRO_PATH = _LOCAL_DEFAULT_INTRO_PATH if os.path.exists(_LOCAL_DEFAULT_INTRO_PATH) else _LEGACY_DEFAULT_INTRO_PATH
MIN_VIDEO_SEC = 15 * 60
MAX_VIDEO_SEC = 30 * 60
DEFAULT_VIDEOS_DIR = r"C:\Users\Leonardo\Downloads\video\media"
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

                                              
                                                                               
                                                     
                        


def _encoding_cfg() -> dict:
    def _opt_str(value) -> str:
        text = str(value).strip() if value is not None else ""
        if text.lower() in {"", "none", "null"}:
            return ""
        return text

    quality = (settings.video_quality or "").strip().lower()

    preset = settings.video_preset
    crf = settings.video_crf
    fps = settings.video_fps
    audio_bitrate = settings.audio_bitrate
    scale_flags = settings.video_scale_flags
    tune = settings.video_tune

    if quality in {"high", "alta"}:
        preset = preset or "slow"
        crf = crf or "18"
        audio_bitrate = audio_bitrate or "192k"
        scale_flags = scale_flags or "lanczos"
        tune = tune or "stillimage"
    elif quality in {"best", "max", "maxima", "máxima"}:
        preset = preset or "veryslow"
        crf = crf or "17"
        audio_bitrate = audio_bitrate or "256k"
        scale_flags = scale_flags or "lanczos"
        tune = tune or "stillimage"
                     
    # Fallbacks
    if not crf or not crf.isdigit():
        crf = "20"
    if not fps:
        fps = "25"

    return {
        "quality": quality,
        "preset": _opt_str(preset) or "veryfast",
        "crf": _opt_str(crf) or "20",
        "fps": _opt_str(fps) or "25",
        "audio_bitrate": _opt_str(audio_bitrate),
        "scale_flags": _opt_str(scale_flags),
        "tune": _opt_str(tune),
    }


def _scale_filter(size: int) -> str:
    cfg = _encoding_cfg()
    flags = cfg.get("scale_flags") or ""
    
    fit = settings.video_fit_mode.lower()
    mode = "decrease" if fit != "crop" else "increase"
    if flags:
        return f"scale={size}:{size}:force_original_aspect_ratio={mode}:flags={flags}"
    return f"scale={size}:{size}:force_original_aspect_ratio={mode}"


def _post_scale_fit(size: int) -> str:
    fit = settings.video_fit_mode.lower()
    if fit == "crop":
        return f"crop={size}:{size}:(iw-{size})/2:(ih-{size})/2"
                               
    return f"pad={size}:{size}:(%d-iw)/2:(%d-ih)/2" % (size, size)


def _build_per_stream_filters(idx: int, fps: str, dur_val: float, size: int = 768) -> list[str]:
    vlabel = f"v{idx}"
    parts: list[str] = []

    fps_val = str(fps)
                                                                               
    try:
        stop_frame = 1.0 / float(fps)
    except Exception:
        stop_frame = 0.04
    pad_sec = max(0.0, float(dur_val) - float(stop_frame))
    pad_sec_s = f"{pad_sec:.6f}"
    dur_s = f"{float(dur_val):.6f}"
    
    fit_mode = settings.video_fit_mode.lower()
    pad_style = settings.video_pad_style.lower()
    blur_style = settings.video_blur
    kenburns = settings.video_kenburns
    kb_rate = settings.video_kb_rate
    kb_max = settings.video_kb_max

    if fit_mode == "pad" and pad_style == "blur":
        parts.append(f"[{idx}:v]split=2[s{idx}a][s{idx}b]")
                                       
        parts.append(
            f"[s{idx}a]fps={fps_val},scale={size}:{size}:force_original_aspect_ratio=increase,"
            f"crop={size}:{size},format=yuv420p,boxblur={blur_style},setsar=1[bg{idx}]"
        )
                                           
        if kenburns:
            frames = max(1, int(round(float(fps) * float(dur_val))))
            parts.append(
                f"[s{idx}b]fps={fps_val},zoompan=z='min(zoom+{kb_rate},{kb_max})':d={frames}:s={size}x{size},format=yuv420p,setsar=1[fg{idx}]"
            )
        else:
            parts.append(
                f"[s{idx}b]fps={fps_val},scale={size}:{size}:force_original_aspect_ratio=decrease,format=yuv420p,setsar=1[fg{idx}]"
            )
        parts.append(
            f"[bg{idx}][fg{idx}]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2,"
            f"tpad=stop_mode=clone:stop_duration={pad_sec_s},trim=duration={dur_s},"
            f"format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS,fps={fps_val}[{vlabel}]"
        )
        return parts

    if kenburns:
        frames = max(1, int(round(float(fps) * float(dur_val))))
        parts.append(
            f"[{idx}:v]fps={fps_val},zoompan=z='min(zoom+{kb_rate},{kb_max})':d={frames}:s={size}x{size},"
            f"format=yuv420p,setsar=1,tpad=stop_mode=clone:stop_duration={pad_sec_s},trim=duration={dur_s},"
            f"settb=AVTB,setpts=PTS-STARTPTS,fps={fps_val}[{vlabel}]"
        )
        return parts

    parts.append(
        f"[{idx}:v]fps={fps_val},{_scale_filter(size)},{_post_scale_fit(size)},setsar=1,format=yuv420p,"
        f"tpad=stop_mode=clone:stop_duration={pad_sec_s},trim=duration={dur_s},"
        f"settb=AVTB,setpts=PTS-STARTPTS,fps={fps_val}[{vlabel}]"
    )
    return parts


def _pick_ffmpeg() -> str:
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin:
        if os.path.exists(env_bin):
            return env_bin
        else:
            print(f"[VIDEO] FFMPEG_BIN no existe, ignorando: {env_bin}")

    local_bin = os.path.join(os.path.dirname(imageio_ffmpeg.__file__), "binaries", "ffmpeg-win-x86_64-v7.1.exe")

                                                                                        
    if os.path.exists(local_bin):
        os.environ["FFMPEG_BIN"] = local_bin
        print(f"[VIDEO] ffmpeg (paquete) seleccionado: {local_bin}")
        return local_bin

    raise FileNotFoundError(
        "No se encontró ffmpeg empaquetado (imageio_ffmpeg). "
        "Instala/rehaz el entorno o define FFMPEG_BIN apuntando a un exe válido."
    )


def _ffmpeg_path(path: str) -> str:
                                                                     
    return os.path.abspath(path).replace("\\", "/")


def _png_signature_ok(path: str) -> bool:
    try:
        if not path or not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            sig = f.read(8)
        return sig == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False

def _looks_like_avif_container(path: str) -> bool:
    try:
        if not path or not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            head = f.read(64)
                                                 
        return (b"ftypavif" in head) or (b"ftypmif1" in head and b"avif" in head)
    except Exception:
        return False


def _looks_like_ffmpeg_decode_error(stderr_text: str) -> bool:
    if not stderr_text:
        return False
    s = stderr_text.lower()
                                                                                                           
    needles = [
        "invalid png signature",
        "error submitting packet",
        "decoding error",
        "invalid data found when processing input",
        "could not find codec parameters",
        "error while decoding",
    ]
    return any(n in s for n in needles)


def _make_placeholder_png(ffmpeg: str, out_path: str, size: int = 768) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={size}x{size}",
        "-frames:v",
        "1",
        _ffmpeg_path(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=20)
    if os.path.exists(out_path) and os.path.getsize(out_path) >= 256 and _png_signature_ok(out_path):
        return out_path
    raise RuntimeError(f"[VIDEO] No se pudo generar placeholder PNG: {out_path}")


def _sanitize_image_for_ffmpeg(
    ffmpeg: str,
    img_path: str,
    out_dir: str,
    idx: int,
    report: list[dict] | None = None,
) -> str:
    def _add_report(status: str, used: str, note: str | None = None) -> None:
        if report is None:
            return
        report.append(
            {
                "index": idx + 1,
                "original": os.path.abspath(img_path),
                "used": os.path.abspath(used),
                "status": status,
                "note": (note or "").strip(),
            }
        )

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"[VIDEO] No se encontró la imagen: {img_path}")

    try:
        if os.path.getsize(img_path) < 1024:
            raise ValueError("archivo demasiado pequeño")
    except Exception:
        raise ValueError(f"[VIDEO] Imagen inválida (tamaño): {img_path}")

    ext = os.path.splitext(img_path)[1].lower()

                                                                                           
    force_convert = ext in {".jpg", ".jpeg"} and _looks_like_avif_container(img_path)
    if force_convert:
        print(f"[VIDEO] WARNING: Detectado AVIF con extensión JPG/JPEG. Convirtiendo a PNG: {img_path}")

    if not force_convert and ext == ".png" and not _png_signature_ok(img_path):
                                                                 
        print(f"[VIDEO] WARNING: PNG inválido (signature). Intentando reparar: {img_path}")
    elif not force_convert:
                                         
        try:
            test_cmd = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                _ffmpeg_path(img_path),
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ]
            res = subprocess.run(test_cmd, check=True, capture_output=True, text=True, timeout=25)
            if _looks_like_ffmpeg_decode_error((res.stderr or "") + (res.stdout or "")):
                raise RuntimeError("FFmpeg reportó error de decodificación")
            _add_report("ok", img_path)
            return img_path
        except Exception:
            print(f"[VIDEO] WARNING: Imagen no decodificable. Intentando convertir a PNG: {img_path}")

                                                  
    os.makedirs(out_dir, exist_ok=True)
    repaired = os.path.join(out_dir, f"img_{idx:02d}_repaired.png")
    conv_cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        _ffmpeg_path(img_path),
        "-frames:v",
        "1",
        _ffmpeg_path(repaired),
    ]
    try:
        res = subprocess.run(conv_cmd, check=True, capture_output=True, text=True, timeout=60)
        if _looks_like_ffmpeg_decode_error((res.stderr or "") + (res.stdout or "")):
            raise RuntimeError("FFmpeg reportó error al convertir")
        if os.path.exists(repaired) and os.path.getsize(repaired) >= 1024 and _png_signature_ok(repaired):
                                                                                       
            try:
                test_repaired = [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    _ffmpeg_path(repaired),
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ]
                res2 = subprocess.run(test_repaired, check=True, capture_output=True, text=True, timeout=25)
                if not _looks_like_ffmpeg_decode_error((res2.stderr or "") + (res2.stdout or "")):
                    _add_report("repaired", repaired)
                    return repaired
            except Exception:
                pass
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "")[:800]
        print(
            f"[VIDEO] ERROR: Imagen inválida en input #{idx + 1}. "
            f"FFmpeg no pudo decodificar/convertir: {img_path}\n{err}"
        )
    except Exception as e:
        print(
            f"[VIDEO] ERROR: Imagen inválida en input #{idx + 1}. "
            f"FFmpeg no pudo decodificar/convertir: {img_path} ({e})"
        )

                                                                
    placeholder = os.path.join(out_dir, f"img_{idx:02d}_PLACEHOLDER.png")
    try:
        ph = _make_placeholder_png(ffmpeg, placeholder, size=768)
        print(f"[VIDEO] WARNING: Usando placeholder para input #{idx + 1}: {img_path}")
        _add_report("placeholder", ph, note="No se pudo decodificar/convertir; se usó placeholder")
        return ph
    except Exception:
                                                                                    
        raise RuntimeError(f"[VIDEO] Imagen inválida en input #{idx + 1}: {img_path}")


def _audio_duration_seconds(path: str) -> float:
    if not os.path.exists(path):
        return 0.0

    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        try:
            with wave.open(path, "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate() or 1
                return frames / float(rate)
        except Exception:
            return 0.0

    import re

    def _parse_duration_text(text: str) -> float:
                                             
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:[\.,]\d+)?)", text, re.IGNORECASE)
        if m:
            h, mnt, sec = m.groups()
            return int(h) * 3600 + int(mnt) * 60 + float(sec.replace(",", "."))
                       
        m = re.search(r"Duraci[óo]n:\s*(\d+):(\d+):(\d+(?:[\.,]\d+)?)", text, re.IGNORECASE)
        if m:
            h, mnt, sec = m.groups()
            return int(h) * 3600 + int(mnt) * 60 + float(sec.replace(",", "."))
                                            
        times = re.findall(r"time=(\d+):(\d+):(\d+(?:[\.,]\d+)?)", text)
        if times:
            h, mnt, sec = times[-1]
            return int(h) * 3600 + int(mnt) * 60 + float(sec.replace(",", "."))
        return 0.0

    def _run(cmd) -> str:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
            return ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
        except Exception:
            return ""

    try:
        ffmpeg_bin = _pick_ffmpeg()
    except Exception:
        raise

    ffprobe_guess = ffmpeg_bin.replace("ffmpeg", "ffprobe")

                                 
    if os.path.exists(ffprobe_guess):
        out = _run([ffprobe_guess, "-v", "quiet", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path])
        try:
            val = float(out.strip().splitlines()[0])
            if val > 0:
                return val
        except Exception:
            pass

                        
    out = _run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path])
    try:
        val = float(out.strip().splitlines()[0])
        if val > 0:
            return val
    except Exception:
        pass

                           
    out = _run([ffmpeg_bin, "-v", "info", "-i", path])
    dur = _parse_duration_text(out)
    if dur > 0:
        return dur

                                       
    out = _run([ffmpeg_bin, "-v", "info", "-i", path, "-f", "null", "-"])
    dur = _parse_duration_text(out)
    if dur > 0:
        return dur

                     
    if out:
        print(f"[VIDEO] WARNING: No se pudo leer duración. Fragmento de salida:\n{out[:4000]}")

    return 0.0


def audio_duration_seconds(path: str) -> float:
    return _audio_duration_seconds(path)


def _debug_dump_duration(path: str):
    print(f"[VIDEO-DEBUG] Dump de duración para: {path}")
    try:
        ffmpeg_bin = _pick_ffmpeg()
        ffprobe_guess = ffmpeg_bin.replace("ffmpeg", "ffprobe")
    except Exception as e:
        ffmpeg_bin = "ffmpeg"
        ffprobe_guess = "ffprobe"
        print(f"[VIDEO-DEBUG] No se pudo obtener ffmpeg de imageio: {e}")

    cmds = [
        [ffprobe_guess, "-v", "debug", "-show_format", "-show_streams", path],
        ["ffprobe", "-v", "debug", "-show_format", "-show_streams", path],
        [ffmpeg_bin, "-v", "info", "-i", path, "-f", "null", "-"],
        [ffmpeg_bin, "-v", "info", "-i", path],
    ]

    for idx, cmd in enumerate(cmds, 1):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            out = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
            print(f"[VIDEO-DEBUG] Cmd {idx}: {' '.join(cmd[:4])} ...")
            print(out[:4000])
        except Exception as e:
            print(f"[VIDEO-DEBUG] Cmd {idx} falló: {e}")


def _pick_video_file(videos_dir: str | None = None) -> str | None:
    base = videos_dir or DEFAULT_VIDEOS_DIR
    base_abs = os.path.abspath(base)
    if not os.path.isdir(base_abs):
        print(f"[VIDEO] Carpeta no encontrada: {base_abs}")
        return None

    candidatos = []
    nombres = os.listdir(base_abs)
    hubo_con_0 = False
    for nombre in nombres:
        if nombre.startswith("0"):
            hubo_con_0 = True
            continue
        ext = os.path.splitext(nombre)[1].lower()
        if ext in SUPPORTED_VIDEO_EXTS:
            candidatos.append(os.path.join(base_abs, nombre))

    if not candidatos:
        if hubo_con_0 and any(os.path.splitext(n)[1].lower() in SUPPORTED_VIDEO_EXTS for n in nombres):
            print("[VIDEO] Todos los videos disponibles ya fueron usados (empiezan con 0).")
        else:
            print(f"[VIDEO] No se hallaron videos en {base_abs} (ext: {', '.join(sorted(SUPPORTED_VIDEO_EXTS))})")
        return None

    elegido = random.choice(candidatos)
    print(f"[VIDEO] Video elegido: {elegido}")
    return elegido


def select_video_base(videos_dir: str | None = None) -> tuple[str | None, float]:
    ruta = _pick_video_file(videos_dir)
    if not ruta:
        return None, 0.0
    dur = _audio_duration_seconds(ruta)
    print(f"[VIDEO] Duración detectada para base: {dur:.2f}s")
    if dur <= 0:
        _debug_dump_duration(ruta)
    return ruta, dur


def _calc_speed_and_padding(video_dur: float, audio_dur: float, *, max_speed: float = 2.5) -> tuple[float, float]:
    if audio_dur <= 0 or video_dur <= 0:
        return 1.0, 0.0

                                                                      
    diff = abs(video_dur - audio_dur)
    if diff <= max(2.0, audio_dur * 0.02):
        return 1.0, 0.0

    if video_dur > audio_dur:
        speed = max(1.0, video_dur / audio_dur)
        speed = min(speed, max_speed)
                                                                                   
        new_dur = video_dur / speed
        if new_dur > audio_dur:
            return speed, max(0.0, new_dur - audio_dur)
        return speed, 0.0

                                                               
    return 1.0, max(0.0, audio_dur - video_dur)


def render_video_base_con_audio(video_path: str, audio_path: str, carpeta_salida: str, *, videos_dir: str | None = None):
    ffmpeg = _pick_ffmpeg()

    video_fs = video_path or _pick_video_file(videos_dir)
    if not video_fs:
        raise FileNotFoundError("[VIDEO] No hay video base disponible")

    video_abs = os.path.abspath(video_fs)
    audio_abs = os.path.abspath(audio_path)
    salida = os.path.join(os.path.abspath(carpeta_salida), "Video_Final.mp4")

    if not os.path.exists(video_abs):
        raise FileNotFoundError(f"[VIDEO] No se encontró el video base: {video_abs}")
    if not os.path.exists(audio_abs):
        raise FileNotFoundError(f"[VIDEO] No se encontró el audio: {audio_abs}")

    dur_video = _audio_duration_seconds(video_abs)
    dur_audio = _audio_duration_seconds(audio_abs)
    speed, pad_sec = _calc_speed_and_padding(dur_video, dur_audio)

    filtros = []
    if speed != 1.0:
        filtros.append(f"setpts=PTS/{speed}")
    cfg = _encoding_cfg()
    fps = cfg["fps"]

    filtros.append(_scale_filter(768))
    filtros.append("pad=768:768:(768-iw)/2:(768-ih)/2")
    filtros.append("setsar=1")
    filtros.append(f"fps={fps}")
    if pad_sec > 0.05:
        filtros.append(f"tpad=stop_duration={pad_sec:.3f}")

    filter_chain = ",".join(filtros)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        _ffmpeg_path(video_abs),
        "-i",
        _ffmpeg_path(audio_abs),
        "-filter_complex",
        f"[0:v]{filter_chain}[v0]",
        "-map",
        "[v0]",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-preset",
        cfg["preset"],
        "-crf",
        cfg["crf"],
        "-tune",
        "film",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        _ffmpeg_path(salida),
    ]

    if cfg.get("audio_bitrate"):
        cmd.insert(cmd.index("-movflags"), "-b:a")
        cmd.insert(cmd.index("-movflags"), cfg["audio_bitrate"])

    print(f"[VIDEO] Base: {video_abs} (dur ~{dur_video:.1f}s) | Audio ~{dur_audio:.1f}s")
    if speed != 1.0:
        print(f"[VIDEO] Acelerando video x{speed:.3f}")
    if pad_sec > 0.05:
        print(f"[VIDEO] Extendiendo último frame {pad_sec:.2f}s para empatar audio")

    subprocess.run(cmd, check=True)
    print(f"[VIDEO] OK: Video final renderizado: {salida}")

                                                                                        
    try:
        base_dir = os.path.dirname(video_abs)
        base_name = os.path.basename(video_abs)
        if not base_name.startswith("0"):
            new_name = "0" + base_name
            new_path = os.path.join(base_dir, new_name)
                                          
            while os.path.exists(new_path):
                new_name = "0" + new_name
                new_path = os.path.join(base_dir, new_name)
            os.rename(video_abs, new_path)
            print(f"[VIDEO] Renombrado para no reutilizar: {new_path}")
    except Exception as e:
        print(f"[VIDEO] WARNING: No se pudo renombrar el video usado: {e}")

    return salida


def _normalize_audio_to_wav(src: str, dst: str, *, rate: int = 48000, channels: int = 1) -> str:
    ffmpeg = _pick_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        src,
        "-ar",
        str(rate),
        "-ac",
        str(channels),
        "-sample_fmt",
        "s16",
        dst,
    ]
    subprocess.run(cmd, check=True)
    return dst


def combine_audios_with_silence(audios, carpeta, gap_seconds=4, *, min_seconds: int | None = MIN_VIDEO_SEC, max_seconds: int | None = MAX_VIDEO_SEC):
    if not audios:
        raise ValueError("[AUDIO] Lista de audios vacía")

    rate = 48000
    channels = 1
    sampwidth = 2          
    carpeta_abs = os.path.abspath(carpeta)
    os.makedirs(carpeta_abs, exist_ok=True)
    salida = os.path.join(carpeta_abs, "audio_con_silencios.wav")

    norm_paths = []
    try:
        for idx, a in enumerate(audios):
            if not os.path.exists(a):
                raise FileNotFoundError(f"[AUDIO] No se encontró: {a}")
            dst = os.path.join(carpeta_abs, f"audio_norm_{idx}.wav")
            norm_paths.append(_normalize_audio_to_wav(a, dst, rate=rate, channels=channels))

        max_frames = int(max_seconds * rate) if max_seconds else None
        min_frames = int(min_seconds * rate) if min_seconds else None
        total_frames = 0
        gap_frames = int(gap_seconds * rate)

        with wave.open(salida, "wb") as out_w:
            out_w.setnchannels(channels)
            out_w.setsampwidth(sampwidth)
            out_w.setframerate(rate)

            for i, npath in enumerate(norm_paths):
                with wave.open(npath, "rb") as r:
                    frames = r.readframes(r.getnframes())
                frames_len = len(frames) // (channels * sampwidth)

                if max_frames is not None and total_frames + frames_len > max_frames:
                    needed = max_frames - total_frames
                    if needed > 0:
                        out_w.writeframes(frames[: needed * channels * sampwidth])
                        total_frames += needed
                    break

                out_w.writeframes(frames)
                total_frames += frames_len

                is_last = i == len(norm_paths) - 1
                if not is_last and (max_frames is None or total_frames < max_frames):
                    add_gap = gap_frames
                    if max_frames is not None:
                        add_gap = min(add_gap, max_frames - total_frames)
                    out_w.writeframes(b"\x00" * add_gap * channels * sampwidth)
                    total_frames += add_gap

            if min_frames is not None and total_frames < min_frames:
                pad_frames = min_frames - total_frames
                out_w.writeframes(b"\x00" * pad_frames * channels * sampwidth)
                total_frames += pad_frames

        return salida
    finally:
        for p in norm_paths:
            try:
                os.remove(p)
            except Exception:
                pass


def render_video_ffmpeg(imagenes, audio, carpeta, tiempo_img=None, *, durations=None):
    print("[VIDEO] Preparando render con FFmpeg (imageio)...")

    if not imagenes:
        raise ValueError("[VIDEO] Lista de imágenes vacía")

    if not carpeta:
        raise ValueError("[VIDEO] Carpeta de salida no válida")

    ffmpeg = _pick_ffmpeg()
    print(f"[VIDEO] FFmpeg encontrado en: {ffmpeg}")

    carpeta_abs = os.path.abspath(carpeta)
    audio_abs = os.path.abspath(audio)

    if not os.path.exists(audio_abs):
        raise FileNotFoundError(f"[VIDEO] No se encontró el audio: {audio_abs}")

                                                                                                              
    imagenes_ok: list[str] = []
    report_entries: list[dict] = []
    repaired_dir = os.path.join(carpeta_abs, "_repaired")
    for idx, img in enumerate(imagenes):
        img_abs = os.path.abspath(img)
        imagenes_ok.append(_sanitize_image_for_ffmpeg(ffmpeg, img_abs, repaired_dir, idx, report=report_entries))

                                                                                     
    try:
        os.makedirs(repaired_dir, exist_ok=True)
        report_json = os.path.join(repaired_dir, "repaired_report.json")
        report_txt = os.path.join(repaired_dir, "repaired_report.txt")

        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report_entries, f, ensure_ascii=False, indent=2)

        ok_count = sum(1 for e in report_entries if e.get("status") == "ok")
        repaired_count = sum(1 for e in report_entries if e.get("status") == "repaired")
        ph_count = sum(1 for e in report_entries if e.get("status") == "placeholder")
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write("Sanitización de imágenes (antes de FFmpeg)\n")
            f.write(f"Total: {len(report_entries)} | ok={ok_count} | repaired={repaired_count} | placeholder={ph_count}\n\n")
            for e in report_entries:
                f.write(f"#{e.get('index')}: {e.get('status')}\n")
                f.write(f"  original: {e.get('original')}\n")
                f.write(f"  used:     {e.get('used')}\n")
                note = (e.get("note") or "").strip()
                if note:
                    f.write(f"  note:     {note}\n")
                f.write("\n")
        print(
            f"[VIDEO] Reporte de imágenes guardado en: {report_txt} "
            f"(ok={ok_count}, repaired={repaired_count}, placeholder={ph_count})"
        )
    except Exception as e:
        print(f"[VIDEO] WARNING: No se pudo guardar reporte de sanitización: {e}")

    dur_list = list(durations or [])
    if dur_list and len(dur_list) < len(imagenes):
        dur_list.extend([dur_list[-1]] * (len(imagenes) - len(dur_list)))

    if tiempo_img is None and not dur_list:
        dur_audio = max(1.0, _audio_duration_seconds(audio_abs))
        tiempo_img = max(1, math.ceil(dur_audio / len(imagenes)))

                                                                                           
                                                                                                         
                                                                                         
                                                                                   
                                                                                           
                                
    inputs: list[str] = [ffmpeg, "-y", "-nostdin", "-hide_banner", "-xerror"]
    filter_parts: list[str] = []
    vlabels: list[str] = []

    cfg = _encoding_cfg()
    fps = cfg["fps"]
    try:
        XF_MS = max(0, int(settings.video_crossfade_ms))
    except Exception:
        XF_MS = 250
    ENABLE_LOUDNORM = bool(settings.enable_loudnorm)

                                                    
                                                                                    
                                                                                  
    per_dur: list[float] = []
    if dur_list:
        for idx in range(len(imagenes_ok)):
            per_dur.append(max(0.5, float(dur_list[idx])))
    else:
        for _ in range(len(imagenes_ok)):
            per_dur.append(max(0.5, float(tiempo_img)))

    try:
        dur_audio_real = max(0.0, _audio_duration_seconds(audio_abs))
    except Exception:
        dur_audio_real = 0.0

    total_video = sum(per_dur)

                                                                                     
                                                                  
    overlap = 0.0
    if len(per_dur) > 1 and XF_MS > 0:
        raw_fade = max(0.02, float(XF_MS) / 1000.0)
        fps_num = float(fps)
        fade_frames = max(1, int(round(fps_num * raw_fade)))
        fade_dur = float(fade_frames) / fps_num
        overlap = (len(per_dur) - 1) * fade_dur

    effective_video = total_video - overlap
    if dur_audio_real > 0 and effective_video + 0.02 < dur_audio_real:
        extra = (dur_audio_real - effective_video) + 0.10
        per_dur[-1] += extra
        print(
            f"[VIDEO] Ajuste anti-recorte: extendiendo última imagen +{extra:.2f}s "
            f"(video efectivo {effective_video:.2f}s < audio {dur_audio_real:.2f}s; overlap~{overlap:.2f}s)"
        )

    for idx, (img, dur_val) in enumerate(zip(imagenes_ok, per_dur)):
                                                                                             
        inputs.extend(["-i", _ffmpeg_path(img)])

        vlabel = f"v{idx}"
        vlabels.append(f"[{vlabel}]")
                                                                                       
        parts = _build_per_stream_filters(idx, fps, dur_val, 768)
        filter_parts.extend(parts)

                             
    audio_input_index = len(imagenes)
    inputs.extend(["-i", _ffmpeg_path(audio_abs)])

    n = len(imagenes)
    filter_complex = ";".join(filter_parts)
    last_video_label = "[v]"
    if n == 1 or XF_MS <= 0:
                                               
        concat_in = "".join(vlabels)
        filter_complex += f";{concat_in}concat=n={n}:v=1:a=0[v]"
    else:
                              
                                                                                               
        raw_fade = max(0.02, float(XF_MS) / 1000.0)
        fps_num = float(fps)
        fade_frames = max(1, int(round(fps_num * raw_fade)))
        fade_dur = float(fade_frames) / fps_num
        cum = per_dur[0]
        offset = max(0.0, cum - fade_dur)
                                                                                               
        filter_complex += (
            f";{vlabels[0]}{vlabels[1]}xfade=transition=fade:duration={fade_dur}:offset={offset}[x1]"
            f";[x1]format=yuv420p,settb=AVTB,setsar=1,setpts=PTS-STARTPTS,fps={fps}[x1n]"
        )
        last = "[x1n]"
        for i in range(2, n):
            cum += per_dur[i - 1]
            offset = max(0.0, cum - (i * fade_dur))
                                                                                         
            next_label = f"[x{i}]"
            next_norm = f"[x{i}n]"
            filter_complex += (
                f";{last}{vlabels[i]}xfade=transition=fade:duration={fade_dur}:offset={offset}{next_label}"
                f";{next_label}format=yuv420p,settb=AVTB,setsar=1,setpts=PTS-STARTPTS,fps={fps}{next_norm}"
            )
            last = next_norm
        last_video_label = last

    salida = os.path.join(carpeta_abs, "Video_Final.mp4")

    cmd = (
        inputs
        + [
            "-filter_complex",
            filter_complex,
            "-map",
            last_video_label,
            "-map",
            f"{audio_input_index}:a:0",
            "-c:v",
            "libx264",
            "-preset",
            cfg["preset"],
            "-crf",
            cfg["crf"],
            "-pix_fmt",
            "yuv420p",
            "-r",
            fps,
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-shortest",
            _ffmpeg_path(salida),
        ]
    )

    if cfg.get("tune"):
        cmd.insert(cmd.index("-pix_fmt"), "-tune")
        cmd.insert(cmd.index("-pix_fmt"), cfg["tune"])

    if cfg.get("audio_bitrate"):
        cmd.insert(cmd.index("-movflags"), "-b:a")
        cmd.insert(cmd.index("-movflags"), cfg["audio_bitrate"])

    if ENABLE_LOUDNORM:
                                                           
        insert_at = cmd.index("-movflags") if "-movflags" in cmd else len(cmd)
        cmd.insert(insert_at, "loudnorm=I=-14:TP=-1.5:LRA=11")
        cmd.insert(insert_at, "-af")

    print("[VIDEO] Ejecutando FFmpeg...")
    subprocess.run(cmd, check=True)

    print(f"[VIDEO] OK: Video renderizado: {salida}")
    return salida


def _slugify_title(title: str | None) -> str:
    if not title:
        return "video"

    cleaned = re.sub(r"[\r\n]+", " ", title).strip()
    if not cleaned:
        return "video"

    words = cleaned.split()

                                                             
                                                       
    max_words = 20
    max_chars = 140

    parts: list[str] = []
    for w in words[:max_words]:
        candidate = (" ".join(parts + [w])).strip()
        if len(candidate) > max_chars:
            break
        parts.append(w)

    shortened = " ".join(parts).strip() or "video"
    shortened = re.sub(r"\s+", " ", shortened)

                                                                  
                                                                     
    safe = re.sub(r"[^\w\s#-]", "", shortened)
    safe = re.sub(r"\s+", " ", safe).strip("- _")
    return safe or "video"


def append_intro_to_video(video_final, intro_path=DEFAULT_INTRO_PATH, output_path=None, title_text=None):
    print("[VIDEO] Preparando concatenación con intro...")

    video_fs = os.path.abspath(video_final)
    intro_fs = os.path.abspath(intro_path)

    if not os.path.exists(video_fs):
        raise FileNotFoundError(f"[VIDEO] No se encontró el video final: {video_fs}")

    if not os.path.exists(intro_fs):
        raise FileNotFoundError(f"[VIDEO] No se encontró el intro: {intro_fs}")

                                                                                                               
    try:
        intro_size = os.path.getsize(intro_fs)
    except Exception:
        intro_size = -1
    if intro_size is not None and intro_size >= 0 and intro_size < 1024:
        raise ValueError(
            f"[VIDEO] El intro parece inválido (tamaño {intro_size} bytes): {intro_fs}. "
            "Reemplázalo por un MP4 real (no vacío) y vuelve a intentar."
        )

    ffmpeg = _pick_ffmpeg()
    base, ext = os.path.splitext(os.path.basename(video_fs))
    carpeta_salida = os.path.dirname(video_fs)
    nombre_final = _slugify_title(title_text) if title_text else base + "_con_intro"
    salida_fs = output_path or os.path.join(carpeta_salida, f"{nombre_final}{ext}")

    video_abs = _ffmpeg_path(video_fs)
    intro_abs = _ffmpeg_path(intro_fs)
    salida = _ffmpeg_path(salida_fs)

    cfg = _encoding_cfg()
    fps = cfg["fps"]

                                                                                            
                                                                                     
    mute_intro = (os.environ.get("INTRO_MUTE_AUDIO") or "").strip().lower() in {"1", "true", "yes", "si", "sí"}
    a1 = "[1:a]volume=0," if mute_intro else "[1:a]"
    filter_complex = (
        f"[0:v]{_scale_filter(768)},pad=768:768:(768-iw)/2:(768-ih)/2,format=yuv420p,setsar=1,settb=AVTB,setpts=PTS-STARTPTS,fps={fps}[v0];"
        f"[1:v]{_scale_filter(768)},pad=768:768:(768-iw)/2:(768-ih)/2,format=yuv420p,setsar=1,settb=AVTB,setpts=PTS-STARTPTS,fps={fps}[v1];"
        "[0:a]aresample=48000,asetpts=PTS-STARTPTS[a0];"
        f"{a1}aresample=48000,asetpts=PTS-STARTPTS[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
    )

    cmd = [
        ffmpeg, "-y", "-nostdin",
        "-i", video_abs,
        "-i", intro_abs,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-preset", cfg["preset"],
        "-crf", cfg["crf"],
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "48000",
        "-movflags", "+faststart",
        salida
    ]

    if cfg.get("tune"):
        cmd.insert(cmd.index("-pix_fmt"), "-tune")
        cmd.insert(cmd.index("-pix_fmt"), cfg["tune"])

    if cfg.get("audio_bitrate"):
        cmd.insert(cmd.index("-movflags"), "-b:a")
        cmd.insert(cmd.index("-movflags"), cfg["audio_bitrate"])

    print("[VIDEO] Ejecutando FFmpeg para agregar intro...")
    subprocess.run(cmd, check=True)

    print(f"[VIDEO] OK: Intro agregada: {salida_fs}")
    return salida_fs


def render_story_clip(audio_path, image_path, carpeta_salida, title_text=None):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[CLIP] Audio no encontrado: {audio_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[CLIP] Imagen no encontrada: {image_path}")

    ffmpeg = _pick_ffmpeg()
    os.makedirs(carpeta_salida, exist_ok=True)

    nombre = _slugify_title(title_text) if title_text else "clip"
    salida_fs = os.path.join(carpeta_salida, f"{nombre}.mp4")

    cfg = _encoding_cfg()
    fps = cfg["fps"]

                                                                                        
    filter_complex = (
        f"[0:v]{_scale_filter(768)},pad=768:768:(768-iw)/2:(768-ih)/2,format=yuv420p,setsar=1,"
        f"tpad=stop_mode=clone:stop_duration=36000,trim=duration=36000,settb=AVTB,setpts=PTS-STARTPTS,fps={fps}[v0]"
    )

    cmd = [
        ffmpeg, "-y", "-nostdin",
        "-i", _ffmpeg_path(image_path),
        "-i", _ffmpeg_path(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[v0]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", cfg["preset"],
        "-crf", cfg["crf"],
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        "-r", fps,
        "-ar", "48000",
        "-movflags", "+faststart",
        _ffmpeg_path(salida_fs),
    ]

    if cfg.get("tune"):
        cmd.insert(cmd.index("-pix_fmt"), "-tune")
        cmd.insert(cmd.index("-pix_fmt"), cfg["tune"])

    if cfg.get("audio_bitrate"):
        cmd.insert(cmd.index("-movflags"), "-b:a")
        cmd.insert(cmd.index("-movflags"), cfg["audio_bitrate"])

    print(f"[CLIP] Renderizando corto: {salida_fs}")
    subprocess.run(cmd, check=True)
    print(f"[CLIP] OK: Generado: {salida_fs}")
    return salida_fs
