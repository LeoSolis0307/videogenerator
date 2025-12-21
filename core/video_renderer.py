import json
import math
import os
import random
import re
import subprocess
import wave
import shutil

import imageio_ffmpeg

DEFAULT_INTRO_PATH = r"C:\Users\Leonardo\Downloads\video\intro.mp4"
MIN_VIDEO_SEC = 15 * 60
MAX_VIDEO_SEC = 30 * 60
DEFAULT_VIDEOS_DIR = r"C:\Users\Leonardo\Downloads\video\media"
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


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
        print(f"[VIDEO] ⚠️ No se pudo leer duración. Fragmento de salida:\n{out[:4000]}")

    return 0.0


def audio_duration_seconds(path: str) -> float:
    \
    return _audio_duration_seconds(path)


def _debug_dump_duration(path: str):
    \
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
    \
\
\
\
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
    \
    ruta = _pick_video_file(videos_dir)
    if not ruta:
        return None, 0.0
    dur = _audio_duration_seconds(ruta)
    print(f"[VIDEO] Duración detectada para base: {dur:.2f}s")
    if dur <= 0:
        _debug_dump_duration(ruta)
    return ruta, dur


def _calc_speed_and_padding(video_dur: float, audio_dur: float, *, max_speed: float = 2.5) -> tuple[float, float]:
    \
\
\
\
\
\
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
    \
\
\
\
\
\
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
    filtros.append("scale=768:768:force_original_aspect_ratio=decrease")
    filtros.append("pad=768:768:(768-iw)/2:(768-ih)/2")
    filtros.append("setsar=1")
    filtros.append("fps=25")
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
        "h264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        _ffmpeg_path(salida),
    ]

    print(f"[VIDEO] Base: {video_abs} (dur ~{dur_video:.1f}s) | Audio ~{dur_audio:.1f}s")
    if speed != 1.0:
        print(f"[VIDEO] Acelerando video x{speed:.3f}")
    if pad_sec > 0.05:
        print(f"[VIDEO] Extendiendo último frame {pad_sec:.2f}s para empatar audio")

    subprocess.run(cmd, check=True)
    print(f"[VIDEO] ✅ Video final renderizado: {salida}")

                                                                                        
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
        print(f"[VIDEO] ⚠️ No se pudo renombrar el video usado: {e}")

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
    \
\
\
\
\
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

    for img in imagenes:
        if not os.path.exists(img):
            raise FileNotFoundError(f"[VIDEO] No se encontró la imagen: {img}")

    dur_list = list(durations or [])
    if dur_list and len(dur_list) < len(imagenes):
        dur_list.extend([dur_list[-1]] * (len(imagenes) - len(dur_list)))

    if tiempo_img is None and not dur_list:
        dur_audio = max(1.0, _audio_duration_seconds(audio_abs))
        tiempo_img = max(1, math.ceil(dur_audio / len(imagenes)))

                                                                                          
                                                                                                         
    inputs: list[str] = [ffmpeg, "-y"]
    filter_parts: list[str] = []
    vlabels: list[str] = []

    for idx, img in enumerate(imagenes):
        dur_val = max(0.5, float(dur_list[idx])) if dur_list else max(0.5, float(tiempo_img))
        inputs.extend(["-loop", "1", "-t", str(dur_val), "-i", _ffmpeg_path(img)])

        vlabel = f"v{idx}"
        vlabels.append(f"[{vlabel}]")
        filter_parts.append(
            f"[{idx}:v]fps=25,scale=768:768:force_original_aspect_ratio=decrease,"
            f"pad=768:768:(768-iw)/2:(768-ih)/2,setsar=1,format=yuv420p,setpts=PTS-STARTPTS[{vlabel}]"
        )

                             
    audio_input_index = len(imagenes)
    inputs.extend(["-i", _ffmpeg_path(audio_abs)])

    n = len(imagenes)
    concat_in = "".join(vlabels)
    filter_complex = ";".join(filter_parts) + f";{concat_in}concat=n={n}:v=1:a=0[v]"

    salida = os.path.join(carpeta_abs, "Video_Final.mp4")

    cmd = (
        inputs
        + [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            f"{audio_input_index}:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "25",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            _ffmpeg_path(salida),
        ]
    )

    print("[VIDEO] Ejecutando FFmpeg...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        out = (res.stdout or "")[:4000]
        err = (res.stderr or "")[:4000]
        if out:
            print(f"[VIDEO] ffmpeg stdout:\n{out}")
        if err:
            print(f"[VIDEO] ffmpeg stderr:\n{err}")
        res.check_returncode()

    print(f"[VIDEO] ✅ Video renderizado: {salida}")
    return salida


def _slugify_title(title: str | None) -> str:
    if not title:
        return "video"

    cleaned = re.sub(r"[\r\n]+", " ", title).strip()
    if not cleaned:
        return "video"

    words = cleaned.split()
    shortened = " ".join(words[:12])[:90].strip()
    shortened = re.sub(r"\s+", " ", shortened)
    safe = re.sub(r"[^\w\s-]", "", shortened)
    safe = safe.replace(" ", "_").strip("_- ")
    return safe or "video"


def append_intro_to_video(video_final, intro_path=DEFAULT_INTRO_PATH, output_path=None, title_text=None):
    print("[VIDEO] Preparando concatenación con intro...")

    video_fs = os.path.abspath(video_final)
    intro_fs = os.path.abspath(intro_path)

    if not os.path.exists(video_fs):
        raise FileNotFoundError(f"[VIDEO] No se encontró el video final: {video_fs}")

    if not os.path.exists(intro_fs):
        raise FileNotFoundError(f"[VIDEO] No se encontró el intro: {intro_fs}")

    ffmpeg = _pick_ffmpeg()
    base, ext = os.path.splitext(os.path.basename(video_fs))
    carpeta_salida = os.path.dirname(video_fs)
    nombre_final = _slugify_title(title_text) if title_text else base + "_con_intro"
    salida_fs = output_path or os.path.join(carpeta_salida, f"{nombre_final}{ext}")

    video_abs = _ffmpeg_path(video_fs)
    intro_abs = _ffmpeg_path(intro_fs)
    salida = _ffmpeg_path(salida_fs)

    filter_complex = (
        "[0:v]fps=25,scale=768:768:force_original_aspect_ratio=decrease,"
        "pad=768:768:(768-iw)/2:(768-ih)/2,setsar=1,setpts=PTS-STARTPTS[v0];"
        "[1:v]fps=25,scale=768:768:force_original_aspect_ratio=decrease,"
        "pad=768:768:(768-iw)/2:(768-ih)/2,setsar=1,setpts=PTS-STARTPTS[v1];"
        "[0:a]aresample=48000,asetpts=PTS-STARTPTS[a0];"
        "[1:a]aresample=48000,asetpts=PTS-STARTPTS[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
    )

    cmd = [
        ffmpeg, "-y",
        "-i", video_abs,
        "-i", intro_abs,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        salida
    ]

    print("[VIDEO] Ejecutando FFmpeg para agregar intro...")
    subprocess.run(cmd, check=True)

    print(f"[VIDEO] ✅ Intro agregada: {salida_fs}")
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

                                                                                              
    filter_complex = (
        "[0:v]fps=25,scale=768:768:force_original_aspect_ratio=decrease,"
        "pad=768:768:(768-iw)/2:(768-ih)/2,setsar=1[v0]"
    )

    cmd = [
        ffmpeg, "-y",
        "-loop", "1",
        "-i", _ffmpeg_path(image_path),
        "-i", _ffmpeg_path(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[v0]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        "-r", "25",
        "-movflags", "+faststart",
        _ffmpeg_path(salida_fs),
    ]

    print(f"[CLIP] Renderizando corto: {salida_fs}")
    subprocess.run(cmd, check=True)
    print(f"[CLIP] ✅ Generado: {salida_fs}")
    return salida_fs
