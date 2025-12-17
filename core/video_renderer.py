import json
import math
import os
import re
import subprocess
import wave

import imageio_ffmpeg

DEFAULT_INTRO_PATH = r"C:\Users\Leonardo\Downloads\video\intro.mp4"
MIN_VIDEO_SEC = 15 * 60
MAX_VIDEO_SEC = 30 * 60


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
    try:
        import subprocess, json

        ffprobe = imageio_ffmpeg.get_ffmpeg_exe().replace("ffmpeg", "ffprobe")
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout or "{}")
        dur = float(data.get("format", {}).get("duration", 0.0))
        return dur
    except Exception:
        return 0.0


def _normalize_audio_to_wav(src: str, dst: str, *, rate: int = 48000, channels: int = 1) -> str:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
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


def combine_audios_with_silence(audios, carpeta, gap_seconds=4):
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

        max_frames = int(MAX_VIDEO_SEC * rate)
        min_frames = int(MIN_VIDEO_SEC * rate)
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

                if total_frames + frames_len > max_frames:
                    needed = max_frames - total_frames
                    if needed > 0:
                        out_w.writeframes(frames[: needed * channels * sampwidth])
                        total_frames += needed
                    break

                out_w.writeframes(frames)
                total_frames += frames_len

                is_last = i == len(norm_paths) - 1
                if not is_last and total_frames < max_frames:
                    add_gap = min(gap_frames, max_frames - total_frames)
                    out_w.writeframes(b"\x00" * add_gap * channels * sampwidth)
                    total_frames += add_gap

            if total_frames < min_frames:
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


def render_video_ffmpeg(imagenes, audio, carpeta, tiempo_img=None):
    print("[VIDEO] Preparando render con FFmpeg (imageio)...")

    if not imagenes:
        raise ValueError("[VIDEO] Lista de imágenes vacía")

    if not carpeta:
        raise ValueError("[VIDEO] Carpeta de salida no válida")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[VIDEO] FFmpeg encontrado en: {ffmpeg}")

    carpeta_abs = os.path.abspath(carpeta)
    img_list = os.path.join(carpeta_abs, "imgs.txt")
    audio_abs = os.path.abspath(audio)

    if not os.path.exists(audio_abs):
        raise FileNotFoundError(f"[VIDEO] No se encontró el audio: {audio_abs}")

    for img in imagenes:
        if not os.path.exists(img):
            raise FileNotFoundError(f"[VIDEO] No se encontró la imagen: {img}")

    if tiempo_img is None:
        dur_audio = max(1.0, _audio_duration_seconds(audio))
        tiempo_img = max(1, math.ceil(dur_audio / len(imagenes)))

    with open(img_list, "w", encoding="utf-8") as f:
        for img in imagenes:
            img_norm = _ffmpeg_path(img)
            f.write(f"file '{img_norm}'\n")
            f.write(f"duration {tiempo_img}\n")

                                                               
        f.write(f"file '{_ffmpeg_path(imagenes[-1])}'\n")

    salida = os.path.join(carpeta_abs, "Video_Final.mp4")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", img_list,
        "-i", audio_abs,
        "-c:v", "h264",                                         
        "-pix_fmt", "yuv420p",
        "-shortest",
        salida
    ]

    print("[VIDEO] Ejecutando FFmpeg...")
    subprocess.run(cmd, check=True)

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

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
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
        "-c:v", "h264",
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

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(carpeta_salida, exist_ok=True)

    nombre = _slugify_title(title_text) if title_text else "clip"
    salida_fs = os.path.join(carpeta_salida, f"{nombre}.mp4")

    cmd = [
        ffmpeg, "-y",
        "-loop", "1",
        "-i", _ffmpeg_path(image_path),
        "-i", _ffmpeg_path(audio_path),
        "-c:v", "h264",
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
