import subprocess
import os
import imageio_ffmpeg

def render_video_ffmpeg(imagenes, audio, carpeta, tiempo_img=10):
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

    def _ffmpeg_path(path: str) -> str:
                                                                         
        return os.path.abspath(path).replace("\\", "/")

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
