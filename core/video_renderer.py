import subprocess
import os
import imageio_ffmpeg

def render_video_ffmpeg(imagenes, audio, carpeta, tiempo_img=10):
    print("[VIDEO] Preparando render con FFmpeg (imageio)...")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[VIDEO] FFmpeg encontrado en: {ffmpeg}")

    img_list = os.path.join(carpeta, "imgs.txt")
    with open(img_list, "w", encoding="utf-8") as f:
        for img in imagenes:
            f.write(f"file '{img}'\n")
            f.write(f"duration {tiempo_img}\n")

                                 
        f.write(f"file '{imagenes[-1]}'\n")

    salida = os.path.join(carpeta, "Video_Final.mp4")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", img_list,
        "-i", audio,
        "-c:v", "h264",                                         
        "-pix_fmt", "yuv420p",
        "-shortest",
        salida
    ]

    print("[VIDEO] Ejecutando FFmpeg...")
    subprocess.run(cmd, check=True)

    print(f"[VIDEO] ✅ Video renderizado: {salida}")
