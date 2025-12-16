from moviepy.editor import *
import math

def render_video(audios, imagenes, carpeta, tiempo_img=10):
    audio = concatenate_audioclips([AudioFileClip(a) for a in audios])
    dur = audio.duration
    clips = []
    for i in range(math.ceil(dur / tiempo_img)):
        img = imagenes[i % len(imagenes)]
        clip = ImageClip(img).set_duration(tiempo_img).set_fps(24)
        clips.append(clip)
    video = concatenate_videoclips(clips).set_audio(audio)
    salida = f"{carpeta}/Video_Final.mp4"
    video.write_videofile(salida, fps=24)
