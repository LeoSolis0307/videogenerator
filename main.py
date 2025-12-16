import asyncio
from core import reddit_scraper, text_processor, tts_engine, image_fetcher, video_renderer
from utils.fs import crear_carpeta, cargar_historial, guardar_historial

def main():
    carpeta = crear_carpeta()
    historial = cargar_historial()

    post = reddit_scraper.obtener_post()
    comments = reddit_scraper.obtener_comentarios(post["permalink"])

    textos, ids = text_processor.filtrar_comentarios(comments, historial)

    audios = asyncio.run(tts_engine.texto_a_audio(textos, carpeta))
    imagenes = image_fetcher.descargar_imagenes(carpeta, 10)

    video_renderer.render_video(audios, imagenes, carpeta)

    guardar_historial(ids)

if __name__ == "__main__":
    main()
