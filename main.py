from core import reddit_scraper
from core import tts
from core import image_downloader
from core import text_processor
from core.video_renderer import render_video_ffmpeg
from utils.fs import crear_carpeta_proyecto


VOZ = "es-MX-JorgeNeural"
VELOCIDAD = "-15%"

def main():
    print("[MAIN] Iniciando proceso")

    # 1️⃣ Crear carpeta del proyecto
    carpeta = crear_carpeta_proyecto()

    # 2️⃣ Obtener post
    post = reddit_scraper.obtener_post()
    if not post:
        print("[MAIN] No se pudo obtener post")
        return

    # 3️⃣ Obtener comentarios
    comentarios = reddit_scraper.obtener_comentarios(post["permalink"])

    textos_en = [
        c["data"]["body"]
        for c in comentarios
        if c["kind"] == "t1"
        and len(c["data"].get("body", "")) > 200
        and "[deleted]" not in c["data"]["body"]
    ][:30]

    if not textos_en:
        print("[MAIN] No hay textos suficientes")
        return

    print(f"[MAIN] {len(textos_en)} textos obtenidos")

    # 🔥 4️⃣ TRADUCCIÓN (CLAVE)
    textos_es = text_processor.traducir_lista(textos_en)

    print("[DEBUG] Primer texto que irá al TTS:")
    print(textos_es[0][:200])

    # 5️⃣ TTS
    audios = tts.generar_audios(textos_es, carpeta)


    if not audios:
        print("[MAIN] No se generaron audios")
        return

    # 6️⃣ Imágenes
    imagenes = image_downloader.descargar_imagenes(carpeta, 10)

    if not imagenes:
        print("[MAIN] No se descargaron imágenes")
        return

    # 7️⃣ Render de video (usa el primer audio como prueba)
    render_video_ffmpeg(imagenes, audios[0], carpeta)

    print("[MAIN] ✅ Proceso finalizado")

if __name__ == "__main__":
    main()
