from core import reddit_scraper
from core import tts
from core import image_downloader
from core import text_processor
from core.video_renderer import render_video_ffmpeg
from utils.fs import crear_carpeta_proyecto


                                                   
                                                      
VOZ = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0"
VELOCIDAD = "-17%"

def main():
    print("[MAIN] Iniciando proceso")

                                    
    carpeta = crear_carpeta_proyecto()

                      
    post = reddit_scraper.obtener_post()
    if not post:
        print("[MAIN] No se pudo obtener post")
        return

                             
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

                              
    textos_es = text_processor.traducir_lista(textos_en)

    print("[DEBUG] Primer texto que irá al TTS:")
    print(textos_es[0][:200])

             
                                                                                 
    audios = tts.generar_audios(textos_es, carpeta, voz=VOZ, velocidad=VELOCIDAD)


    if not audios:
        print("[MAIN] No se generaron audios")
        return

                  
    imagenes = image_downloader.descargar_imagenes(carpeta, 10)

    if not imagenes:
        print("[MAIN] No se descargaron imágenes")
        return

                                                           
    render_video_ffmpeg(imagenes, audios[0], carpeta)

    print("[MAIN] ✅ Proceso finalizado")

if __name__ == "__main__":
    main()
