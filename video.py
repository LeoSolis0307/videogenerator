import requests
import random
import os
import asyncio
import edge_tts
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips
from deep_translator import GoogleTranslator
import math
import time

# --- CONFIGURACIÓN ---
# Puedes usar un subreddit en inglés, el bot lo traducirá.
# Ejemplo: 'r/AskReddit', 'r/nosleep', 'r/entitledparents'
URL_SUBREDDIT = "https://www.reddit.com/r/AskReddit/top/?t=week"

VOZ_IA = "es-MX-JorgeNeural" 
VELOCIDAD_VOZ = "-20%" # Reduce la velocidad un 20%
ARCHIVO_HISTORIAL = "historial_global.txt"
DURACION_OBJETIVO_MINUTOS = 5 # Pongo 5 min para probar rápido. Pon 30 para media hora.
CANTIDAD_IMAGENES = 5
TIEMPO_POR_IMAGEN = 15 # Cada imagen se muestra 15 segundos antes de cambiar a la siguiente

# --- GESTIÓN DE HISTORIAL ---
def cargar_historial():
    if not os.path.exists(ARCHIVO_HISTORIAL):
        open(ARCHIVO_HISTORIAL, 'w').close()
        return set()
    with open(ARCHIVO_HISTORIAL, "r") as f:
        return set(line.strip() for line in f if line.strip())

def guardar_lista_historial(ids_nuevos):
    with open(ARCHIVO_HISTORIAL, "a") as f:
        for i in ids_nuevos:
            f.write(f"{i}\n")

# --- TRADUCTOR ---
def traducir_a_espanol(texto):
    """Traduce texto de cualquier idioma al español usando Google Translate (Gratis)"""
    try:
        # Dividimos si es muy largo para no saturar el traductor (máx 5000 chars)
        if len(texto) > 4500:
            texto = texto[:4500] 
        
        traductor = GoogleTranslator(source='auto', target='es')
        traducido = traductor.translate(texto)
        return traducido
    except Exception as e:
        print(f"⚠️ Error traduciendo: {e}. Usando texto original.")
        return texto

# --- 1. RECOLECTOR DE HISTORIAS ---
def obtener_pack_historias(historial_usados):
    """
    Entra al TOP semanal, elige un post automáticamente y saca sus comentarios.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    # 1. Obtener lista de posts populares (formato JSON)
    # Limpiamos la URL para evitar errores
    url_base = URL_SUBREDDIT.split('?')[0] # Quita lo que esté después del ?
    if url_base.endswith("/"): url_base = url_base[:-1]
    
    # URL para sacar los posts del subreddit
    url_api_subreddit = f"{url_base}/.json?t=week&limit=5"

    print(f"🕵️ Buscando el mejor post de la semana en AskReddit...")

    try:
        # PASO A: Buscar un post válido en el índice
        resp_sub = requests.get(url_api_subreddit, headers=headers)
        if resp_sub.status_code != 200:
            print(f"❌ Error conectando al Subreddit: {resp_sub.status_code}")
            return [], []
            
        posts_data = resp_sub.json()['data']['children']
        
        url_post_elegido = None
        titulo_post = ""
        
        # Buscamos un post que no sea pegajoso (sticky) y tenga comentarios
        for p in posts_data:
            data = p['data']
            if not data['stickied'] and data['num_comments'] > 10:
                url_post_elegido = f"https://www.reddit.com{data['permalink']}"
                titulo_post = data['title']
                break
        
        if not url_post_elegido:
            print("❌ No se encontró ningún post bueno hoy.")
            return [], []
            
        print(f"✅ Post Elegido: {titulo_post}")
        print(f"🔗 Link: {url_post_elegido}")

        # PASO B: Entrar a ESE post y sacar los comentarios (La lógica anterior)
        url_json_post = url_post_elegido[:-1] + ".json"
        respuesta = requests.get(url_json_post, headers=headers)
        
        datos = respuesta.json()
        comentarios = datos[1]['data']['children']
        
        # META DE CARACTERES
        META_CARACTERES = DURACION_OBJETIVO_MINUTOS * 60 * 12
        historias_seleccionadas = []
        total_chars = 0
        ids_nuevos = []
        
        # Añadimos el título del post como introducción
        titulo_es = traducir_a_espanol(titulo_post)
        historias_seleccionadas.append(f"La pregunta de hoy es: {titulo_es}")

        # Mezclar comentarios
        random.shuffle(comentarios)

        for c in comentarios:
            if c['kind'] == 't1' and 'body' in c['data']:
                texto_original = c['data']['body']
                c_id = c['data']['id']
                
                # Filtros
                if (c_id not in historial_usados and 
                    c_id not in ids_nuevos and 
                    len(texto_original) > 300 and "[deleted]" not in texto_original):
                    
                    print(f"   Using ID: {c_id} ({len(texto_original)} chars). Traduciendo...")
                    
                    # Traducción
                    texto_espanol = traducir_a_espanol(texto_original)
                    
                    historias_seleccionadas.append(texto_espanol)
                    ids_nuevos.append(c_id)
                    total_chars += len(texto_espanol)
                    
                    print(f"   ✅ Traducido. Total: {total_chars}/{META_CARACTERES}")
                    
                    if total_chars >= META_CARACTERES:
                        break
            
        return historias_seleccionadas, ids_nuevos

    except Exception as e:
        print(f"Error crítico: {e}")
        return [], []

# --- 2. GENERADOR DE AUDIO (LENTO) ---
async def generar_audio_lento(lista_textos):
    archivos_audio = []
    print(f"🔊 Generando {len(lista_textos)} audios con IA (Velocidad {VELOCIDAD_VOZ})...")
    
    for i, texto in enumerate(lista_textos):
        archivo = f"temp_part_{i}.mp3"
        try:
            # Aquí aplicamos la velocidad lenta (rate)
            communicate = edge_tts.Communicate(texto, VOZ_IA, rate=VELOCIDAD_VOZ)
            await communicate.save(archivo)
            archivos_audio.append(archivo)
        except Exception as e:
            print(f"   Error generando audio {i}: {e}")
            
    return archivos_audio

# --- 3. GENERADOR DE IMÁGENES ---
def generar_imagenes_ia(cantidad):
    lista_imagenes = []
    print(f"🎨 Generando {cantidad} imágenes únicas...")
    
    prompts = [
        "cinematic dark mystery story background 4k",
        "atmospheric horror rainy night window view",
        "creepy liminal space corridor realistic",
        "dramatic foggy street night lighting",
        "abstract fear dark psychological thriller art"
    ]
    
    for i in range(cantidad):
        nombre = f"fondo_ia_{i}.jpg"
        # Variamos el prompt cada vez
        prompt = prompts[i % len(prompts)] + f" {random.randint(1,5000)}"
        url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
        
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(nombre, 'wb') as f:
                    f.write(resp.content)
                lista_imagenes.append(nombre)
                print(f"   - Imagen {i+1} lista.")
            time.sleep(2)
        except:
            pass
            
    return lista_imagenes

# --- 4. ENSAMBLAJE ROBUSTO ---
def ensamblar_video():
    # A. Datos y Traducción
    historial = cargar_historial()
    textos, nuevos_ids = obtener_pack_historias(historial)
    
    if not textos:
        print("❌ No se encontraron historias válidas.")
        return

    # B. Audio
    archivos_audio_temp = asyncio.run(generar_audio_lento(textos))
    if not archivos_audio_temp: return

    # C. Imágenes
    imagenes_files = generar_imagenes_ia(CANTIDAD_IMAGENES)
    # Si falla la IA, usa una imagen local por seguridad
    if not imagenes_files:
        if os.path.exists("fondo.jpg"):
            imagenes_files = ["fondo.jpg"]
        else:
            print("❌ No hay imágenes generadas ni fondo.jpg.")
            return

    print("--- 🎬 Renderizando Video Final ---")
    
    try:
        # 1. Unir Audios en uno solo gigante
        clips_audio = [AudioFileClip(f) for f in archivos_audio_temp]
        audio_final = concatenate_audioclips(clips_audio)
        duracion_total = audio_final.duration
        print(f"⏱️ Duración Total del Video: {duracion_total/60:.2f} minutos.")

        # 2. Crear el Bucle Visual (LA CORRECCIÓN)
        # Calculamos cuántos clips de imagen necesitamos para llenar el audio
        # Ejemplo: Audio 300s, Imagen dura 15s -> Necesitamos 20 clips
        cantidad_clips_visuales = int(math.ceil(duracion_total / TIEMPO_POR_IMAGEN))
        
        clips_video = []
        print(f"   - Creando {cantidad_clips_visuales} cambios de imagen...")
        
        for i in range(cantidad_clips_visuales):
            # Usamos el operador % para rotar: 0, 1, 2, 3, 4, 0, 1...
            img_actual = imagenes_files[i % len(imagenes_files)]
            
            # Creamos el clip de imagen
            clip = ImageClip(img_actual).set_duration(TIEMPO_POR_IMAGEN)
            clips_video.append(clip)

        # 3. Unificar video
        video_final = concatenate_videoclips(clips_video)
        video_con_audio = video_final.set_audio(audio_final)

        # 4. Exportar
        nombre_salida = f"video_reddit_{int(time.time())}.mp4"
        print(f"   Exportando a {nombre_salida}...")
        video_con_audio.write_videofile(nombre_salida, fps=24, verbose=False, logger=None)

        # 5. Limpiar temporales
        print("🧹 Limpiando archivos temporales...")
        for f in archivos_audio_temp + imagenes_files:
            try: os.remove(f)
            except: pass

        # 6. Guardar historial
        guardar_lista_historial(nuevos_ids)
        print(f"✅ ¡Video creado exitosamente! Guardado en: {nombre_salida}")

    except Exception as e:
        print(f"❌ Error al renderizar: {e}")

# --- MAIN ---
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 GENERADOR DE VIDEOS REDDIT -> ESPAÑOL")
    print("=" * 50)
    ensamblar_video()