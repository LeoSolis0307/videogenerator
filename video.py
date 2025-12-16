import requests
import random
import os
import asyncio
import edge_tts
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips
from deep_translator import GoogleTranslator
import math
import time
import shutil
from datetime import datetime

# --- CONFIGURACIÓN ---
URL_SUBREDDIT = "https://www.reddit.com/r/AskReddit/top/?t=week"
VOZ_IA = "es-MX-JorgeNeural"
VELOCIDAD_VOZ = "-15%"  
DURACION_OBJETIVO_MINUTOS = 30  
CANTIDAD_IMAGENES = 10 
TIEMPO_POR_IMAGEN = 10 

# --- GESTIÓN DE CARPETAS ---
def crear_carpeta_proyecto():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_carpeta = f"Video_{timestamp}"
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
    print(f"📁 Carpeta del proyecto creada: {nombre_carpeta}")
    return nombre_carpeta

# --- GESTIÓN DE HISTORIAL ---
ARCHIVO_HISTORIAL = "historial_global.txt"
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
    try:
        if len(texto) > 4500: texto = texto[:4500]
        return GoogleTranslator(source='auto', target='es').translate(texto)
    except:
        return texto

# --- 1. RECOLECTOR (CORREGIDO) ---
def obtener_contenido_reddit(carpeta_destino, historial_usados):
    # HEADER ACTUALIZADO: Mucho más realista para evitar bloqueo
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    
    print("🕵️ Buscando post viral...")
    try:
        # Paso 1: Obtener post
        url_base = URL_SUBREDDIT.split('?')[0]
        if url_base.endswith("/"): url_base = url_base[:-1]
        
        # Añadimos raw_json=1 para evitar problemas de formato
        resp = requests.get(f"{url_base}/.json?t=week&limit=10&raw_json=1", headers=headers)
        
        if resp.status_code == 429:
            print("❌ ERROR 429: Reddit nos bloqueó por hacer muchas peticiones. Espera unos minutos.")
            return None, None, None
        
        if resp.status_code != 200:
            print(f"❌ Error conectando a Reddit: Código {resp.status_code}")
            return None, None, None
            
        posts = resp.json()['data']['children']
        post_elegido = None
        for p in posts:
            # Buscamos posts con muchos comentarios y que no sean anuncios
            if not p['data']['stickied'] and p['data']['num_comments'] > 50:
                post_elegido = p['data']
                break
        
        if not post_elegido: 
            print("❌ No se encontró un post adecuado.")
            return None, None, None

        titulo_en = post_elegido['title']
        url_comments = f"https://www.reddit.com{post_elegido['permalink']}"
        if url_comments.endswith("/"): url_comments = url_comments[:-1]
        
        print(f"✅ Post encontrado: {titulo_en}")
        
        # Paso 2: Obtener comentarios
        # Añadimos un pequeño delay para no parecer robot
        time.sleep(2) 
        resp_comm = requests.get(f"{url_comments}.json?raw_json=1", headers=headers)
        
        if resp_comm.status_code != 200:
            print(f"❌ Error al entrar al post: Código {resp_comm.status_code}")
            return None, None, None

        comentarios = resp_comm.json()[1]['data']['children']
        random.shuffle(comentarios)
        
        textos_en = []
        textos_es = []
        ids_nuevos = []
        total_chars = 0
        META = DURACION_OBJETIVO_MINUTOS * 60 * 14
        
        # Título
        titulo_es = traducir_a_espanol(titulo_en)
        textos_en.append(f"Title: {titulo_en}")
        textos_es.append(f"La pregunta es: {titulo_es}")
        
        print("📝 Recopilando y traduciendo historias...")
        for c in comentarios:
            if c['kind'] == 't1' and 'body' in c['data']:
                body = c['data']['body']
                cid = c['data']['id']
                
                if cid not in historial_usados and len(body) > 200 and "[deleted]" not in body:
                    print(f"   - Procesando historia ID {cid}...")
                    textos_en.append(f"\n--- Story ID {cid} ---\n{body}")
                    
                    trad = traducir_a_espanol(body)
                    textos_es.append(trad)
                    
                    ids_nuevos.append(cid)
                    total_chars += len(trad)
                    
                    if total_chars >= META: break
        
        # Guardar txts
        with open(os.path.join(carpeta_destino, "1-ingles.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(textos_en))
            
        with open(os.path.join(carpeta_destino, "1-espanol.txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(textos_es))
            
        return textos_es, ids_nuevos

    except Exception as e:
        print(f"❌ Error crítico en Reddit: {e}")
        # Debug: Imprime qué devolvió Reddit si falla
        # print(resp.text) 
        return None, None, None

# --- 2. AUDIO ---
async def generar_audio(carpeta, textos):
    archivos = []
    print("🔊 Generando audios...")
    for i, txt in enumerate(textos):
        ruta = os.path.join(carpeta, f"audio_part_{i}.mp3")
        try:
            communicate = edge_tts.Communicate(txt, VOZ_IA, rate=VELOCIDAD_VOZ)
            await communicate.save(ruta)
            archivos.append(ruta)
        except Exception as e:
            print(f"Error audio {i}: {e}")
    return archivos

# --- 3. IMÁGENES ---
def descargar_imagenes(carpeta, cantidad):
    rutas = []
    prompts = [
        "cinematic dark mystery atmosphere 4k", "creepy forest night fog realistic",
        "abandoned building interior dark lighting", "rainy window night city view",
        "shadowy silhouette horror thriller art", "vintage old photo mystery style",
        "surreal dreamscape dark fantasy", "noir detective style rainy street",
        "gothic architecture night fog", "abstract fear anxiety texture"
    ]
    
    print("🎨 Descargando imágenes...")
    for i in range(cantidad):
        ruta = os.path.join(carpeta, f"img_{i}.jpg")
        # Forzar variedad con números aleatorios grandes
        prompt = prompts[i % len(prompts)] + f" {random.randint(1,999999)}"
        url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
        
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(ruta, "wb") as f:
                    f.write(resp.content)
                rutas.append(ruta)
                print(f"   - Imagen {i+1} guardada.")
            time.sleep(1.5) 
        except:
            pass
            
    if not rutas and os.path.exists("fondo.jpg"):
        shutil.copy("fondo.jpg", os.path.join(carpeta, "fondo_backup.jpg"))
        rutas.append(os.path.join(carpeta, "fondo_backup.jpg"))
        
    return rutas

# --- 4. VIDEO ---
def armar_video():
    carpeta = crear_carpeta_proyecto()
    historial = cargar_historial()
    
    # 1. Contenido
    textos_es, ids = obtener_contenido_reddit(carpeta, historial)
    if not textos_es: 
        print("❌ No se pudo obtener contenido. Abortando.")
        return
    
    # 2. Audio
    audios = asyncio.run(generar_audio(carpeta, textos_es))
    if not audios: return
    
    # 3. Imágenes
    imagenes = descargar_imagenes(carpeta, CANTIDAD_IMAGENES)
    if not imagenes: 
        print("❌ Sin imágenes.")
        return

    print("--- 🎬 Renderizando ---")
    try:
        # Unir Audio
        clips_aud = [AudioFileClip(a) for a in audios]
        audio_final = concatenate_audioclips(clips_aud)
        duracion = audio_final.duration
        audio_path = os.path.join(carpeta, "audio_completo.mp3")
        audio_final.write_audiofile(audio_path)
        print(f"⏱️ Duración: {duracion/60:.1f} min")

        # Crear Video
        clips_vid = []
        # Calcular clips necesarios
        num_clips = int(math.ceil(duracion / TIEMPO_POR_IMAGEN))
        
        print(f"   - Creando {num_clips} segmentos de video...")
        
        for i in range(num_clips):
            img_path = imagenes[i % len(imagenes)]
            
            # Recargar imagen explícitamente para evitar cache de MoviePy
            clip = ImageClip(img_path).set_duration(TIEMPO_POR_IMAGEN).set_fps(24)
            clip = clip.crossfadein(1.0)
            clips_vid.append(clip)
            
        video = concatenate_videoclips(clips_vid, method="compose")
        video = video.set_duration(duracion)
        video = video.set_audio(AudioFileClip(audio_path))
        
        salida = os.path.join(carpeta, "Video_Final.mp4")
        video.write_videofile(salida, fps=24, threads=4, codec="libx264")
        
        print(f"✅ ¡LISTO! Todo guardado en: {carpeta}")
        guardar_lista_historial(ids)
        
        video.close()
        audio_final.close()
        # Cerrar clips de audio
        for c in clips_aud: c.close()

    except Exception as e:
        print(f"❌ Error fatal en renderizado: {e}")

if __name__ == "__main__":
    armar_video()