import os
import random
import time

from PIL import Image
import requests

PROMPTS = [
    "dark cinematic atmosphere",
    "creepy forest fog",
    "abandoned building night",
    "rainy city noir",
    "mysterious shadows",
    "cinematic thriller lighting",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*",
    "Referer": "https://pollinations.ai/"
}

TIMEOUT = 15
REINTENTOS = 3


def _recortar_abajo(path: str, porcentaje: float = 0.20) -> None:
    try:
        with Image.open(path) as img:
            w, h = img.size
            corte = int(h * porcentaje)
            nuevo_h = max(1, h - corte)
            recortada = img.crop((0, 0, w, nuevo_h))
            recortada.save(path)
    except Exception:
        print(f"[IMG] ⚠️ No se pudo recortar {path}")


def descargar_imagenes(carpeta, cantidad):
    print("[IMG] Descargando imágenes...")
    rutas = []

    session = requests.Session()
    session.headers.update(HEADERS)

    for i in range(cantidad):
        prompt = random.choice(PROMPTS)
        seed = random.randint(1, 9999999)

        url = (
            "https://image.pollinations.ai/prompt/"
            f"{prompt.replace(' ', '%20')}?seed={seed}"
        )

        ruta = os.path.join(carpeta, f"img_{i}.jpg")
        print(f"[IMG] Imagen {i+1}/{cantidad}")

        exito = False
        for intento in range(REINTENTOS):
            try:
                r = session.get(url, timeout=TIMEOUT)
                if r.status_code == 200 and len(r.content) > 15_000:
                    with open(ruta, "wb") as f:
                        f.write(r.content)
                    _recortar_abajo(ruta)
                    rutas.append(ruta)
                    exito = True
                    break
            except Exception:
                print(f"[IMG]   intento {intento+1}/{REINTENTOS} falló")

            time.sleep(2 + intento)                      

        if not exito:
            print("[IMG]   usando imagen de respaldo")
            backup = os.path.join(carpeta, f"img_fallback_{i}.jpg")
            if os.path.exists("fallback.jpg"):
                with open("fallback.jpg", "rb") as src, open(backup, "wb") as dst:
                    dst.write(src.read())
                _recortar_abajo(backup)
                rutas.append(backup)

        time.sleep(3)                            

    print(f"[IMG] {len(rutas)} imágenes listas")
    return rutas
