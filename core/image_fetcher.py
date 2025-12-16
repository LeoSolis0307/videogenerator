import requests
import os
import random
import time

PROMPTS = [
    "dark cinematic atmosphere",
    "mystery horror night fog",
    "creepy forest realistic",
]

def descargar_imagenes(carpeta, cantidad):
    rutas = []
    for i in range(cantidad):
        prompt = random.choice(PROMPTS) + str(random.randint(1,99999))
        url = f"https://image.pollinations.ai/prompt/{prompt}"
        ruta = os.path.join(carpeta, f"img_{i}.jpg")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(ruta, "wb") as f:
                    f.write(r.content)
                rutas.append(ruta)
        except:
            pass
        time.sleep(1)
    return rutas
