import os
from datetime import datetime

                           
            
                           
BASE_OUTPUT_DIR = "output"
HISTORIAL_FILE = "storage/historial.txt"


                           
          
                           
def crear_carpeta_proyecto(prefix="Video") -> str:
    \
\
\
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = os.path.join(BASE_OUTPUT_DIR, f"{prefix}_{timestamp}")
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def asegurar_directorio(path: str):
    \
\
\
    os.makedirs(path, exist_ok=True)


                           
           
                           
def cargar_historial() -> set:
    \
\
\
    asegurar_directorio(os.path.dirname(HISTORIAL_FILE))
    if not os.path.exists(HISTORIAL_FILE):
        open(HISTORIAL_FILE, "w", encoding="utf-8").close()
        return set()

    with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def guardar_historial(ids):
    \
\
\
    if not ids:
        return
    asegurar_directorio(os.path.dirname(HISTORIAL_FILE))
    with open(HISTORIAL_FILE, "a", encoding="utf-8") as f:
        for i in ids:
            f.write(f"{i}\n")


                           
                    
                           
def guardar_texto(path: str, texto: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(texto)


def leer_texto(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
