## Funcionamiento

- Genera una historia a partir de un tema
- Genera/descarga imágenes para acompañar la historia
- Genera audio con TTS (Edge neural o SAPI5 offline)

## Cómo correr

Requisitos:

- Python 3.11
- FFmpeg (si no lo tenés en PATH, el proyecto usa el binario de `imageio-ffmpeg` al instalar dependencias)

Ejecutar:

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Te va a pedir opciones por consola (género, cantidad, etc.).
