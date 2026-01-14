# Video Generator (historias + voz + clips)

Hice este proyecto para generar videos estilo TikTok/Reels de forma semi-automática: agarra un tema, arma una historia, genera imágenes, le mete voz (TTS) y lo renderiza en video.

No es perfecto (todavía), pero sí saca videos y me sirvió para practicar un montón de cosas: scraping, prompts, TTS, render con FFmpeg y algo de automatización.

## Qué hace

- Genera historias por tema (también puede usar temas desde un archivo)
- Descarga/genera imágenes para cada parte
- Genera audio con TTS (Edge neural o SAPI5 offline)
- Renderiza un video final con FFmpeg (con intro opcional)
- Guarda el output en carpetas bajo `output/`

## Requisitos rápidos

- Python 3.11
- FFmpeg instalado y en `PATH` (o que lo detecte el entorno)
- (Opcional) Ollama corriendo si usas generación local de texto/visión

## Cómo correr

1) Crear/activar venv e instalar deps (si todavía no lo hiciste)

2) Ejecutar:

```bash
python main.py
```

Te va a pedir opciones por consola (género, cantidad, etc.).

## Estructura del repo (lo importante)

- `main.py`: entrada principal
- `core/`: lógica fuerte (tts, generador de historia, renderer, etc.)
- `utils/`: helpers (archivos de temas, rutas, etc.)
- `tools/`: scripts de soporte
- `output/`: se generan los renders acá (está ignorado en git)

## Notas

- El repo **no** sube renders/videos generados al historial (eran gigas). Si quieres ver ejemplos, mejor los dejo como links o en releases.
- Si algo falla, normalmente es por FFmpeg/voices/credenciales/modelos (es lo primero que reviso).

## Idea de uso (mi flujo)

- Pongo una lista de temas
- Corro el script
- Reviso el output
- Si un video sale raro, ajusto el prompt o la voz y vuelvo a correr

---

Si quieres, puedo agregar un par de comandos “listos” tipo `run.bat` para Windows o una mini guía de troubleshooting.
