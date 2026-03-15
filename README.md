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

## Módulo nuevo: voces locales (texto -> audio)

Se agregó un módulo dedicado para generar audio local sin pipeline de video:

```bash
python -m core.local_tts --text "Hola, esto es una prueba" --out output/prueba.wav --engine kokoro
```

También acepta MP3 (convierte con ffmpeg):

```bash
python -m core.local_tts --text "Hola" --out output/prueba.mp3 --engine kokoro
```

Voces en español (Kokoro): `ef_dora`, `em_alex`, `em_santa`.

Ejemplo eligiendo voz:

```bash
python -m core.local_tts --text-file tmp_story_input.txt --out output/kokoro.wav --engine kokoro --voice ef_dora --lang es --speed 0.95
```

Notas:

- `kokoro-onnx` descarga automáticamente los archivos de modelo al primer uso en `storage/kokoro/`.
- La salida soportada es `.wav` o `.mp3`.

## Módulo nuevo: clonación de voz local (WAV/MP3/M4A -> voz clonada)

Se agregó un módulo para clonar una voz desde una referencia de audio y sintetizar texto con esa voz:

```bash
python -m core.voice_clone --ref ruta/mi_voz.wav --text "Hola, esto es una prueba" --out output/clone.wav --lang es
```

También funciona con referencia MP3/M4A y salida MP3:

```bash
python -m core.voice_clone --ref ruta/mi_voz.mp3 --text-file tmp_story_input.txt --out output/clone.mp3 --lang es
```

Para priorizar calidad de clonación (recorte de silencios + referencia de 30s):

```bash
python -m core.voice_clone --ref ruta/mi_voz.m4a --text-file tmp_story_input.txt --out output/clone_hq.wav --lang es --cpu --ref-max-sec 30 --ref-trim-silence --ref-pick-best
```

Notas:

- Modelo por defecto: `tts_models/multilingual/multi-dataset/xtts_v2`.
- El módulo intenta usar GPU automáticamente (`CUDA` o `DirectML`), y si no hay, usa CPU.
- Si quieres forzar CPU: `--cpu`.
- Requiere `ffmpeg` para normalizar la referencia y para salida MP3.
- Dependencias en archivo aparte: `requirements.voice_clone.txt`.
- En Windows, Coqui TTS suele requerir Python 3.10. Si usas Python 3.11, la instalación puede fallar por compatibilidad.

Instalación sugerida para clonación de voz:

```bash
py -3.10 -m venv .venv-voiceclone
.\.venv-voiceclone\Scripts\Activate.ps1
python -m pip install -r requirements.voice_clone.txt
```

Si durante la instalación aparece `Microsoft Visual C++ 14.0 or greater is required`, instala Build Tools y repite:

```powershell
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --silent --accept-package-agreements --accept-source-agreements --override "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

## Aceleración GPU (AMD RX 7600) para render de video

Ahora el render usa GPU por defecto (`VIDEO_CODEC=h264_amf`) y valida soporte real al iniciar render.
Si no hay encoder GPU disponible o falla la prueba de encode, el proceso se detiene con el motivo exacto.

Configuración recomendada para AMD:

```powershell
$env:VIDEO_CODEC="h264_amf"
$env:VIDEO_CODEC_ARGS="-quality quality -usage transcoding -rc cqp -qp_i 22 -qp_p 22"
python main.py
```

Lanzador único (PowerShell) para modo GPU estricto:

```powershell
.\tools\run_main_gpu.ps1
```

Notas rápidas:

- Esto acelera el **encode de video** (AMF). Filtros complejos de FFmpeg pueden seguir en CPU.
- ComfyUI es opcional y ahora viene desactivado por defecto (`COMFYUI_ENABLED=0`). Si quieres usarlo, habilítalo explícitamente con `COMFYUI_ENABLED=1`.
- Si necesitas permitir fallback CPU manualmente: `REQUIRE_GPU=0` y `VIDEO_CODEC=libx264`.

## Buscador web de imágenes (extra)

Para el flujo web de `custom_video`, ahora hay más fuentes y fallback anti-bot:

- Fuentes por defecto: `ddg,openverse,wikimedia,flickr,sites`
- Anti-bot/proxy para descargas bloqueadas: `IMG_ANTIBOT_PROXY=1`

Ejemplo en `.env`:

```env
IMG_SOURCES=flickr,wikimedia,openverse
IMG_ANTIBOT_PROXY=1
IMG_ANTIBOT_BROWSER=1
WIKIMEDIA_USER_AGENT=MiBotImagenes/1.0 (contacto: tu_correo@dominio.com)
CUSTOM_COMFY_FALLBACK_ON_MISSING_IMAGES=1
CUSTOM_ALLOW_PLACEHOLDER_IMAGES=0
```

Si activas `IMG_ANTIBOT_BROWSER=1`, instala navegador de Playwright una vez:

```bash
python -m playwright install chromium
```

Notas:

- El fallback con ComfyUI solo se intenta si activas ambos: `COMFYUI_ENABLED=1` y `CUSTOM_COMFY_FALLBACK_ON_MISSING_IMAGES=1`.
- Por defecto NO se permiten placeholders negros (`CUSTOM_ALLOW_PLACEHOLDER_IMAGES=0`): si no hay imagen válida, el render se detiene con error claro.
- `ddg` y `sites` quedan opcionales (puedes añadirlos en `IMG_SOURCES`), pero en algunas redes pueden bloquear con 403 y volver lento el proceso.

## Notas de mantenimiento (para escalar)

Hay una guía rápida para cambios futuros en:

- `docs/AGENT_NOTES.md`

Incluye mapa de módulos, contratos que no conviene romper, checklist para agregar nuevos módulos y deuda técnica priorizada.

## Ajustes recomendados para Reddit (timeouts/red)

Si tu conexión está lenta o Reddit responde con demoras, puedes endurecer el importador con:

```env
REDDIT_TIMEOUT_SEC=30
REDDIT_REQUEST_RETRIES=3
REDDIT_RETRY_BACKOFF_S=1.0
```

Esto evita fallos por timeout transitorio y aplica reintentos automáticos con espera incremental.

Para reducir bloqueos 429, también puedes limitar carga del scraper:

```env
REDDIT_IMPORT_SUBREDDITS=AskReddit,relationship_advice
REDDIT_FEED_PROFILE=light
REDDIT_FALLBACK_WAIT_S=8
REDDIT_MAX_COOLDOWN_WAIT_S=12
```

- `REDDIT_IMPORT_SUBREDDITS`: lista CSV de subreddits a consultar.
- `REDDIT_FEED_PROFILE`: `full` (más cobertura), `light` (menos requests), `minimal` (mínimo). Por defecto ahora: `light`.
- `REDDIT_FALLBACK_WAIT_S`: espera antes del reintento automático anti-429.
- `REDDIT_MAX_COOLDOWN_WAIT_S`: tope de espera por intento cuando Reddit responde rate-limit.
- `REDDIT_STATE_RETENTION_DAYS`: cuántos días recordar posts/comentarios ya explorados.
- `REDDIT_STATE_MAX_POST_IDS`: tope de IDs de posts cacheados.
- `REDDIT_STATE_MAX_COMMENT_IDS`: tope de IDs de comentarios cacheados.

Optimización anti-repetición (persistente):

- El importador guarda estado en `storage/reddit_import_state.json`.
- Evita volver a revisar posts/comentarios ya explorados en corridas previas.
- Rota automáticamente subreddits y feeds por ejecución para priorizar zonas no exploradas.

Si quieres una reducción fuerte de 429, activa OAuth de Reddit (script app):

```env
REDDIT_CLIENT_ID=tu_client_id
REDDIT_CLIENT_SECRET=tu_client_secret
```

Con esas variables, el importador prioriza `oauth.reddit.com` y deja `reddit/api/old` como fallback.

Chequeo rápido OAuth (1 comando):

```bash
python tools/check_reddit_oauth.py
```

Si está bien configurado, verás `✅ Token OAuth obtenido`.
