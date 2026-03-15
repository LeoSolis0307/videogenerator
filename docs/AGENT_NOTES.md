# AGENT NOTES (escalabilidad y mantenimiento)

Este archivo está pensado como guía rápida para futuras modificaciones del proyecto sin romper flujo existente.

## 1) Mapa rápido del sistema

- `main.py`
  - Orquestador CLI (selección de acción + wiring de dependencias).
  - Las acciones pesadas deben delegarse a módulos `core/*`.
- `core/cli_actions.py`
  - Handlers de acciones CLI grandes (`4`, `5`, `7`, `8`) desacoplados del entrypoint.
  - Usa `CliActionContext` para inyectar callbacks y evitar acoplar al estado global de `main.py`.
- `core/custom_video.py`
  - Flujo principal de video personalizado (plan, imágenes, render, post-proceso).
  - Punto de integración entre LLM, imágenes, TTS y renderer.
- `core/video_renderer.py`
  - Pipeline FFmpeg y validaciones de encode (incluye precheck GPU).
  - Cualquier cambio aquí impacta casi todos los renders.
- `core/llm/client.py`
  - Cliente de Ollama, retries y parseo robusto de JSON.
- `core/config.py`
  - Fuente única de configuración (`settings`) vía env vars.
- `utils/topic_file.py` + `core/topic_db.py`
  - Dedupe y trazabilidad de temas (archivo + DB).

## 2) Contratos que NO se deben romper

- `custom_plan.json` y `plan.json`:
  - Mantener claves existentes para compatibilidad hacia atrás.
- Funciones públicas usadas desde `main.py`:
  - `custom_video.generar_guion_personalizado_a_plan(...)`
  - `custom_video.renderizar_video_personalizado_desde_plan(...)`
  - `video_renderer.render_video_ffmpeg(...)`
  - `video_renderer.render_video_base_con_audio(...)`
- Si se agrega metadata nueva a JSON, debe ser opcional (fallback seguro).

## 3) Reglas de escalabilidad para cambios nuevos

1. **Config first**
   - Toda nueva perilla debe vivir en `core/config.py` y leerse desde `settings`.
   - Evitar `os.environ.get(...)` regado en módulos de negocio.

2. **Main minimal**
   - `main.py` sólo coordina entradas/salidas.
   - La lógica compleja va en `core/*`.

3. **Sin side effects ocultos**
   - Funciones utilitarias deben ser puras cuando sea posible.
   - Si hay I/O (archivo, red, subprocess), dejarlo explícito en el nombre.

4. **Errores accionables**
   - Mensajes con contexto (`qué falló` + `qué hacer`).

5. **Cambios seguros por fases**
   - Refactor interno primero (sin cambiar comportamiento).
   - Luego mejoras funcionales si se necesitan.

## 4) Patrón recomendado para nuevos módulos

Crear módulos con esta estructura mínima:

- `core/nuevo_modulo.py`
  - `def run(...) -> Resultado:` (entrypoint claro)
  - helpers privados `_...`
  - tipos explícitos en parámetros/retornos

Checklist rápido:

- [ ] Nueva config en `core/config.py`
- [ ] Validación/fallback por defecto
- [ ] Logs legibles prefijados (`[MAIN]`, `[VIDEO]`, `[LLM]`)
- [ ] Sin romper firmas públicas existentes
- [ ] Compilación OK (`python -m compileall -q main.py core utils`)

## 5) Deuda técnica priorizada (siguiente iteración)

1. Extraer modelos de datos de planes (TypedDict/dataclass) para evitar diccionarios sueltos.
2. Separar capa de proveedores de imagen por backend (Bing/Openverse/Wikimedia/etc.).
3. Añadir suite de tests de regresión para flujos `acción 4/5/7/8` con fixtures de planes.
4. Dividir `core/cli_actions.py` por submódulos (`custom_plan_actions`, `resume_actions`) si crece más.

## 6) Comandos de verificación rápida

- Sintaxis:
  - `C:/Users/leona/Downloads/videogenerator/.venv/Scripts/python.exe -m compileall -q main.py core utils`
- Unittests:
  - `C:/Users/leona/Downloads/videogenerator/.venv/Scripts/python.exe -m unittest discover -s tests -q`

## 7) Nota sobre tests actuales

Hay una expectativa desactualizada en `tests/test_config.py` sobre `custom_min_video_sec` (el valor actual real en config es 40).
Antes de usar tests como gate estricto, alinear ese test con el valor/config objetivo del proyecto.
