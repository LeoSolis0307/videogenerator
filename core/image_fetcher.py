import json
import math
import os
import random
import time
import shutil

from PIL import Image
import requests

                        
COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_WORKFLOW = os.environ.get("COMFYUI_WORKFLOW", "api_bytedance_seedream4.json")
                                                                             
COMFY_PROMPT_NODE_ID = os.environ.get("COMFYUI_PROMPT_NODE_ID", "6")
COMFY_PROMPT_FIELD = os.environ.get("COMFYUI_PROMPT_FIELD", "text")
COMFY_SAVE_NODE_ID = os.environ.get("COMFYUI_SAVE_NODE_ID", "2")
COMFY_TIMEOUT = int(os.environ.get("COMFYUI_TIMEOUT", "800"))

                                                                    
OLLAMA_URL = (os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate") or "http://localhost:11434/api/generate").strip()
                                                             
OLLAMA_TEXT_MODEL = (os.environ.get("OLLAMA_TEXT_MODEL") or "gemma2:9b").strip() or "gemma2:9b"
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "90") or "90")
                                                                            
OLLAMA_TEXT_NUM_CTX_DEFAULT = int((os.environ.get("OLLAMA_TEXT_NUM_CTX") or os.environ.get("OLLAMA_NUM_CTX") or "2048").strip() or "2048")

                                                                
                                                    
OLLAMA_OPTIONS_JSON = (os.environ.get("OLLAMA_OPTIONS_JSON") or "").strip()


def _ollama_extra_options() -> dict:
    if not OLLAMA_OPTIONS_JSON:
        return {}
    try:
        obj = json.loads(OLLAMA_OPTIONS_JSON)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        print("[IMG] ⚠️ OLLAMA_OPTIONS_JSON no es JSON válido; ignorando")
        return {}

                                  
PROMPTS = [
    "dark cinematic atmosphere",
    "creepy forest fog",
    "abandoned building night",
    "rainy city noir",
    "mysterious shadows",
    "cinematic thriller lighting",
]


def _ollama_generar_json(prompt: str) -> dict:
    extra = _ollama_extra_options()
    options = {
        "temperature": 0.65,
        "num_predict": 700,
    }
                                                                                                  
    if "num_ctx" not in extra:
        options["num_ctx"] = max(256, int(OLLAMA_TEXT_NUM_CTX_DEFAULT))
    options.update(extra)

    payload = {
        "model": OLLAMA_TEXT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("response") or "").strip()
    return json.loads(text)


def generar_prompts_historia(historia: str, *, max_prompts: int = 12) -> list[str]:
    \
\
\
\
    historia = (historia or "").strip()
    if not historia:
        return PROMPTS

    sys_prompt = (
        "You are a visual director. Read the Spanish story and propose 6 to 14 visual prompts (English, concise) "
        "covering key beats in strict chronological order. Each prompt must include: main subjects, setting, action, "
        "emotion/mood, lighting, camera angle. Prioritize simple, clearly drawable scenes (1-3 subjects), avoid text, "
        "tiny details, or abstract ideas. If a beat is too hard to illustrate realistically, simplify it to the last clear scene. "
        "Keep it photorealistic, cinematic, square framing 768x768. Return JSON: {\"prompts\": [\"prompt1\", ...]} only."
    )
    user_prompt = f"STORY:\n{historia}\n\nJSON only."

    try:
        data = _ollama_generar_json(sys_prompt + "\n\n" + user_prompt)
        candidatos = data.get("prompts") if isinstance(data, dict) else None
        if isinstance(candidatos, list):
            limpios = [str(p).strip() for p in candidatos if str(p).strip()]
            if limpios:
                return limpios[:max_prompts]
    except Exception as e:
        print(f"[IMG] ⚠️ Ollama no devolvió prompts: {e}")
        return []


def _extender_prompts_para_duracion(prompts: list[str], dur_audio: float | None, *, max_prompts: int = 12) -> list[str]:
    base = list(prompts or []) or list(PROMPTS)
    if not dur_audio or dur_audio <= 0:
        return base

    segmentos = max(1, math.ceil(dur_audio / 10.0))
    target = min(max_prompts, max(segmentos, len(base)))
    seed_list = list(base)
    while len(base) < target:
        seed = seed_list[len(base) % len(seed_list)]
        base.append(f"{seed} cinematic continuation")
    return base[:target]


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


def _comfy_workflow_cargado() -> dict | None:
    if not os.path.exists(COMFY_WORKFLOW):
        return None
    try:
        with open(COMFY_WORKFLOW, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[IMG] ⚠️ No se pudo leer workflow ComfyUI: {e}")
        return None


def _comfy_inyectar_prompt(workflow: dict, prompt: str) -> None:
    try:
        node = workflow.get(COMFY_PROMPT_NODE_ID)
        if not node or "inputs" not in node:
            raise KeyError("Nodo de prompt no encontrado en workflow")
        node["inputs"][COMFY_PROMPT_FIELD] = prompt
                                                                
        if "seed" in node.get("inputs", {}):
            node["inputs"]["seed"] = random.randint(1, 1_000_000_000)
                                                                                             
        if "max_images" in node.get("inputs", {}):
            try:
                node["inputs"]["max_images"] = 1
            except Exception:
                pass
                                                                                                
    except Exception as e:
        raise RuntimeError(f"No se pudo inyectar prompt en workflow: {e}")


def _comfy_post_prompt(workflow: dict) -> str:
    url = f"{COMFY_URL}/prompt"
    resp = requests.post(url, json={"prompt": workflow}, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        cuerpo = (resp.text or "")[:500]
        raise RuntimeError(f"HTTP {resp.status_code} al llamar /prompt: {cuerpo}") from e

    data = resp.json()
    pid = data.get("prompt_id")
    if not pid:
        raise RuntimeError("ComfyUI no devolvió prompt_id")
    return str(pid)


def _comfy_poll_images(prompt_id: str) -> list[dict]:
    url = f"{COMFY_URL}/history/{prompt_id}"
    start = time.time()
    while time.time() - start < COMFY_TIMEOUT:
        time.sleep(1.5)
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            if prompt_id not in data:
                continue
            outputs = data[prompt_id].get("outputs", {})
            node_out = outputs.get(str(COMFY_SAVE_NODE_ID)) or outputs.get(COMFY_SAVE_NODE_ID)
            if not node_out:
                                                                               
                for val in outputs.values():
                    if isinstance(val, dict) and val.get("images"):
                        node_out = val
                        break
            if node_out and node_out.get("images"):
                return node_out["images"]
        except Exception:
            pass
    raise TimeoutError("ComfyUI no devolvió imágenes a tiempo")


def _comfy_descargar_imagen(entry: dict, carpeta: str, idx: int) -> str:
    filename = entry.get("filename")
    subfolder = entry.get("subfolder", "")
    tipo = entry.get("type", "output") or "output"
    if not filename:
        raise RuntimeError("Entrada de imagen ComfyUI sin filename")

    params = {"filename": filename, "subfolder": subfolder, "type": tipo}
    url = f"{COMFY_URL}/view"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"img_{idx}.png")
    with open(ruta, "wb") as f:
        f.write(resp.content)
                                                                      
    return ruta


def _generar_imagen_comfy(prompt: str, carpeta: str, idx: int, workflow_base: dict) -> str:
    wf = json.loads(json.dumps(workflow_base))                         
    _comfy_inyectar_prompt(wf, prompt)
    pid = _comfy_post_prompt(wf)
    images = _comfy_poll_images(pid)
    if not images:
        raise RuntimeError("ComfyUI no devolvió imágenes")
    return _comfy_descargar_imagen(images[0], carpeta, idx)


def generar_imagen_prueba(prompt: str, carpeta: str) -> str | None:
    \
    prompt = (prompt or "").strip() or random.choice(PROMPTS)
    workflow = _comfy_workflow_cargado()
    if not workflow:
        print(f"[IMG] ❌ No se encontró workflow ComfyUI en {COMFY_WORKFLOW}")
        return None

    print(f"[IMG] Generando imagen de prueba con ComfyUI @ {COMFY_URL}")
    try:
        return _generar_imagen_comfy(prompt, carpeta, 0, workflow)
    except Exception as e:
        print(f"[IMG] ⚠️ ComfyUI falló en la imagen de prueba: {e}")
        return None


def descargar_imagenes(carpeta, cantidad):
    print("[IMG] Descargando imágenes...")
    prompts = [random.choice(PROMPTS) for _ in range(cantidad)]
    return descargar_imagenes_desde_prompts(carpeta, prompts, dur_audio=None)


def descargar_imagenes_desde_prompts(carpeta: str, prompts: list[str], *, dur_audio: float | None) -> list[str]:
    \
\
\
\
    print("[IMG] Descargando imágenes desde prompts...")
    workflow = _comfy_workflow_cargado()
    if not workflow:
        print(f"[IMG] ❌ No se encontró workflow ComfyUI en {COMFY_WORKFLOW}")
        return []

    prompts_final = _extender_prompts_para_duracion(prompts, dur_audio)
    print(f"[IMG] Usando ComfyUI @ {COMFY_URL} con workflow {COMFY_WORKFLOW} | prompts: {len(prompts_final)}")

    rutas = []
    for i, prompt in enumerate(prompts_final):
        try:
            rutas.append(_generar_imagen_comfy(prompt, carpeta, i, workflow))
        except Exception as e:
            print(f"[IMG] ⚠️ ComfyUI falló en la imagen {i+1}/{len(prompts_final)}: {e}")
            print("[IMG] ❌ Sin alternativas automáticas: abortando generación de imágenes")
            return []

    if len(rutas) == len(prompts_final):
        print(f"[IMG] {len(rutas)} imágenes listas (ComfyUI)")
        return rutas

    print(f"[IMG] Solo se generaron {len(rutas)}/{len(prompts_final)} imágenes")
    return []
