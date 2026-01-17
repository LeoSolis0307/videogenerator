import hashlib
import json
import math
import os
import re
import shutil
import time

from core import custom_video, image_downloader, reddit_scraper, story_generator, text_processor, tts
from core import topic_db
from core.video_renderer import (
    append_intro_to_video,
    audio_duration_seconds,
    combine_audios_with_silence,
    render_story_clip,
    render_video_base_con_audio,
    render_video_ffmpeg,
    select_video_base,
)
from utils.fs import crear_carpeta_proyecto, guardar_historial
from utils import topic_file
from utils import topic_importer


                                                   
                                                                                               
VOZ = "es-MX-JorgeNeural"
                                                
VELOCIDAD = "-10%"                                                

                                                                       
                                                                                                
                                                    
VIDEO_QUALITY = (os.environ.get("VIDEO_QUALITY") or "high").strip()
if VIDEO_QUALITY:
    os.environ["VIDEO_QUALITY"] = VIDEO_QUALITY

HISTORIAS_BASE = "historias"
HISTORIAS_GENEROS = {
    "1": "Drama y Relaciones",
    "2": "Terror y Paranormal",
    "3": "Venganza y Karens",
    "4": "Preguntas y Curiosidades",
}


def _pedir_entero(mensaje: str, *, minimo: int = 1, default: int = 1) -> int:
    try:
        val = int(input(mensaje).strip())
        if val < minimo:
            return default
        return val
    except Exception:
        return default


def _parse_indices_csv(raw: str, *, max_index: int) -> list[int]:
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[int] = []
    seen = set()
    for p in parts:
                        
        if "-" in p:
            a_raw, b_raw = [x.strip() for x in p.split("-", 1)]
            try:
                a = int(a_raw)
                b = int(b_raw)
            except Exception:
                continue
            if a <= 0 or b <= 0:
                continue
            start = min(a, b)
            end = max(a, b)
            for i in range(start, end + 1):
                if i < 1 or i > int(max_index):
                    continue
                if i in seen:
                    continue
                seen.add(i)
                out.append(i)
            continue

                           
        try:
            i = int(p)
        except Exception:
            continue
        if i < 1 or i > int(max_index):
            continue
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def _estimar_segundos(texto: str) -> float:
    palabras = len((texto or "").split())
    if palabras <= 0:
        return 0.0
                                                                                        
    wpm = 140.0
    estimado = (palabras / wpm) * 60.0                   
    estimado = estimado * 1.50 + 2.0                           
    return max(3.0, estimado)


def _crear_silencio(segundos: float, carpeta: str) -> str:
    import wave

    seg = max(0.0, segundos)
    if seg <= 0:
        return ""
    rate = 48000
    channels = 1
    sampwidth = 2
    frames = int(seg * rate)
    path = os.path.join(carpeta, f"silencio_{seg:.2f}s.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(b"\x00" * frames * channels * sampwidth)
    return path


def _filtrar_comentarios(comentarios, limite=200):
    filtrados = []
    for c in comentarios:
        if c.get("kind") != "t1":
            continue
        body = c.get("data", {}).get("body", "")
        cid = c.get("data", {}).get("id", "")
        if not body or "[deleted]" in body:
            continue
        if len(body) <= 80:
            continue
        filtrados.append((body, cid))
        if len(filtrados) >= limite:
            break
    return filtrados


def _leer_historias_locales(carpeta_genero: str):
    os.makedirs(carpeta_genero, exist_ok=True)
    historias = []
    for nombre in sorted(os.listdir(carpeta_genero)):
        if nombre.startswith("0"):
            continue            
        if not nombre.lower().endswith(".txt"):
            continue
        ruta = os.path.join(carpeta_genero, nombre)
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                texto = f.read().strip()
            if texto:
                historias.append((texto, ruta))
        except Exception as e:
            print(f"[MAIN] ⚠️ No se pudo leer {ruta}: {e}")
    return historias


def _marcar_historias_usadas(rutas):
    for ruta in rutas:
        try:
            base = os.path.basename(ruta)
            dirname = os.path.dirname(ruta)
            if base.startswith("0"):
                continue
            nuevo = os.path.join(dirname, "0" + base)
                                                              
            while os.path.exists(nuevo):
                nuevo = os.path.join(dirname, "0" + os.path.basename(nuevo))
            os.rename(ruta, nuevo)
        except Exception as e:
            print(f"[MAIN] ⚠️ No se pudo marcar historia como usada ({ruta}): {e}")


def _seleccionar_genero() -> str:
    print("Elige género:")
    for clave, nombre in HISTORIAS_GENEROS.items():
        print(f"  {clave}. {nombre}")
    seleccion = input("Opción: ").strip()
    genero = HISTORIAS_GENEROS.get(seleccion, HISTORIAS_GENEROS["1"])
                                                                                                
    for nombre in HISTORIAS_GENEROS.values():
        os.makedirs(os.path.join(HISTORIAS_BASE, nombre), exist_ok=True)
    return genero


def _es_si(raw: str) -> bool:
    return (raw or "").strip().lower() in {"s", "si", "sí", "y", "yes", "1", "true"}


class _WindowsKeepAwake:
    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._ok = False
        self._kernel32 = None

    def __enter__(self):
        if (not self._enabled) or os.name != "nt":
            return self
        try:
            import ctypes

            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED

            self._kernel32 = ctypes.windll.kernel32
            self._kernel32.SetThreadExecutionState(flags)
            self._ok = True
        except Exception:
            self._ok = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if (not self._enabled) or os.name != "nt":
            return False
        if not self._ok:
            return False
        try:
            import ctypes

            ES_CONTINUOUS = 0x80000000
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            pass
        return False


def _start_cancel_suspend_listener(flag: dict) -> None:
    return


def _maybe_suspend_with_grace(cancel_flag: dict, *, grace_seconds: int = 60) -> None:
    return


def _planes_pendientes() -> list[str]:
    pendientes = []
    base = os.path.abspath("output")
    if not os.path.isdir(base):
        return pendientes
    for entry in os.scandir(base):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("0"):
            continue
        plan_file = os.path.join(entry.path, "plan.json")
        if os.path.exists(plan_file):
            pendientes.append(entry.path)
                                                                     
    pendientes.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pendientes


def _custom_plans_pendientes() -> list[str]:
    pendientes: list[str] = []
    base = os.path.abspath("output")
    if not os.path.isdir(base):
        return pendientes
    for entry in os.scandir(base):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("0"):
            continue
        plan_file = os.path.join(entry.path, "custom_plan.json")
        if not os.path.exists(plan_file):
            continue


        done = False
        finalized = False
        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                plan = json.load(f) or {}
            done = bool(plan.get("render_done"))
            finalized = bool(plan.get("topic_finalized"))
        except Exception:
            done = False
            finalized = False

        if not done:
            try:
                vf = os.path.join(entry.path, "Video_Final.mp4")
                if os.path.exists(vf) and os.path.getsize(vf) > 250_000:
                    done = True
            except Exception:
                pass

        
        # Pendiente si:
        # - falta render, o
        # - ya renderizó pero falta finalizar (DB/archivo)
        if (not done) or (done and not finalized):
            pendientes.append(entry.path)

    pendientes.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pendientes


def _custom_plan_flags(ruta: str) -> dict:
    plan_path = os.path.join(ruta, "custom_plan.json")
    rendered = False
    finalized = False
    try:
        if os.path.exists(plan_path):
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f) or {}
            rendered = bool(plan.get("render_done"))
            finalized = bool(plan.get("topic_finalized"))
    except Exception:
        pass
    if not rendered:
        try:
            vf = os.path.join(ruta, "Video_Final.mp4")
            if os.path.exists(vf) and os.path.getsize(vf) > 250_000:
                rendered = True
        except Exception:
            pass
    return {"rendered": rendered, "finalized": finalized, "plan_path": plan_path}


def _finalizar_tema_custom_renderizado(ruta: str) -> None:
    try:
        plan_path = os.path.join(ruta, "custom_plan.json")
        if not os.path.exists(plan_path):
            return
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f) or {}
        brief = str(plan.get("brief") or "").strip()
        topic_source = str(plan.get("topic_source") or "").strip().lower()
        topic_file_path = str(plan.get("topic_file") or "").strip()

        if brief:
            topic_db.register_topic_if_new(
                brief,
                kind="custom",
                plan_dir=ruta,
                threshold=0.98,
            )

        try:
            topic_db.delete_by_plan_dir(ruta, kind="custom_pending")
        except Exception:
            pass

        if topic_source == "file" and topic_file_path and brief:
            marcado = topic_file.mark_topic_used(brief, topic_file_path)
            if marcado:
                print("[MAIN] ✅ Tema marcado como usado en archivo.")

        try:
            import time as _time

            plan["topic_finalized"] = True
            plan["topic_finalized_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudo finalizar tema (DB/archivo): {e}")


def _marcar_plan_completado(path: str):
    if not path:
        return
    base = os.path.abspath(path)
    parent = os.path.dirname(base)
    name = os.path.basename(base)
    if name.startswith("0"):
        return
    nuevo = os.path.join(parent, "0" + name)
    while os.path.exists(nuevo):
        nuevo = os.path.join(parent, "0" + os.path.basename(nuevo))
    shutil.move(base, nuevo)
    plan_json = os.path.join(nuevo, "plan.json")
    if os.path.exists(plan_json):
        try:
            os.remove(plan_json)
        except Exception:
            print(f"[MAIN] ⚠️ No se pudo borrar plan.json en {nuevo}")
    print(f"[MAIN] Plan marcado como completado: {nuevo}")


def _generar_timeline(prompts: list[str], dur_est: float) -> list[dict]:
    prompts = [p for p in (prompts or []) if p]
    if not prompts:
        return []
    dur = max(5.0, dur_est or 0)
    n = max(len(prompts), math.ceil(dur / 10.0))
    seg_dur = max(5.0, min(10.0, dur / n))
    timeline = []
    start = 0.0
    for i in range(n):
        end = start + seg_dur
        prompt = prompts[i % len(prompts)]
        timeline.append({"prompt": prompt, "start": round(start, 2), "end": round(end, 2)})
        start = end
    return timeline


def _segmentar_historia_en_prompts(historia: str, prompts: list[str]) -> list[str]:
    texto = (historia or "").strip()
    prompts_limpios = [p for p in (prompts or []) if p]
    if not texto:
        return []
    if not prompts_limpios:
        return [texto]

    oraciones = re.split(r"(?<=[.!?¡¿])\s+", texto)
    oraciones = [o.strip() for o in oraciones if o.strip()]
    if not oraciones:
        return [texto]

    objetivo = min(len(prompts_limpios), max(1, len(oraciones)))
    chunk = max(1, math.ceil(len(oraciones) / objetivo))
    segmentos = []
    for i in range(0, len(oraciones), chunk):
        segmentos.append(" ".join(oraciones[i : i + chunk]).strip())

                                                                             
    segmentos = [s for s in segmentos if s]
    if len(segmentos) > objetivo:
        segmentos = segmentos[:objetivo]

    return segmentos


def _generar_video(usar_video_base: bool, indice: int, total: int, *, usar_historias_locales: bool, carpeta_genero: str | None, fase: int, plan_path: str | None) -> bool:
    print(f"[MAIN] ===== Video {indice}/{total} (fase {fase}) =====")

    carpeta = crear_carpeta_proyecto(prefix="plan" if fase == 1 else None)

    video_base_path = None
    video_base_dur = 0.0
    plan_data = None

                         
    if fase == 2:
        if not plan_path:
            print("[MAIN] Debes indicar la ruta de plan (carpeta con plan.json)")
            return False
        plan_file = os.path.join(plan_path, "plan.json")
        if not os.path.exists(plan_file):
            print(f"[MAIN] No se encontró plan.json en {plan_path}")
            return False
        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
        except Exception as e:
            print(f"[MAIN] No se pudo leer plan: {e}")
            return False
        usar_video_base = bool(plan_data.get("usar_video_base", usar_video_base))
        texto_elegido = plan_data.get("texto_en", "")
        texto_es_plan = plan_data.get("texto_es", "")
        ids_seleccionados = plan_data.get("ids", []) or [""]
        prompts_plan = plan_data.get("prompts", [])
        timeline_plan = plan_data.get("timeline", [])
        if not texto_es_plan:
            print("[MAIN] El plan no tiene texto_es")
            return False
        textos_en = [texto_elegido or texto_es_plan]
        textos_es = [texto_es_plan]
        prompts = prompts_plan
        timeline = timeline_plan
        carpeta = plan_path                               
    else:
                                      
        if usar_video_base:
            video_base_path, video_base_dur = select_video_base(None)
            if not video_base_path:
                print("[MAIN] No se encontró video base utilizable")
                return False
            if video_base_dur <= 0:
                print("[MAIN] No se pudo detectar duración del video base. Abortando.")
                return False
            print(f"[MAIN] Video base seleccionado: {video_base_path} (~{video_base_dur:.1f}s)")

        if usar_historias_locales:
            if not carpeta_genero:
                print("[MAIN] No se especificó carpeta de historias")
                return False
            comentarios_filtrados = _leer_historias_locales(carpeta_genero)
            if not comentarios_filtrados:
                print(f"[MAIN] No se encontraron historias en {carpeta_genero}")
                return False
        else:
            post = reddit_scraper.obtener_post()
            if not post:
                print("[MAIN] No se pudo obtener post")
                return False

            comentarios = reddit_scraper.obtener_comentarios(post["permalink"])
            comentarios_filtrados = _filtrar_comentarios(comentarios, limite=200)

        if not comentarios_filtrados:
            print("[MAIN] No se encontraron historias")
            return False

                                                                    
        comentarios_filtrados = sorted(comentarios_filtrados, key=lambda t: len(t[0]), reverse=True)
        texto_elegido, cid_elegido = comentarios_filtrados[0]
        comentarios_filtrados = [(texto_elegido, cid_elegido)]

        textos_en = [texto_elegido]
        ids_seleccionados = [cid_elegido]

        if usar_historias_locales:
            _marcar_historias_usadas(ids_seleccionados)

    print(f"[MAIN] {len(textos_en)} textos obtenidos")

                                                                     
    if fase != 2:
        textos_es = text_processor.traducir_lista(textos_en)

    print("[DEBUG] Primer texto que irá al TTS:")
    print(textos_es[0][:200])

    if fase == 2:
        historia_es = textos_es[0]
        prompts = prompts or []
        segments_plan = plan_data.get("segments", []) if plan_data else []
        segments = segments_plan or _segmentar_historia_en_prompts(historia_es, prompts)
        if len(segments) < len(prompts):
            prompts = prompts[: len(segments)]
        timeline = plan_data.get("timeline", []) if plan_data else []
    else:
                                                    
        historia_es = textos_es[0]
        prompts = image_downloader.generar_prompts_historia(historia_es)
        if not prompts:
            print("[MAIN] No se generaron prompts de imagen (IA)")
            return False

        segments = _segmentar_historia_en_prompts(historia_es, prompts)
        if not segments:
            print("[MAIN] No se pudo segmentar la historia para sincronizar audio e imágenes")
            return False
        if len(segments) < len(prompts):
            prompts = prompts[: len(segments)]

    if not segments:
        print("[MAIN] No hay segmentos listos para sincronizar audio e imágenes")
        return False

    dur_est = _estimar_segundos(historia_es)
    timeline = _generar_timeline(prompts, dur_est)

    if fase == 1:
        plan = {
            "texto_en": textos_en[0],
            "texto_es": textos_es[0],
            "prompts": prompts,
            "timeline": timeline,
            "dur_est": dur_est,
            "ids": ids_seleccionados,
            "segments": segments,
            "usar_video_base": usar_video_base,
        }
        plan_file = os.path.join(carpeta, "plan.json")
        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"[MAIN] ✅ Plan guardado en {plan_file}")
        return True

                                                                           
    if fase == 2 and timeline:
        dur_est = timeline[-1].get("end", _estimar_segundos(textos_es[0]))
    else:
        dur_est = _estimar_segundos(textos_es[0])

    imagenes = image_downloader.descargar_imagenes_desde_prompts(carpeta, prompts, dur_audio=None)
    if not imagenes:
        print("[MAIN] No se descargaron imágenes")
        return False

                                                    
    audios = tts.generar_audios(segments, carpeta, voz=VOZ, velocidad=VELOCIDAD)
    if not audios:
        print("[MAIN] No se generaron audios")
        return False

    if len(audios) != len(imagenes) or len(audios) != len(prompts):
        min_items = min(len(audios), len(imagenes), len(prompts))
        print(f"[MAIN] Ajustando a {min_items} segmentos por desbalance entre audios/prompts/imagenes")
        audios = audios[:min_items]
        imagenes = imagenes[:min_items]
        prompts = prompts[:min_items]
        segments = segments[:min_items]
    if not audios or not imagenes:
        print("[MAIN] No hay suficientes audios o imágenes para renderizar")
        return False

    duraciones_imgs = [max(0.6, audio_duration_seconds(a)) for a in audios]
    audio_final = combine_audios_with_silence(audios, carpeta, gap_seconds=0, min_seconds=None, max_seconds=None)

    if usar_video_base:
        try:
            video_final = render_video_base_con_audio(video_base_path, audio_final, carpeta, videos_dir=None)
        except Exception as e:
            print(f"[MAIN] Error renderizando con video base: {e}")
            return False
    else:
                                                                       
        timeline = []
        pos = 0.0
        for prompt, d in zip(prompts, duraciones_imgs):
            start = pos
            end = pos + d
            timeline.append({"prompt": prompt, "start": round(start, 2), "end": round(end, 2)})
            pos = end

        video_final = render_video_ffmpeg(imagenes, audio_final, carpeta, tiempo_img=None, durations=duraciones_imgs)

    append_intro_to_video(video_final, title_text=textos_es[0])

    try:
        claves = []
        for cid, texto in zip(ids_seleccionados, textos_es):
            if cid:
                claves.append(f"reddit_comment_used:{cid}")
            else:
                h = hashlib.sha1(texto.encode("utf-8")).hexdigest()
                claves.append(f"reddit_comment_hash:{h}")
        if claves:
            guardar_historial(claves)
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudo guardar historial de historias usadas: {e}")

    print("[MAIN] ✅ Video generado")

                                                    
    if fase == 2 and plan_path:
        try:
            _marcar_plan_completado(plan_path)
        except Exception as e:
            print(f"[MAIN] ⚠️ No se pudo marcar plan completado: {e}")
    return True


if __name__ == "__main__":
    print("[MAIN] Iniciando proceso")

                                                                             
    try:
        topic_file.ensure_topics_file(topic_file.TOPICS_FILE_DEFAULT)
        disponibles = topic_file.load_topics_available_with_flags(topic_file.TOPICS_FILE_DEFAULT)
        print(
            f"[MAIN] Temas en archivo (sin prefijo 0): {len(disponibles)} -> {topic_file.TOPICS_FILE_DEFAULT}"
        )
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudo leer temas_custom.txt: {e}")

                                                                             
                                                                                                     
    text_model = getattr(custom_video, "OLLAMA_TEXT_MODEL", "") or "(desconocido)"
    vision_model = (os.environ.get("VISION_MODEL") or "minicpm-v:latest").strip() or "minicpm-v:latest"
    ollama_url = (os.environ.get("OLLAMA_URL") or "http://localhost:11434/api/generate").strip()
    print(f"[MAIN] Modelo texto: {text_model} (OLLAMA_TEXT_MODEL/OLLAMA_MODEL o default)")
    print(f"[MAIN] Modelo visión: {vision_model} (env VISION_MODEL)")
    print(f"[MAIN] Ollama URL: {ollama_url} (env OLLAMA_URL)")

    accion = input(
        "¿Qué deseas hacer? (1 = Videos, 2 = Textos, 3 = Imagen de prueba, 4 = Video personalizado, 5 = Renderizar personalizado, 6 = Importar temas, 7 = Reanudar último personalizado, 8 = Reanudar TODOS personalizados pendientes): "
    ).strip()

    if accion == "6":
        print("\n[MAIN] Importar temas a storage/temas_custom.txt")
        modo = input("Fuente (1=pegar texto, 2=archivo .txt) [1]: ").strip()
        blob = ""
        if modo == "2":
            ruta = input("Ruta del archivo .txt: ").strip().strip('"')
            if not ruta or not os.path.exists(ruta):
                print("[MAIN] Ruta inválida")
                raise SystemExit(0)
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    blob = f.read()
            except Exception as e:
                print(f"[MAIN] No se pudo leer archivo: {e}")
                raise SystemExit(0)
        else:
            print("Pega el texto (multi-línea). Escribe END en una línea sola para terminar:")
            lines: list[str] = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip() == "END":
                    break
                lines.append(line)
            blob = "\n".join(lines)

        prompts = topic_importer.parse_prompts_from_blob(blob)
        if not prompts:
            print("[MAIN] No se encontraron prompts.")
            raise SystemExit(0)

        aceptados, descartados = topic_importer.dedupe_prompts(prompts, topics_path=topic_file.TOPICS_FILE_DEFAULT)

        if aceptados:
            topic_file.append_topics([(p, False) for p in aceptados], path=topic_file.TOPICS_FILE_DEFAULT)

        print(f"\n[MAIN] Agregados: {len(aceptados)}")
        for i, p in enumerate(aceptados, start=1):
            print(f"  + {i}. {p[:120]}")

        print(f"\n[MAIN] Descartados: {len(descartados)}")
        for i, (p, motivo) in enumerate(descartados, start=1):
            print(f"  - {i}. ({motivo}) {p[:120]}")

        if descartados:
            sel = input(
                "\n¿Quieres agregar alguno de los descartados de todos modos? (ej: 1,3,5 | Enter=skip): "
            ).strip()
            if sel:
                idxs = _parse_indices_csv(sel, max_index=len(descartados))
                if idxs:
                    force = [(descartados[i - 1][0], True) for i in idxs]
                    topic_file.append_topics(force, path=topic_file.TOPICS_FILE_DEFAULT)
                    print(f"[MAIN] Forzados agregados: {len(force)} (con prefijo '!')")

        print(f"\n[MAIN] Listo. Archivo: {topic_file.TOPICS_FILE_DEFAULT}")
        raise SystemExit(0)

    if accion == "2":
        total_textos = _pedir_entero("¿Cuántas historias generar?: ", minimo=1, default=1)
        genero = _seleccionar_genero()
        try:
            resultados = story_generator.generar_historias(genero, total_textos)
            print(f"[MAIN] Historias generadas: {len(resultados)}/{total_textos}")
        except Exception as e:
            print(f"[MAIN] ⚠️ Error generando historias: {e}")
        raise SystemExit(0)

    if accion == "3":
        prompt = input("Prompt para la imagen de prueba: ").strip()
        carpeta_prueba = os.path.join("imagenes_prueba")
        os.makedirs(carpeta_prueba, exist_ok=True)
        try:
            ruta = image_downloader.generar_imagen_prueba(prompt, carpeta_prueba)
            if ruta:
                print(f"[MAIN] ✅ Imagen guardada en: {ruta}")
            else:
                print("[MAIN] ❌ No se pudo generar la imagen de prueba")
        except Exception as e:
            print(f"[MAIN] ⚠️ Error generando imagen de prueba: {e}")
        raise SystemExit(0)

    if accion == "4":
        origen_temas = input("Temas (1=escribir, 2=archivo storage/temas_custom.txt) [1]: ").strip()

        temas_path = topic_file.TOPICS_FILE_DEFAULT
        temas_disponibles: list[tuple[str, bool]] = []
        if origen_temas == "2":
            topic_file.ensure_topics_file(temas_path)
            temas_disponibles = topic_file.load_topics_available_with_flags(temas_path)
            if not temas_disponibles:
                print(f"[MAIN] No hay temas disponibles en {temas_path} (sin prefijo '0').")
                print("[MAIN] Agrega 1 tema por línea en el archivo y vuelve a intentar.")
                raise SystemExit(0)
            print("[MAIN] Revisando temas repetidos contra la DB...")
            elegibles: list[tuple[str, bool]] = []
            repetidos = 0
            for tema, forced in temas_disponibles:
                brief = (tema or "").strip()
                if not brief:
                    continue
                if forced:
                    elegibles.append((brief, True))
                    continue
                try:
                    match = topic_db.find_similar_topic(brief, kinds=("custom", "custom_pending"), threshold=0.90)
                except Exception:
                    match = None
                if match is not None:
                    repetidos += 1
                    continue
                elegibles.append((brief, False))

            print(
                f"[MAIN] Temas: {len(temas_disponibles)} | Elegibles (no repetidos): {len(elegibles)} | Repetidos: {repetidos}"
            )
            if not elegibles:
                print("[MAIN] No hay temas elegibles (todos parecen repetidos).")
                raise SystemExit(0)

            max_n = len(elegibles)
            total_custom = _pedir_entero(f"¿Cuántos videos sacar del archivo? (max {max_n}): ", minimo=1, default=1)
            total_custom = min(total_custom, max_n)
            temas_disponibles = elegibles
        else:
            total_custom = _pedir_entero("¿Cuántos videos personalizados quieres crear?: ", minimo=1, default=1)

        dur_opt = input("Duración mínima para TODOS (1=1 minuto, 2=5 minutos) [1]: ").strip()
        min_seconds = 60 if dur_opt != "2" else 300

        sel_all = input("¿Quieres elegir manualmente las imágenes para TODOS los videos? (s/N): ").strip().lower()
        seleccionar_imagenes_all = sel_all == "s"

        auto_render_after_plans = _es_si(input("¿Al terminar los guiones/planes, renderizar automáticamente? (s/N): "))
        unload_text_model_after_plans = False
        if auto_render_after_plans:
            unload_text_model_after_plans = _es_si(
                input("¿Intentar descargar/unload el modelo de texto al terminar guiones? (s/N): ")
            )

                                                                           
        briefs: list[str] = []
        seleccionar_flags: list[bool] = []
        brief_topic_files: list[str | None] = []

        def _tema_repetido(brief: str) -> bool:
            try:
                                                                         
                match = topic_db.find_similar_topic(brief, kinds=("custom", "custom_pending"))
            except Exception as e:
                print(f"[MAIN] ⚠️ No se pudo validar tema en DB: {e}")
                match = None
            if match is None:
                return False
            print(
                "[MAIN] ⚠️ Tema repetido detectado. "
                f"(sim={match.similarity:.2f}) Ya existe algo muy parecido: '{match.brief[:120]}'"
            )
            return True

        print("\n[MAIN] Selección de temas...")
        if origen_temas == "2":
                                                                              
            i = 0
            for tema, forced in temas_disponibles:
                if len(briefs) >= total_custom:
                    break
                i += 1
                brief = (tema or "").strip()
                if not brief:
                    continue
                if not forced and _tema_repetido(brief):
                    continue

                seleccionar_imagenes = seleccionar_imagenes_all

                briefs.append(brief)
                seleccionar_flags.append(seleccionar_imagenes)
                brief_topic_files.append(temas_path)

            if len(briefs) < total_custom:
                print(
                    f"[MAIN] ⚠️ Solo se pudieron tomar {len(briefs)}/{total_custom} temas del archivo "
                    "(los demás se saltaron por repetidos)."
                )
                if not briefs:
                    raise SystemExit(0)
        else:
            print("[MAIN] Ingresa los prompts de cada historia (luego comenzará el render).")
            for i in range(1, total_custom + 1):
                while True:
                    brief = input(f"Prompt/Tema para la historia {i}/{total_custom}: ").strip()
                    if not brief:
                        print("[MAIN] ⚠️ El prompt no puede estar vacío.")
                        continue
                    if _tema_repetido(brief):
                        print("[MAIN] Escribe otro tema para evitar repetir videos.")
                        continue
                    break

                seleccionar_imagenes = seleccionar_imagenes_all

                briefs.append(brief)
                seleccionar_flags.append(seleccionar_imagenes)
                brief_topic_files.append(None)

        print("\n[MAIN] Fase 1: generando SOLO guiones/planes (sin imágenes, sin render)...")

                                                                                 
        try:
            if not custom_video.check_text_llm_ready():
                print("[MAIN] ❌ Abortando Fase 1: el LLM de texto no está disponible en Ollama.")
                print("[MAIN] 💡 Tip (Gemma 2): instala uno más liviano y úsalo para guiones:")
                print("[MAIN]   - `ollama pull gemma2:2b`  (recomendado para PCs con poca RAM)")
                print("[MAIN]   - `setx OLLAMA_TEXT_MODEL gemma2:2b`  (abre una nueva terminal luego)")
                raise SystemExit(1)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[MAIN] ❌ Abortando Fase 1: fallo el preflight de Ollama: {e}")
            raise SystemExit(1)

        creados = 0
        planes_creados: list[str] = []
        for i, (brief, seleccionar_imagenes, tema_file) in enumerate(
            zip(briefs, seleccionar_flags, brief_topic_files), start=1
        ):
            try:
                carpeta = custom_video.generar_guion_personalizado_a_plan(
                    brief,
                    min_seconds=min_seconds,
                    seleccionar_imagenes=seleccionar_imagenes,
                )
                if carpeta:
                                                                                                
                    try:
                        topic_db.register_topic_if_new(brief, kind="custom_pending", plan_dir=carpeta, threshold=0.90)
                    except Exception as e:
                        print(f"[MAIN] ⚠️ No se pudo registrar tema pendiente en DB: {e}")

                                                                                
                    try:
                        plan_path = os.path.join(carpeta, "custom_plan.json")
                        if os.path.exists(plan_path):
                            with open(plan_path, "r", encoding="utf-8") as f:
                                plan = json.load(f) or {}
                            if tema_file:
                                plan["topic_source"] = "file"
                                plan["topic_file"] = tema_file
                            else:
                                plan["topic_source"] = "manual"
                            with open(plan_path, "w", encoding="utf-8") as f:
                                json.dump(plan, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"[MAIN] ⚠️ No se pudo guardar metadata del tema: {e}")
                    creados += 1
                    planes_creados.append(carpeta)
                    print(f"[MAIN] ✅ Plan creado {i}/{total_custom}: {carpeta}")
                else:
                    print(f"[MAIN] ❌ No se pudo crear plan {i}/{total_custom}")
            except Exception as e:
                print(f"[MAIN] ⚠️ Error creando plan (historia {i}/{total_custom}): {e}")

        print(f"[MAIN] Planes creados: {creados}/{total_custom}")

        if auto_render_after_plans and planes_creados:
            if unload_text_model_after_plans:
                try:
                    ok = custom_video.intentar_descargar_modelo_texto()
                    print("[MAIN] Unload modelo texto:", "OK" if ok else "No se pudo / no aplica")
                except Exception as e:
                    print(f"[MAIN] No se pudo intentar unload: {e}")

            exitos = 0
            try:
                with _WindowsKeepAwake(enabled=True):
                    for k, ruta in enumerate(planes_creados, start=1):
                        try:
                            ok = custom_video.renderizar_video_personalizado_desde_plan(
                                ruta,
                                voz=VOZ,
                                velocidad=VELOCIDAD,
                                interactive=False,
                            )
                            if ok:
                                exitos += 1
                            print(f"[MAIN] Render {k}/{len(planes_creados)}:", "OK" if ok else "FAIL")
                        except Exception as e:
                            print(f"[MAIN] Error renderizando (plan {k}): {e}")
            except KeyboardInterrupt:
                print("\n[MAIN] Interrumpido por el usuario.")
                raise SystemExit(0)

            print(f"[MAIN] Renders completados: {exitos}/{len(planes_creados)}")
            raise SystemExit(0)
        descargar = input("¿Quieres intentar descargar el modelo de texto para liberar VRAM? (s/N): ").strip().lower()
        if descargar == "s":
            try:
                ok = custom_video.intentar_descargar_modelo_texto()
                print("[MAIN] Unload modelo texto:", "✅ OK" if ok else "❌ No se pudo / no aplica")
            except Exception as e:
                print(f"[MAIN] ⚠️ No se pudo intentar unload: {e}")

        print("[MAIN] Ahora usa la opción 5 para Fase 2 (imágenes+render) desde el plan.")
        raise SystemExit(0)

    if accion == "5":
        planes = _custom_plans_pendientes()
        if planes:
            print("\n[MAIN] Planes personalizados disponibles:")
            for i, p in enumerate(planes, start=1):
                print(f"  {i}. {p}")

            total_renders = _pedir_entero("¿Cuántos renders quieres hacer?: ", minimo=1, default=1)
            sel_raw = input("Selecciona planes (ej: 1,2,3) [1]: ").strip()
            seleccion = _parse_indices_csv(sel_raw, max_index=len(planes))
            if not seleccion:
                seleccion = [1]

                                                                                           
            if total_renders > len(seleccion):
                for j in range(1, len(planes) + 1):
                    if j in seleccion:
                        continue
                    seleccion.append(j)
                    if len(seleccion) >= total_renders:
                        break

                                                                 
            seleccion = seleccion[:total_renders]

            exitos = 0
            interactive_mode = len(seleccion) <= 1

            try:
                with _WindowsKeepAwake(enabled=True):
                    for k, idx in enumerate(seleccion, start=1):
                        ruta = planes[idx - 1]
                        try:
                            ok = custom_video.renderizar_video_personalizado_desde_plan(
                                ruta,
                                voz=VOZ,
                                velocidad=VELOCIDAD,
                                interactive=interactive_mode,
                            )
                            if ok:
                                exitos += 1

                            if ok:
                                _finalizar_tema_custom_renderizado(ruta)

                            print(
                                f"[MAIN] Render {k}/{len(seleccion)} (plan {idx}):",
                                "✅ Exito" if ok else "❌ Falló",
                            )
                        except Exception as e:
                            print(f"[MAIN] ⚠️ Error renderizando (plan {idx}): {e}")
            except KeyboardInterrupt:
                print("\n[MAIN] Interrumpido por el usuario.")
                raise SystemExit(0)

            print(f"[MAIN] Renders completados: {exitos}/{len(seleccion)}")
            raise SystemExit(0)
        else:
            raw = input("Rutas (separadas por coma) de carpeta output/custom_... o custom_plan.json: ").strip().strip('"')
            if not raw:
                print("[MAIN] No se indicó ruta")
                raise SystemExit(0)

            rutas = [r.strip().strip('"') for r in raw.split(",") if r.strip().strip('"')]
            total_renders = len(rutas)
            exitos = 0
            for i, ruta in enumerate(rutas, start=1):
                try:
                    ok = custom_video.renderizar_video_personalizado_desde_plan(
                        ruta,
                        voz=VOZ,
                        velocidad=VELOCIDAD,
                        interactive=(len(rutas) <= 1),
                    )
                    if ok:
                        exitos += 1
                        _finalizar_tema_custom_renderizado(ruta)
                    print(f"[MAIN] Render {i}/{total_renders}:", "✅ Exito" if ok else "❌ Falló")
                except Exception as e:
                    print(f"[MAIN] ⚠️ Error renderizando ({ruta}): {e}")

            print(f"[MAIN] Renders completados: {exitos}/{total_renders}")
            raise SystemExit(0)

    if accion == "7":
        planes = _custom_plans_pendientes()
        if not planes:
            print("[MAIN] No hay planes personalizados pendientes para reanudar.")
            raise SystemExit(0)

        ruta = planes[0]
        print(f"[MAIN] Reanudando último plan personalizado pendiente: {ruta}")
        try:
            flags = _custom_plan_flags(ruta)
            if flags.get("rendered"):
                _finalizar_tema_custom_renderizado(ruta)
                print("[MAIN] Reanudación: ✅ Ya estaba renderizado; solo se finalizó.")
            else:
                with _WindowsKeepAwake(enabled=True):
                    ok = custom_video.renderizar_video_personalizado_desde_plan(
                        ruta,
                        voz=VOZ,
                        velocidad=VELOCIDAD,
                        interactive=False,
                    )
                if ok:
                    _finalizar_tema_custom_renderizado(ruta)
                print("[MAIN] Reanudación:", "✅ Exito" if ok else "❌ Falló")
        except KeyboardInterrupt:
            print("\n[MAIN] Interrumpido por el usuario.")
            raise SystemExit(0)
        except Exception as e:
            print(f"[MAIN] ⚠️ Error reanudando último plan: {e}")
            raise SystemExit(0)

        raise SystemExit(0)

    if accion == "8":
        planes = _custom_plans_pendientes()
        if not planes:
            print("[MAIN] No hay planes personalizados pendientes.")
            raise SystemExit(0)

        print(f"[MAIN] Reanudando cola de personalizados pendientes: {len(planes)}")
        exitos = 0
        try:
            with _WindowsKeepAwake(enabled=True):
                for i, ruta in enumerate(planes, start=1):
                    try:
                        flags = _custom_plan_flags(ruta)
                        if flags.get("rendered"):
                            _finalizar_tema_custom_renderizado(ruta)
                            exitos += 1
                            print(f"[MAIN] {i}/{len(planes)}: ✅ Ya renderizado; finalizado: {ruta}")
                            continue

                        ok = custom_video.renderizar_video_personalizado_desde_plan(
                            ruta,
                            voz=VOZ,
                            velocidad=VELOCIDAD,
                            interactive=False,
                        )
                        if ok:
                            _finalizar_tema_custom_renderizado(ruta)
                            exitos += 1
                        print(f"[MAIN] {i}/{len(planes)}:", "✅ Exito" if ok else "❌ Falló", f"| {ruta}")
                    except Exception as e:
                        print(f"[MAIN] ⚠️ Error en cola (item {i}/{len(planes)}): {e}")
        except KeyboardInterrupt:
            print("\n[MAIN] Interrumpido por el usuario.")
            raise SystemExit(0)

        print(f"[MAIN] Cola completada: {exitos}/{len(planes)}")
        raise SystemExit(0)

                    
    fase = _pedir_entero("Selecciona fase (1=plan, 2=render pendientes): ", minimo=1, default=2)

    plan_paths = []
    if fase == 2:
        pend = _planes_pendientes()
        if not pend:
            print("[MAIN] No hay planes pendientes (carpetas sin 0 con plan.json)")
            raise SystemExit(0)
        print(f"[MAIN] Planes pendientes detectados: {len(pend)}")
        for p in pend:
            print(f"   - {p}")
        plan_paths = pend
        usar_video_base = False                     
        usar_historias_locales = False
        carpeta_genero = None
        total_videos = len(plan_paths)
    else:
        total_videos = _pedir_entero("¿Cuántos videos (planes) generar?: ", minimo=1, default=1)
        modo = input("Selecciona modo (1 = IA/imagenes, 2 = Video base): ").strip()
        usar_video_base = modo == "2"

        origen = input("Origen de historias (1 = Reddit automático, 2 = Carpeta 'historias'): ").strip()
        usar_historias_locales = origen == "2"
        carpeta_genero = None
        plan_paths = [None] * total_videos

        if usar_historias_locales:
            genero = _seleccionar_genero()
            carpeta_genero = os.path.join(HISTORIAS_BASE, genero)
            print(f"[MAIN] Usando historias locales desde: {carpeta_genero}")

    exitos = 0
    for i in range(total_videos):
        plan_path = plan_paths[i] if i < len(plan_paths) else None
        try:
            if _generar_video(
                usar_video_base,
                i + 1,
                total_videos,
                usar_historias_locales=usar_historias_locales,
                carpeta_genero=carpeta_genero,
                fase=fase,
                plan_path=plan_path,
            ):
                exitos += 1
        except Exception as e:
            print(f"[MAIN] ⚠️ Error inesperado en video {i+1}/{total_videos}: {e}")
    print(f"[MAIN] Finalizado: {exitos}/{total_videos} operaciones completadas con éxito")
