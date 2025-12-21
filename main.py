import hashlib
import json
import math
import os
import re
import shutil

from core import custom_video, image_downloader, reddit_scraper, story_generator, text_processor, tts
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


                                                   
                                                                                               
VOZ = "es-MX-JorgeNeural"
VELOCIDAD = "-15%"                          

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


def _planes_pendientes() -> list[str]:
    \
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
    \
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
        if os.path.exists(plan_file):
            pendientes.append(entry.path)
    pendientes.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pendientes


def _marcar_plan_completado(path: str):
    \
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
    \
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
    \
\
\
\
\
\
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
    accion = input("¿Qué deseas hacer? (1 = Videos, 2 = Textos, 3 = Imagen de prueba, 4 = Video personalizado, 5 = Renderizar personalizado): ").strip()

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
        brief = input("Tema/brief para video personalizado (usar imágenes reales de internet): ").strip()
        dur_opt = input("Duración mínima (1=1 minuto, 2=5 minutos) [1]: ").strip()
        min_seconds = 60 if dur_opt != "2" else 300
        sel_opt = input("¿Quieres elegir manualmente las imágenes antes del render? (s/N): ").strip().lower()
        seleccionar_imagenes = sel_opt == "s"
        try:
            ok = custom_video.generar_video_personalizado(
                brief,
                voz=VOZ,
                velocidad=VELOCIDAD,
                min_seconds=min_seconds,
                seleccionar_imagenes=seleccionar_imagenes,
            )
            print("[MAIN] Resultado:", "✅ Exito" if ok else "❌ Falló")
        except Exception as e:
            print(f"[MAIN] ⚠️ Error en video personalizado: {e}")
        raise SystemExit(0)

    if accion == "5":
        planes = _custom_plans_pendientes()
        if planes:
            print("\n[MAIN] Planes personalizados disponibles:")
            for i, p in enumerate(planes, start=1):
                print(f"  {i}. {p}")
            idx = _pedir_entero("Selecciona número de plan: ", minimo=1, default=1)
            idx = max(1, min(idx, len(planes)))
            ruta = planes[idx - 1]
        else:
            ruta = input("Ruta de carpeta (output/custom_... ) o archivo custom_plan.json: ").strip().strip('"')
            if not ruta:
                print("[MAIN] No se indicó ruta")
                raise SystemExit(0)

        try:
            ok = custom_video.renderizar_video_personalizado_desde_plan(ruta, voz=VOZ, velocidad=VELOCIDAD)
            print("[MAIN] Resultado:", "✅ Exito" if ok else "❌ Falló")
        except Exception as e:
            print(f"[MAIN] ⚠️ Error renderizando: {e}")
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
