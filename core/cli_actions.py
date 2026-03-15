import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from core import custom_video, reddit_scraper, topic_db
from utils import topic_file


@dataclass(frozen=True)
class CliActionContext:
    voz: str
    velocidad: str
    pedir_entero: Callable[..., int]
    pedir_texto_multiline: Callable[..., str | None]
    parse_indices_csv: Callable[..., list[int]]
    es_si: Callable[[str], bool]
    preguntar_tts_render: Callable[..., tuple[str, str]]
    windows_keep_awake_cls: type
    es_error_gpu_bloqueante: Callable[[Exception], bool]
    custom_plans_pendientes: Callable[[], list[str]]
    custom_plans_todos: Callable[[], list[str]]
    custom_plan_flags: Callable[[str], dict]
    finalizar_tema_custom_renderizado: Callable[[str], None]


def _filtrar_comentarios_reddit(comentarios: list[dict], limite: int = 200) -> list[tuple[str, str]]:
    filtrados: list[tuple[str, str]] = []
    for c in comentarios:
        if c.get("kind") != "t1":
            continue
        body = c.get("data", {}).get("body", "")
        cid = c.get("data", {}).get("id", "")
        if not body or "[deleted]" in body:
            continue
        if len(body) <= 80:
            continue
        if not reddit_scraper.es_historia_narrativa(body, min_chars=900):
            continue
        filtrados.append((body, cid))
        if len(filtrados) >= limite:
            break
    return filtrados


def _cargar_historias_reddit_importadas(carpeta: str = os.path.join("historias", "Reddit Virales")) -> list[dict[str, str | None]]:
    if not os.path.isdir(carpeta):
        return []

    out: list[dict[str, str | None]] = []
    for nombre in sorted(os.listdir(carpeta)):
        if nombre.startswith("0"):
            continue
        if not nombre.lower().endswith(".txt"):
            continue

        txt_path = os.path.join(carpeta, nombre)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                texto = (f.read() or "").strip()
        except Exception:
            continue

        if not texto:
            continue

        base = os.path.splitext(nombre)[0]
        meta_path = os.path.join(carpeta, f"{base}.json")
        comment_id = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                cid = (meta.get("comment_id") or "").strip()
                comment_id = cid or None
            except Exception:
                comment_id = None

        out.append(
            {
                "texto": texto,
                "txt_path": txt_path,
                "meta_path": meta_path if os.path.exists(meta_path) else None,
                "comment_id": comment_id,
            }
        )

    out.sort(key=lambda x: len(str(x.get("texto") or "")), reverse=True)
    return out


def _marcar_archivo_usado(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    dirname = os.path.dirname(path)
    base = os.path.basename(path)
    if base.startswith("0"):
        return path
    nuevo = os.path.join(dirname, "0" + base)
    while os.path.exists(nuevo):
        nuevo = os.path.join(dirname, "0" + os.path.basename(nuevo))
    os.replace(path, nuevo)
    return nuevo


def _marcar_historia_reddit_importada_usada(txt_path: str | None, meta_path: str | None) -> None:
    try:
        _marcar_archivo_usado(txt_path or "")
        if meta_path:
            _marcar_archivo_usado(meta_path)
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudo marcar historia importada como usada: {e}")


def run_action_4(ctx: CliActionContext) -> None:
    origen_temas = input(
        "Temas (1=escribir, 2=archivo storage/temas_custom.txt, 3=historias de Reddit nuevas) [1]: "
    ).strip()

    temas_path = topic_file.TOPICS_FILE_DEFAULT
    temas_disponibles: list[tuple[str, bool]] = []
    reddit_disponibles: list[tuple[str, str]] = []
    reddit_importadas: list[dict[str, str | None]] = []
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
        total_custom = ctx.pedir_entero(f"¿Cuántos videos sacar del archivo? (max {max_n}): ", minimo=1, default=1)
        total_custom = min(total_custom, max_n)
        temas_disponibles = elegibles
    elif origen_temas == "3":
        reddit_importadas = _cargar_historias_reddit_importadas()
        if reddit_importadas:
            print(f"[MAIN] Historias importadas disponibles en historias/Reddit Virales: {len(reddit_importadas)}")
            max_n = len(reddit_importadas)
            total_custom = ctx.pedir_entero(
                f"¿Cuántos videos sacar de historias importadas de Reddit? (max {max_n}): ",
                minimo=1,
                default=1,
            )
        else:
            print("[MAIN] No hay historias importadas disponibles. Buscando historias frescas en Reddit...")
            try:
                post = reddit_scraper.obtener_post()
            except Exception as e:
                print(f"[MAIN] ❌ No se pudo obtener post de Reddit: {e}")
                raise SystemExit(1)

            if not post:
                print("[MAIN] ❌ No se pudo obtener post de Reddit.")
                raise SystemExit(1)

            try:
                comentarios = reddit_scraper.obtener_comentarios(post.get("permalink", ""))
            except Exception as e:
                print(f"[MAIN] ❌ No se pudieron obtener comentarios de Reddit: {e}")
                raise SystemExit(1)

            reddit_disponibles = _filtrar_comentarios_reddit(comentarios, limite=200)
            if not reddit_disponibles:
                print("[MAIN] ❌ No se encontraron historias útiles en Reddit para convertir a video.")
                raise SystemExit(0)

            reddit_disponibles = sorted(reddit_disponibles, key=lambda t: len(t[0]), reverse=True)
            max_n = len(reddit_disponibles)
            total_custom = ctx.pedir_entero(f"¿Cuántos videos sacar de Reddit? (max {max_n}): ", minimo=1, default=1)
        total_custom = min(total_custom, max_n)
    else:
        total_custom = ctx.pedir_entero("¿Cuántos videos personalizados quieres crear?: ", minimo=1, default=1)

    dur_opt = input("Duración para TODOS (1=1-3 minutos, 2=5+ minutos, 3=debug 5 segundos) [1]: ").strip()
    if dur_opt == "2":
        min_seconds = 300
        max_seconds = None
    elif dur_opt == "3":
        min_seconds = 5
        max_seconds = 8
        print("[MAIN] 🧪 Modo debug rápido activado: objetivo ~5 segundos para validar pipeline.")
    else:
        min_seconds = 40
        max_seconds = 180
        print("[MAIN] ℹ️ Modo corto ajustado: mínimo 40 segundos y máximo 3 minutos.")

    sel_all = input("¿Quieres elegir manualmente las imágenes para TODOS los videos? (s/N): ").strip().lower()
    seleccionar_imagenes_all = sel_all == "s"

    auto_render_after_plans = ctx.es_si(input("¿Al terminar los guiones/planes, renderizar automáticamente? (s/N): "))
    unload_text_model_after_plans = False
    if auto_render_after_plans:
        unload_text_model_after_plans = ctx.es_si(
            input("¿Intentar descargar/unload el modelo de texto al terminar guiones? (s/N): ")
        )

    render_voz = ctx.voz
    render_velocidad = ctx.velocidad
    if auto_render_after_plans:
        render_voz, render_velocidad = ctx.preguntar_tts_render(
            voz_default=ctx.voz,
            velocidad_default=ctx.velocidad,
        )

    briefs: list[str] = []
    seleccionar_flags: list[bool] = []
    brief_topic_files: list[str | None] = []
    brief_reddit_ids: list[str | None] = []
    brief_reddit_txt_paths: list[str | None] = []
    brief_reddit_meta_paths: list[str | None] = []

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
        for tema, forced in temas_disponibles:
            if len(briefs) >= total_custom:
                break
            brief = (tema or "").strip()
            if not brief:
                continue
            if not forced and _tema_repetido(brief):
                continue

            seleccionar_imagenes = seleccionar_imagenes_all

            briefs.append(brief)
            seleccionar_flags.append(seleccionar_imagenes)
            brief_topic_files.append(temas_path)
            brief_reddit_ids.append(None)
            brief_reddit_txt_paths.append(None)
            brief_reddit_meta_paths.append(None)

        if len(briefs) < total_custom:
            print(
                f"[MAIN] ⚠️ Solo se pudieron tomar {len(briefs)}/{total_custom} temas del archivo "
                "(los demás se saltaron por repetidos)."
            )
            if not briefs:
                raise SystemExit(0)
    elif origen_temas == "3":
        if reddit_importadas:
            for item in reddit_importadas:
                if len(briefs) >= total_custom:
                    break

                texto_reddit = (item.get("texto") or "").strip()
                if not texto_reddit:
                    continue

                brief = (
                    "Convierte esta historia real de Reddit en un guion narrativo para video en español, "
                    "manteniendo el conflicto principal y los detalles clave:\n\n"
                    f"{texto_reddit}"
                )
                if _tema_repetido(brief):
                    continue

                seleccionar_imagenes = seleccionar_imagenes_all

                briefs.append(brief)
                seleccionar_flags.append(seleccionar_imagenes)
                brief_topic_files.append(None)
                brief_reddit_ids.append((item.get("comment_id") or "").strip() or None)
                brief_reddit_txt_paths.append((item.get("txt_path") or "").strip() or None)
                brief_reddit_meta_paths.append((item.get("meta_path") or "").strip() or None)
        else:
            for texto_reddit, cid in reddit_disponibles:
                if len(briefs) >= total_custom:
                    break

                brief = (
                    "Convierte esta historia real de Reddit en un guion narrativo para video en español, "
                    "manteniendo el conflicto principal y los detalles clave:\n\n"
                    f"{(texto_reddit or '').strip()}"
                )
                if _tema_repetido(brief):
                    continue

                seleccionar_imagenes = seleccionar_imagenes_all

                briefs.append(brief)
                seleccionar_flags.append(seleccionar_imagenes)
                brief_topic_files.append(None)
                brief_reddit_ids.append((cid or "").strip() or None)
                brief_reddit_txt_paths.append(None)
                brief_reddit_meta_paths.append(None)

        if len(briefs) < total_custom:
            print(
                f"[MAIN] ⚠️ Solo se pudieron tomar {len(briefs)}/{total_custom} historias de Reddit "
                "(algunas se saltaron por repetidas)."
            )
            if not briefs:
                raise SystemExit(0)
    else:
        print("[MAIN] Ingresa los prompts de cada historia (luego comenzará el render).")
        for i in range(1, total_custom + 1):
            while True:
                brief = ctx.pedir_texto_multiline(f"Prompt/Tema para la historia {i}/{total_custom}:")
                if brief is None:
                    print("[MAIN] ❌ Entrada finalizada (EOF) antes de capturar el prompt. Abortando para evitar bucle.")
                    raise SystemExit(1)
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
            brief_reddit_ids.append(None)
            brief_reddit_txt_paths.append(None)
            brief_reddit_meta_paths.append(None)

    print("\n[MAIN] Fase 1: generando SOLO guiones/planes (sin imágenes, sin render)...")

    if min_seconds <= 10:
        print("[MAIN] ℹ️ Modo debug 5s: se omite preflight del LLM de texto.")
    else:
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
    for i, (brief, seleccionar_imagenes, tema_file, reddit_id, reddit_txt_path, reddit_meta_path) in enumerate(
        zip(
            briefs,
            seleccionar_flags,
            brief_topic_files,
            brief_reddit_ids,
            brief_reddit_txt_paths,
            brief_reddit_meta_paths,
        ),
        start=1,
    ):
        try:
            carpeta = custom_video.generar_guion_personalizado_a_plan(
                brief,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
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
                        elif reddit_txt_path:
                            plan["topic_source"] = "reddit_imported"
                            plan["reddit_story_file"] = reddit_txt_path
                            if reddit_id:
                                plan["reddit_comment_id"] = reddit_id
                        elif reddit_id:
                            plan["topic_source"] = "reddit"
                            plan["reddit_comment_id"] = reddit_id
                        else:
                            plan["topic_source"] = "manual"
                        with open(plan_path, "w", encoding="utf-8") as f:
                            json.dump(plan, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[MAIN] ⚠️ No se pudo guardar metadata del tema: {e}")

                if reddit_txt_path:
                    _marcar_historia_reddit_importada_usada(reddit_txt_path, reddit_meta_path)

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
            with ctx.windows_keep_awake_cls(enabled=True):
                for k, ruta in enumerate(planes_creados, start=1):
                    try:
                        ok = custom_video.renderizar_video_personalizado_desde_plan(
                            ruta,
                            voz=render_voz,
                            velocidad=render_velocidad,
                            interactive=False,
                        )
                        if ok:
                            exitos += 1
                        else:
                            print(
                                "[MAIN] ❌ Render cancelado: falló la generación de voz (TTS) "
                                "o faltan audios válidos."
                            )
                        print(f"[MAIN] Render {k}/{len(planes_creados)}:", "OK" if ok else "FAIL")
                    except Exception as e:
                        if ctx.es_error_gpu_bloqueante(e):
                            print(f"[MAIN] ❌ Render detenido por requisito GPU: {e}")
                            raise SystemExit(1)
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


def run_action_5(ctx: CliActionContext) -> None:
    planes = ctx.custom_plans_pendientes()
    mostrando_todos = False
    if not planes:
        planes = ctx.custom_plans_todos()
        mostrando_todos = bool(planes)
    if planes:
        if mostrando_todos:
            print("\n[MAIN] No hay pendientes; mostrando todos los planes personalizados:")
        else:
            print("\n[MAIN] Planes personalizados disponibles:")
        for i, p in enumerate(planes, start=1):
            print(f"  {i}. {p}")

        total_renders = ctx.pedir_entero("¿Cuántos renders quieres hacer?: ", minimo=1, default=1)
        sel_raw = input("Selecciona planes (ej: 1,2,3) [1]: ").strip()
        seleccion = ctx.parse_indices_csv(sel_raw, max_index=len(planes))
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
        render_voz, render_velocidad = ctx.preguntar_tts_render(
            voz_default=ctx.voz,
            velocidad_default=ctx.velocidad,
        )

        try:
            with ctx.windows_keep_awake_cls(enabled=True):
                for k, idx in enumerate(seleccion, start=1):
                    ruta = planes[idx - 1]
                    try:
                        ok = custom_video.renderizar_video_personalizado_desde_plan(
                            ruta,
                            voz=render_voz,
                            velocidad=render_velocidad,
                            interactive=interactive_mode,
                        )
                        if ok:
                            exitos += 1
                        else:
                            print(
                                "[MAIN] ❌ Render cancelado: falló la generación de voz (TTS) "
                                "o faltan audios válidos."
                            )

                        if ok:
                            ctx.finalizar_tema_custom_renderizado(ruta)

                        print(
                            f"[MAIN] Render {k}/{len(seleccion)} (plan {idx}):",
                            "✅ Exito" if ok else "❌ Falló",
                        )
                    except Exception as e:
                        if ctx.es_error_gpu_bloqueante(e):
                            print(f"[MAIN] ❌ Render detenido por requisito GPU: {e}")
                            raise SystemExit(1)
                        print(f"[MAIN] ⚠️ Error renderizando (plan {idx}): {e}")
        except KeyboardInterrupt:
            print("\n[MAIN] Interrumpido por el usuario.")
            raise SystemExit(0)

        print(f"[MAIN] Renders completados: {exitos}/{len(seleccion)}")
        raise SystemExit(0)

    raw = input("Rutas (separadas por coma) de carpeta output/custom_... o custom_plan.json: ").strip().strip('"')
    if not raw:
        print("[MAIN] No se encontraron planes custom y no se indicó ruta")
        raise SystemExit(0)

    rutas = [r.strip().strip('"') for r in raw.split(",") if r.strip().strip('"')]
    total_renders = len(rutas)
    exitos = 0
    render_voz, render_velocidad = ctx.preguntar_tts_render(
        voz_default=ctx.voz,
        velocidad_default=ctx.velocidad,
    )
    for i, ruta in enumerate(rutas, start=1):
        try:
            ok = custom_video.renderizar_video_personalizado_desde_plan(
                ruta,
                voz=render_voz,
                velocidad=render_velocidad,
                interactive=(len(rutas) <= 1),
            )
            if ok:
                exitos += 1
                ruta_plan = os.path.abspath(ruta)
                plan_dir = ruta_plan if os.path.isdir(ruta_plan) else os.path.dirname(ruta_plan)
                ctx.finalizar_tema_custom_renderizado(plan_dir)
            else:
                print(
                    "[MAIN] ❌ Render cancelado: falló la generación de voz (TTS) "
                    "o faltan audios válidos."
                )
            print(f"[MAIN] Render {i}/{total_renders}:", "✅ Exito" if ok else "❌ Falló")
        except Exception as e:
            if ctx.es_error_gpu_bloqueante(e):
                print(f"[MAIN] ❌ Render detenido por requisito GPU: {e}")
                raise SystemExit(1)
            print(f"[MAIN] ⚠️ Error renderizando ({ruta}): {e}")

    print(f"[MAIN] Renders completados: {exitos}/{total_renders}")
    raise SystemExit(0)


def run_action_7(ctx: CliActionContext) -> None:
    planes = ctx.custom_plans_pendientes()
    if not planes:
        print("[MAIN] No hay planes personalizados pendientes para reanudar.")
        raise SystemExit(0)

    ruta = planes[0]
    print(f"[MAIN] Reanudando último plan personalizado pendiente: {ruta}")
    try:
        flags = ctx.custom_plan_flags(ruta)
        if flags.get("rendered"):
            ctx.finalizar_tema_custom_renderizado(ruta)
            print("[MAIN] Reanudación: ✅ Ya estaba renderizado; solo se finalizó.")
        else:
            with ctx.windows_keep_awake_cls(enabled=True):
                ok = custom_video.renderizar_video_personalizado_desde_plan(
                    ruta,
                    voz=ctx.voz,
                    velocidad=ctx.velocidad,
                    interactive=False,
                )
            if ok:
                ctx.finalizar_tema_custom_renderizado(ruta)
            else:
                print(
                    "[MAIN] ❌ Reanudación cancelada: falló la generación de voz (TTS) "
                    "o faltan audios válidos."
                )
            print("[MAIN] Reanudación:", "✅ Exito" if ok else "❌ Falló")
    except KeyboardInterrupt:
        print("\n[MAIN] Interrumpido por el usuario.")
        raise SystemExit(0)
    except Exception as e:
        if ctx.es_error_gpu_bloqueante(e):
            print(f"[MAIN] ❌ Render detenido por requisito GPU: {e}")
            raise SystemExit(1)
        print(f"[MAIN] ⚠️ Error reanudando último plan: {e}")
        raise SystemExit(0)

    raise SystemExit(0)


def run_action_8(ctx: CliActionContext) -> None:
    planes = ctx.custom_plans_pendientes()
    if not planes:
        print("[MAIN] No hay planes personalizados pendientes.")
        raise SystemExit(0)

    print(f"[MAIN] Reanudando cola de personalizados pendientes: {len(planes)}")
    exitos = 0
    try:
        with ctx.windows_keep_awake_cls(enabled=True):
            for i, ruta in enumerate(planes, start=1):
                try:
                    flags = ctx.custom_plan_flags(ruta)
                    if flags.get("rendered"):
                        ctx.finalizar_tema_custom_renderizado(ruta)
                        exitos += 1
                        print(f"[MAIN] {i}/{len(planes)}: ✅ Ya renderizado; finalizado: {ruta}")
                        continue

                    ok = custom_video.renderizar_video_personalizado_desde_plan(
                        ruta,
                        voz=ctx.voz,
                        velocidad=ctx.velocidad,
                        interactive=False,
                    )
                    if ok:
                        ctx.finalizar_tema_custom_renderizado(ruta)
                        exitos += 1
                    else:
                        print(
                            "[MAIN] ❌ Item cancelado: falló la generación de voz (TTS) "
                            "o faltan audios válidos."
                        )
                    print(f"[MAIN] {i}/{len(planes)}:", "✅ Exito" if ok else "❌ Falló", f"| {ruta}")
                except Exception as e:
                    if ctx.es_error_gpu_bloqueante(e):
                        print(f"[MAIN] ❌ Render detenido por requisito GPU: {e}")
                        raise SystemExit(1)
                    print(f"[MAIN] ⚠️ Error en cola (item {i}/{len(planes)}): {e}")
    except KeyboardInterrupt:
        print("\n[MAIN] Interrumpido por el usuario.")
        raise SystemExit(0)

    print(f"[MAIN] Cola completada: {exitos}/{len(planes)}")
    raise SystemExit(0)


def run_action_11(ctx: CliActionContext) -> None:
    planes = ctx.custom_plans_pendientes()
    mostrando_todos = False
    if not planes:
        planes = ctx.custom_plans_todos()
        mostrando_todos = bool(planes)

    if not planes:
        print("[MAIN] No hay planes personalizados para mejorar.")
        raise SystemExit(0)

    if mostrando_todos:
        print("\n[MAIN] No hay pendientes; mostrando todos los planes personalizados para mejorar:")
    else:
        print("\n[MAIN] Planes personalizados pendientes para mejorar:")

    for i, p in enumerate(planes, start=1):
        print(f"  {i}. {p}")

    idx_plan = ctx.pedir_entero("¿Qué plan quieres mejorar? (número): ", minimo=1, default=1)
    if idx_plan > len(planes):
        idx_plan = len(planes)

    pasadas = ctx.pedir_entero("¿Cuántas pasadas de mejora?: ", minimo=1, default=1)
    try:
        pasadas = min(int(pasadas), 10)
    except Exception:
        pasadas = 1

    def _brief_mejorado_desde_plan(plan: dict) -> str:
        brief = str(plan.get("brief") or "").strip()
        title = str(plan.get("title_es") or "").strip()
        hook = str(plan.get("hook_es") or "").strip()
        script = str(plan.get("script_es") or "").strip()

        base = brief
        if not base:
            base = title or hook or script[:300]
        if not base:
            base = "Historia impactante para video narrado en español"

        if script:
            script = script[:2500]

        extra = (
            "\n\nMejora este plan para maximizar retención en YouTube Shorts: "
            "hook inicial más fuerte, mejor ritmo, más claridad y cierre memorable, "
            "sin inventar datos verificables."
        )
        if script:
            extra += f"\n\nContexto del guion actual (resumen/base):\n{script}"
        return (base + extra).strip()

    origen_inicial = planes[idx_plan - 1]
    ruta_actual = origen_inicial
    creados = 0

    for pasada in range(1, pasadas + 1):
        plan_path = os.path.join(ruta_actual, "custom_plan.json")
        try:
            if not os.path.exists(plan_path):
                print(f"[MAIN] ⚠️ No existe custom_plan.json en: {ruta_actual}")
                break

            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f) or {}

            brief_mejorado = _brief_mejorado_desde_plan(plan)

            min_seconds = plan.get("min_seconds")
            max_seconds = plan.get("max_seconds")
            seleccionar_imagenes = bool(plan.get("seleccionar_imagenes"))

            try:
                min_seconds = int(min_seconds) if min_seconds is not None else None
            except Exception:
                min_seconds = None
            try:
                max_seconds = int(max_seconds) if max_seconds is not None else None
            except Exception:
                max_seconds = None

            nueva_ruta = custom_video.generar_guion_personalizado_a_plan(
                brief_mejorado,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                seleccionar_imagenes=seleccionar_imagenes,
            )
            if not nueva_ruta:
                print(f"[MAIN] ❌ No se pudo mejorar en pasada {pasada}/{pasadas}")
                break

            try:
                nuevo_plan_path = os.path.join(nueva_ruta, "custom_plan.json")
                if os.path.exists(nuevo_plan_path):
                    with open(nuevo_plan_path, "r", encoding="utf-8") as f:
                        nuevo = json.load(f) or {}
                    nuevo["improved_from_plan_dir"] = os.path.abspath(ruta_actual)
                    nuevo["improved_root_plan_dir"] = os.path.abspath(origen_inicial)
                    nuevo["improvement_pass"] = int(pasada)
                    nuevo["improved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(nuevo_plan_path, "w", encoding="utf-8") as f:
                        json.dump(nuevo, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[MAIN] ⚠️ Plan mejorado creado, pero no se guardó metadata: {e}")

            ruta_actual = nueva_ruta
            creados += 1
            print(f"[MAIN] ✅ Pasada {pasada}/{pasadas} creada: {nueva_ruta}")
        except Exception as e:
            print(f"[MAIN] ⚠️ Error en pasada {pasada}/{pasadas}: {e}")
            break

    print(f"[MAIN] Mejora completada: {creados}/{pasadas}")
    if creados > 0:
        print(f"[MAIN] Último plan mejorado: {ruta_actual}")
    print("[MAIN] Usa la opción 5 para renderizar los planes mejorados cuando quieras.")
    raise SystemExit(0)
