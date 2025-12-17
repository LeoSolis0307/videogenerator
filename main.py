import hashlib
import os

from core import image_downloader, reddit_scraper, text_processor, tts
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


                                                   
                                                      
VOZ = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0"
VELOCIDAD = "-17%"


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


def main():
	print("[MAIN] Iniciando proceso")

	modo = input("Selecciona modo (1 = IA/imagenes, 2 = Video base): ").strip()
	usar_video_base = modo == "2"

	carpeta = crear_carpeta_proyecto()

	video_base_path = None
	video_base_dur = 0.0
	if usar_video_base:
		video_base_path, video_base_dur = select_video_base(None)
		if not video_base_path:
			print("[MAIN] No se encontró video base utilizable")
			return
		if video_base_dur <= 0:
			print("[MAIN] No se pudo detectar duración del video base. Abortando.")
			return
		print(f"[MAIN] Video base seleccionado: {video_base_path} (~{video_base_dur:.1f}s)")

	post = reddit_scraper.obtener_post()
	if not post:
		print("[MAIN] No se pudo obtener post")
		return

	comentarios = reddit_scraper.obtener_comentarios(post["permalink"])
	comentarios_filtrados = _filtrar_comentarios(comentarios, limite=200)

	if usar_video_base and video_base_dur > 0:
		gap = 4.0
		target_min = video_base_dur * 0.85
		target_max = video_base_dur * 1.02
		over_shoot = target_max * 2.50                                                            

                                                                   
		comentarios_filtrados = sorted(comentarios_filtrados, key=lambda t: len(t[0]), reverse=True)
		seleccion = []
		estims = []
		acumulado = 0.0
		for txt, cid in comentarios_filtrados:
			est = _estimar_segundos(txt)
			if est > over_shoot:
				continue                                             
			extra_gap = gap if seleccion else 0.0
                                                                   
			if acumulado < target_min:
				acumulado += extra_gap + est
				seleccion.append((txt, cid))
				estims.append(est)
				continue
                                                                          
			if acumulado + extra_gap + est <= over_shoot:
				acumulado += extra_gap + est
				seleccion.append((txt, cid))
				estims.append(est)
			else:
				break

                                                                                                     
		while seleccion and acumulado > target_max:
			est = estims.pop() if estims else 0.0
			seleccion.pop()
			gap_remove = gap if seleccion else 0.0
			acumulado -= (gap_remove + est)
			if acumulado >= target_min:
				break

                                                                                      
		if not seleccion:
			acumulado = 0.0
			estims = []
			seleccion = []
			for txt, cid in sorted(comentarios_filtrados, key=lambda t: len(t[0])):
				est = _estimar_segundos(txt)
				if est > over_shoot:
					continue
				extra_gap = gap if seleccion else 0.0
				if acumulado + extra_gap + est <= target_max:
					acumulado += extra_gap + est
					seleccion.append((txt, cid))
					estims.append(est)
					if acumulado >= target_min:
						break

                                                                                                        
		if acumulado < target_min:
			existentes = set((t, c) for t, c in seleccion)
			for txt, cid in sorted(comentarios_filtrados, key=lambda t: len(t[0])):
				if (txt, cid) in existentes:
					continue
				est = _estimar_segundos(txt)
				if est > over_shoot:
					continue
				extra_gap = gap if seleccion else 0.0
				if acumulado + extra_gap + est <= target_max:
					acumulado += extra_gap + est
					seleccion.append((txt, cid))
					estims.append(est)
					existentes.add((txt, cid))
					if acumulado >= target_min:
						break

		if not seleccion:
			print("[MAIN] No hay historias suficientes para el video base.")
			return

		if acumulado < target_min:
			print(
			 f"[MAIN] Aviso: estimado corto ({acumulado:.1f}s < {target_min:.1f}s); se continúa y se validará con audios reales."
			)

		comentarios_filtrados = seleccion
		print(
		 f"[MAIN] Historias seleccionadas (estimado) {len(comentarios_filtrados)} en ~{acumulado:.1f}s para video de {video_base_dur:.1f}s"
		)

	textos_en = [txt for (txt, _cid) in comentarios_filtrados]
	ids_seleccionados = [cid for (_txt, cid) in comentarios_filtrados]

	if not textos_en:
		print("[MAIN] No hay textos suficientes")
		return

	print(f"[MAIN] {len(textos_en)} textos obtenidos")

	textos_es = text_processor.traducir_lista(textos_en)

	print("[DEBUG] Primer texto que irá al TTS:")
	print(textos_es[0][:200])

	audios = tts.generar_audios(textos_es, carpeta, voz=VOZ, velocidad=VELOCIDAD)

	if not audios:
		print("[MAIN] No se generaron audios")
		return

	if usar_video_base:
		gap = 4.0
		target_min = video_base_dur * 0.85
		target_max = video_base_dur * 1.02

		duraciones = [audio_duration_seconds(a) for a in audios]
		seleccion_audios = []
		seleccion_textos = []
		seleccion_ids = []
		acumulado = 0.0
		for a, d, t, cid in zip(audios, duraciones, textos_es, ids_seleccionados):
			if d <= 0:
				continue
			if d > target_max:
				print(f"[MAIN] Saltando audio que excede el máximo ({d:.1f}s > {target_max:.1f}s)")
				continue
			extra_gap = gap if seleccion_audios else 0.0
			proximo = acumulado + extra_gap + d
			if proximo <= target_max:
				acumulado = proximo
				seleccion_audios.append(a)
				seleccion_textos.append(t)
				seleccion_ids.append(cid)
			else:
				break

		if not seleccion_audios:
			print("[MAIN] Los audios generados no caben en la duración del video base. Abortando.")
			return

		if acumulado < target_min:
			faltante = target_min - acumulado
                                                                 
			pad = min(faltante, 20.0)
			if pad > 0.1:
				silencio = _crear_silencio(pad, carpeta)
				seleccion_audios = [silencio] + seleccion_audios
				seleccion_textos = ["(silencio)"] + seleccion_textos
				seleccion_ids = [""] + seleccion_ids
				total_audio = sum(audio_duration_seconds(a) for a in seleccion_audios)
				total_gap = gap * (len(seleccion_audios) - 1) if len(seleccion_audios) > 1 else 0.0
				acumulado = total_audio + total_gap
			print(f"[MAIN] Audio corto: se agregó {pad:.1f}s de silencio inicial (total estimado ~{acumulado:.1f}s)")
			if acumulado < target_min:
				print(f"[MAIN] Audio total ({acumulado:.1f}s) aún queda por debajo del 85% del video ({target_min:.1f}s). Abortando.")
				return

		audios = seleccion_audios
		textos_es = seleccion_textos
		ids_seleccionados = seleccion_ids
		print(
		 f"[MAIN] Historias recortadas a {len(audios)} para caber en ~{video_base_dur:.1f}s (audio real ~{acumulado:.1f}s)"
		)

		try:
			audio_final = combine_audios_with_silence(
			 audios,
			 carpeta,
			 gap_seconds=gap,
			 min_seconds=None,
			 max_seconds=target_max,
			)
			audio_final_dur = audio_duration_seconds(audio_final)
			if audio_final_dur <= 0 or audio_final_dur < target_min or audio_final_dur > target_max:
				print(
				 f"[MAIN] Audio final fuera de rango ({audio_final_dur:.2f}s). "
				 f"Debe estar entre {target_min:.2f}s y {target_max:.2f}s. Abortando."
				)
				return
			video_final = render_video_base_con_audio(video_base_path, audio_final, carpeta, videos_dir=None)
		except Exception as e:
			print(f"[MAIN] Error renderizando con video base: {e}")
			return
	else:
		audio_final = combine_audios_with_silence(audios, carpeta, gap_seconds=4)

		imagenes = image_downloader.descargar_imagenes(carpeta, 10)

		if not imagenes:
			print("[MAIN] No se descargaron imágenes")
			return

		cortos_dir = os.path.join(carpeta, "cortos")
		for idx, (audio_path, texto) in enumerate(zip(audios, textos_es)):
			img = imagenes[idx % len(imagenes)]
			render_story_clip(audio_path, img, cortos_dir, title_text=texto)

		video_final = render_video_ffmpeg(imagenes, audio_final, carpeta)

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

	print("[MAIN] ✅ Proceso finalizado")


if __name__ == "__main__":
	main()
