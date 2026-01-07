import asyncio
import os
import random
import re
import subprocess
import sys
import tempfile
import wave

import pyttsx3

try:
    import edge_tts
except Exception:                    
    edge_tts = None


                       
                                                    
                                                                   
VOZ_ID_LOCAL_POR_DEFECTO = (
    r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0"
)

                                                                                                 
VELOCIDAD_WPM_POR_DEFECTO = 200

                                          
VELOCIDAD_EDGE = "-20%"

                                                    
                                                                                                                
                                                                
EDGE_TTS_STYLE = (os.environ.get("EDGE_TTS_STYLE") or "").strip()
EDGE_TTS_STYLE_DEGREE = (os.environ.get("EDGE_TTS_STYLE_DEGREE") or "").strip()


def _parece_voz_edge(voz: str | None) -> bool:
    if not voz:
        return False
                                                           
    return "neural" in voz.lower()


def _velocidad_local_a_wpm(velocidad) -> int:
    \
\
\
\
\
\
\
    if isinstance(velocidad, int):
        return max(50, min(400, velocidad))

    if isinstance(velocidad, str):
        v = velocidad.strip()
        if v.isdigit():
            return max(50, min(400, int(v)))

        if v.endswith("%"):
            try:
                pct = float(v[:-1])
                                                  
                factor = 1.0 + (pct / 100.0)
                wpm = int(round(VELOCIDAD_WPM_POR_DEFECTO * factor))
                return max(50, min(400, wpm))
            except ValueError:
                pass

    return VELOCIDAD_WPM_POR_DEFECTO


def _normalizar_id_voz(valor: str | None) -> str:
    if not valor:
        return ""
    return " ".join(str(valor).split()).lower()


def _split_text(texto: str, max_chars: int = 500) -> list[str]:
    texto = (texto or "").strip()
    if not texto:
        return [""]
    if len(texto) <= max_chars:
        return [texto]

    parts = re.split(r"(?<=[\.!\?;:])\s+", texto)
    chunks: list[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not buf:
            buf = p
            continue
        if len(buf) + 1 + len(p) <= max_chars:
            buf = buf + " " + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    final: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i : i + max_chars])
    return final


def _xml_escape(s: str) -> str:
    s = (s or "")
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&apos;")
    return s


def _build_edge_ssml(*, text: str, voice: str, style: str, style_degree: str | None = None) -> str:
    \
\
\
\
    v = (voice or "").strip()
    st = (style or "").strip()
    deg = (style_degree or "").strip()
    if not v or not st:
        return (text or "").strip()

                                                                    
    lang = "es-MX"
    try:
        if "-" in v:
            parts = v.split("-")
            if len(parts) >= 2:
                lang = parts[0] + "-" + parts[1]
    except Exception:
        pass

    escaped = _xml_escape((text or "").strip())
    attrs = f' style="{_xml_escape(st)}"'
    if deg:
                                                                           
        attrs += f' styledegree="{_xml_escape(deg)}"'

    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="{_xml_escape(lang)}">'
        f'<voice name="{_xml_escape(v)}">'
        f'<mstts:express-as{attrs}>'
        f'{escaped}'
        f'</mstts:express-as>'
        f'</voice>'
        f'</speak>'
    )


def _concat_wavs(input_wavs: list[str], output_wav: str) -> None:
    if not input_wavs:
        return
    with wave.open(input_wavs[0], "rb") as w0:
        params = w0.getparams()

    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    with wave.open(output_wav, "wb") as out:
        out.setparams(params)
        for p in input_wavs:
            with wave.open(p, "rb") as wi:
                out.writeframes(wi.readframes(wi.getnframes()))


def _render_wav_subprocess(*, texto: str, ruta_wav: str, voz_id: str | None, rate_wpm: int, timeout_s: int) -> bool:
    \
    worker = os.path.join(os.path.dirname(__file__), "tts_worker.py")
    if not os.path.exists(worker):
        raise FileNotFoundError(worker)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as f:
        text_path = f.name
        f.write(texto)

    try:
        cmd = [
            sys.executable,
            worker,
            "--text-file",
            text_path,
            "--out-wav",
            ruta_wav,
            "--rate",
            str(rate_wpm),
        ]
        if voz_id:
            cmd += ["--voice", voz_id]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            if err:
                print(f"[TTS-LOCAL] ⚠️ tts_worker falló: {err[:300]}")
            return False
        return os.path.exists(ruta_wav) and os.path.getsize(ruta_wav) > 0
    except subprocess.TimeoutExpired:
        print(f"[TTS-LOCAL] ⚠️ Timeout generando WAV ({timeout_s}s).")
        return False
    finally:
        try:
            os.remove(text_path)
        except Exception:
            pass


def _seleccionar_voz_pyttsx3(engine, voz_solicitada: str | None):
    \
\
\
\
    voces = engine.getProperty("voices") or []
    if not voces:
        return None, None

                                                                      
    if voz_solicitada:
        for v in voces:
            if v.id == voz_solicitada:
                engine.setProperty("voice", v.id)
                return getattr(v, "name", None), v.id
                                                                           
        voz_lower = voz_solicitada.lower()
        for v in voces:
            nombre = (getattr(v, "name", "") or "").lower()
            vid = (getattr(v, "id", "") or "").lower()
            if voz_lower in nombre or voz_lower in vid:
                engine.setProperty("voice", v.id)
                return getattr(v, "name", None), v.id

                                                      
    candidatos = []
    for v in voces:
        nombre = (getattr(v, "name", "") or "").lower()
        vid = (getattr(v, "id", "") or "").lower()
        score = 0
        tiene_jorge = "jorge" in nombre or "jorge" in vid
        tiene_mex = (
            "mex" in nombre
            or "mex" in vid
            or "es-mx" in nombre
            or "es-mx" in vid
        )

        if tiene_jorge:
            score += 10
        if tiene_mex:
            score += 5

                                                                                                  
        if (tiene_jorge or tiene_mex) and ("span" in nombre or "espa" in nombre):
            score += 2

        if score:
            candidatos.append((score, v))

    if candidatos:
        candidatos.sort(key=lambda t: t[0], reverse=True)
        elegido = candidatos[0][1]
        engine.setProperty("voice", elegido.id)
        return getattr(elegido, "name", None), elegido.id

    return None, None


async def _generar_audios_edge(textos, carpeta, voz, velocidad):
    archivos_audio = []
    style = (EDGE_TTS_STYLE or "").strip()
    deg = (EDGE_TTS_STYLE_DEGREE or "").strip()
    if style:
        extra = f" style={style}" + (f" degree={deg}" if deg else "")
    else:
        extra = ""
    print(f"[TTS-EDGE] Generando {len(textos)} audios con {voz} (rate {velocidad}){extra}...")

    for i, texto in enumerate(textos):
        nombre_archivo = f"audio_{i}.mp3"
        ruta_completa = os.path.join(carpeta, nombre_archivo)

        texto_in = (texto or "").strip()
        if style and texto_in:
                                                                                                            
            ssml = _build_edge_ssml(text=texto_in, voice=voz, style=style, style_degree=deg)
        else:
            ssml = texto_in

        for intento in range(3):
            try:
                await asyncio.sleep(random.uniform(2.0, 4.0))
                communicate = edge_tts.Communicate(ssml, voz, rate=velocidad)
                await communicate.save(ruta_completa)
                if os.path.exists(ruta_completa) and os.path.getsize(ruta_completa) > 0:
                    archivos_audio.append(ruta_completa)
                    print(f"   - Audio {i+1}/{len(textos)} generado.")
                    break
            except Exception as e:
                print(f"   ⚠️ Fallo en audio {i} (Intento {intento+1}/3): {e}")
                                                                           
                if style and intento == 0:
                    try:
                        await asyncio.sleep(1.0)
                        communicate = edge_tts.Communicate(texto_in, voz, rate=velocidad)
                        await communicate.save(ruta_completa)
                        if os.path.exists(ruta_completa) and os.path.getsize(ruta_completa) > 0:
                            archivos_audio.append(ruta_completa)
                            print(f"   - Audio {i+1}/{len(textos)} generado (sin estilo por fallback).")
                            break
                    except Exception:
                        pass
                if "403" in str(e):
                    print("   🛑 Bloqueo detectado. Esperando 20 segundos...")
                    if intento < 2:
                        await asyncio.sleep(20)
                else:
                    if intento < 2:
                        await asyncio.sleep(2)

    return archivos_audio


def generar_audios(textos, carpeta, voz=None, velocidad=None):
\
\
\
\
\
                        
    if _parece_voz_edge(voz):
        if edge_tts is None:
            print("❌ edge-tts no está disponible en este entorno.")
            return []
        voz_edge = voz
        vel_edge = velocidad if isinstance(velocidad, str) and "%" in velocidad else VELOCIDAD_EDGE
        try:
            return asyncio.run(_generar_audios_edge(textos, carpeta, voz_edge, vel_edge))
        except Exception as e:
            print(f"Error crítico en TTS-EDGE: {e}")
            return []

                                                                                                        
    rutas_generadas = []
    print("[TTS-LOCAL] Iniciando TTS local robusto (subproceso por audio)...")

    try:
        voz_solicitada = voz if isinstance(voz, str) and voz.strip() else VOZ_ID_LOCAL_POR_DEFECTO
        vel_local = _velocidad_local_a_wpm(velocidad)
        print(f"[TTS-LOCAL] Voz solicitada: {voz_solicitada}")
        print(f"[TTS-LOCAL] Velocidad (WPM): {vel_local}")

        total = len(textos)
        for idx, texto in enumerate(textos):
            texto = (textos[idx] or "").strip().replace("\u200b", " ")
            ruta = os.path.join(carpeta, f"audio_{idx}.wav")
            print(f"   - Renderizando audio {idx+1}/{total} (chars={len(texto)})")

            timeout_s = int(max(45, min(240, 20 + (len(texto) / 18))))
            ok = _render_wav_subprocess(
                texto=texto,
                ruta_wav=ruta,
                voz_id=voz_solicitada,
                rate_wpm=vel_local,
                timeout_s=timeout_s,
            )

            if ok:
                rutas_generadas.append(ruta)
                continue

            chunks = _split_text(texto, max_chars=450)
            if len(chunks) == 1:
                print("[TTS-LOCAL] ❌ Falló este audio incluso sin split.")
                continue

            print(f"[TTS-LOCAL] Reintentando con split en {len(chunks)} partes...")
            temp_paths: list[str] = []
            for j, ch in enumerate(chunks):
                tmp = os.path.join(carpeta, f"audio_{idx}_part{j}.wav")
                temp_paths.append(tmp)
                ok_part = _render_wav_subprocess(
                    texto=ch,
                    ruta_wav=tmp,
                    voz_id=voz_solicitada,
                    rate_wpm=vel_local,
                    timeout_s=90,
                )
                if not ok_part:
                    print(f"[TTS-LOCAL] ❌ Falló parte {j+1}/{len(chunks)}")
                    temp_paths = []
                    break

            if temp_paths:
                _concat_wavs(temp_paths, ruta)
                for p in temp_paths:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                if os.path.exists(ruta) and os.path.getsize(ruta) > 0:
                    rutas_generadas.append(ruta)

        print(f"[TTS-LOCAL] ✅ Terminados {len(rutas_generadas)} audios correctamente.")
        return rutas_generadas
    except Exception as e:
        print(f"❌ Error en motor local: {e}")
        return []