import edge_tts
import os
import asyncio

VOICE = "es-MX-JorgeNeural"
RATE = "-15%"

async def texto_a_audio(textos, carpeta):
    rutas = []
    for i, txt in enumerate(textos):
        ruta = os.path.join(carpeta, f"audio_{i}.mp3")
        comm = edge_tts.Communicate(txt, VOICE, rate=RATE)
        await comm.save(ruta)
        rutas.append(ruta)
    return rutas
