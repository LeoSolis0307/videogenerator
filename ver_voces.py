import asyncio

import edge_tts
import pyttsx3


def listar_voces_locales():
    engine = pyttsx3.init()
    voces = engine.getProperty("voices")

    print("--- VOCES INSTALADAS EN TU WINDOWS (SAPI5) ---")
    for index, voz in enumerate(voces):
        print(f"\nOpción {index}:")
        print(f" - Nombre: {voz.name}")
        print(f" - ID: {voz.id}")
        if "Mexico" in voz.name or "Jorge" in voz.name:
            print("   ¡¡¡ ESTA PARECE SER LA CORRECTA !!! <--- COPIA EL ID")


async def listar_voces_edge(prefijo_idioma="es"):
    voces = await edge_tts.list_voices()
    if prefijo_idioma:
        voces = [v for v in voces if v["Locale"].lower().startswith(prefijo_idioma.lower())]

    print("\n--- VOCES DISPONIBLES EN edge-tts (Azure/Edge) ---")
    for index, voz in enumerate(voces):
        print(f"\nOpción {index}:")
        print(f" - Nombre corto: {voz['ShortName']}")
        print(f" - Idioma: {voz['Locale']}")
        print(f" - Género: {voz.get('Gender', 'N/D')}")


if __name__ == "__main__":
    listar_voces_locales()
    try:
        asyncio.run(listar_voces_edge())
    except Exception as exc:
        print(f"No se pudieron listar las voces de edge-tts: {exc}")