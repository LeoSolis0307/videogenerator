import subprocess
import os

def generar_audios(textos, carpeta, voz=None, velocidad=None):
    print("[TTS] Usando TTS nativo de Windows (SAPI)")
    rutas = []

    for i, texto in enumerate(textos):
        ruta = os.path.join(carpeta, f"audio_{i}.wav")
        print(f"[TTS] Generando audio {i+1}/{len(textos)}")

                       
        texto = texto.replace('"', "'")

        comando = [
            "powershell",
            "-Command",
            f'''
Add-Type -AssemblyName System.Speech;
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$speak.Rate = 0;
$speak.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet,
                           [System.Speech.Synthesis.VoiceAge]::Adult,
                           0,
                           [cultureinfo]"es-MX");
$speak.SetOutputToWaveFile("{ruta}");
$speak.Speak("{texto}");
$speak.Dispose();
'''
        ]

        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rutas.append(ruta)

    print("[TTS] Audios generados con SAPI")
    return rutas
