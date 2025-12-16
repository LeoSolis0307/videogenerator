import argparse
import os
import sys

import pyttsx3


def main() -> int:
    parser = argparse.ArgumentParser(description="Renderiza un WAV con pyttsx3 (SAPI5) en un proceso aislado.")
    parser.add_argument("--text-file", required=True, help="Ruta a archivo .txt (utf-8) con el texto")
    parser.add_argument("--out-wav", required=True, help="Ruta de salida .wav")
    parser.add_argument("--voice", default=None, help="ID de voz SAPI5 (HKEY_...) o substring")
    parser.add_argument("--rate", type=int, default=200, help="Velocidad (WPM)")
    parser.add_argument("--volume", type=float, default=1.0, help="Volumen 0.0-1.0")
    args = parser.parse_args()

    if not os.path.exists(args.text_file):
        print(f"[tts_worker] No existe text-file: {args.text_file}", file=sys.stderr)
        return 2

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_wav)), exist_ok=True)

    engine = pyttsx3.init()

                                    
    if args.voice:
        try:
            engine.setProperty("voice", args.voice)
        except Exception:
                                   
            try:
                voices = engine.getProperty("voices") or []
                vlow = args.voice.lower()
                for v in voices:
                    name = (getattr(v, "name", "") or "").lower()
                    vid = (getattr(v, "id", "") or "").lower()
                    if vlow in name or vlow in vid:
                        engine.setProperty("voice", v.id)
                        break
            except Exception:
                pass

    engine.setProperty("rate", max(50, min(400, int(args.rate))))
    engine.setProperty("volume", max(0.0, min(1.0, float(args.volume))))

    try:
        engine.save_to_file(text, args.out_wav)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[tts_worker] Error: {e}", file=sys.stderr)
        return 1

    if not os.path.exists(args.out_wav) or os.path.getsize(args.out_wav) == 0:
        print("[tts_worker] No se generó WAV o quedó vacío", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
