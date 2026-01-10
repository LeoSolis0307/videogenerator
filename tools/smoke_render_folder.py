import os
import sys

                                                                  
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
                                            
    os.environ.setdefault("VIDEO_PRESET", "ultrafast")
    os.environ.setdefault("VIDEO_CRF", "28")
    os.environ.setdefault("VIDEO_CROSSFADE_MS", "0")
    os.environ.setdefault("ENABLE_LOUDNORM", "0")

    folder = sys.argv[1] if len(sys.argv) > 1 else r"output\custom_20260109_194537"

    from core.video_renderer import render_video_ffmpeg

    imgs = [
        os.path.join(folder, "seg_01_chosen.png"),
        os.path.join(folder, "seg_02_chosen.png"),
        os.path.join(folder, "seg_03_chosen.jpg"),
        os.path.join(folder, "seg_04_chosen.png"),
        os.path.join(folder, "seg_05_chosen.jpg"),
        os.path.join(folder, "seg_06_chosen.jpg"),
    ]
    audio = os.path.join(folder, "audio_con_silencios.wav")

    print("folder:", os.path.abspath(folder))
    print("audio :", os.path.abspath(audio))

    out = render_video_ffmpeg(imgs, audio, folder, durations=[0.5] * len(imgs))
    print("video_out:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
