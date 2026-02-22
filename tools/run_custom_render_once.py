import os
import sys
from pathlib import Path

                                                       
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import custom_video
from core.config import settings


def main() -> int:
    plan_dir = os.environ.get("CUSTOM_PLAN_DIR") or str(ROOT / "output" / "custom_20260107_212105")

    voz = os.environ.get("CUSTOM_VOICE") or settings.custom_voice or "es-MX-JorgeNeural"
    velocidad = os.environ.get("CUSTOM_SPEED") or "-10%"

    ok = custom_video.renderizar_video_personalizado_desde_plan(
        plan_dir,
        voz=voz,
        velocidad=velocidad,
        interactive=False,
    )
    print("OK" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
