import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import custom_video


def main() -> int:
    try:
        brief = pathlib.Path("tmp_story_full_user.txt").read_text(encoding="utf-8")
        plan_dir = custom_video.generar_guion_personalizado_a_plan(
            brief,
            min_seconds=60,
            max_seconds=180,
            seleccionar_imagenes=False,
        )
        print("PLAN_DIR", plan_dir)
        if not plan_dir:
            print("ERROR RuntimeError plan_dir None")
            return 2

        ok = custom_video.renderizar_video_personalizado_desde_plan(
            plan_dir,
            voz="es-MX-JorgeNeural",
            velocidad="-10%",
            interactive=False,
        )
        print("RENDER_OK", ok)
        return 0 if ok else 3
    except Exception as exc:
        print("ERROR", type(exc).__name__, str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
