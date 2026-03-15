import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _is_placeholder_title(title: str) -> bool:
    t = (title or "").strip().lower()
    return "placeholder" in t


def _sanitize_plan(plan_path: Path) -> tuple[int, int]:
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    segments = data.get("segments") or []
    if not isinstance(segments, list):
        return 0, 0

    touched_segments = 0
    removed_selections = 0

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        image_selection = seg.get("image_selection")
        if not isinstance(image_selection, dict):
            continue

        selected = image_selection.get("selected")
        fallback = str(image_selection.get("fallback") or "").strip().lower()
        title = str((selected or {}).get("title") or "").strip().lower() if isinstance(selected, dict) else ""

        if fallback == "placeholder" or _is_placeholder_title(title):
            seg.pop("image_selection", None)
            touched_segments += 1
            removed_selections += 1

    if touched_segments:
        data["segments"] = segments
        backup = plan_path.with_suffix(plan_path.suffix + ".bak_no_black")
        shutil.copyfile(plan_path, backup)
        plan_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return touched_segments, removed_selections


def _remove_placeholder_files(plan_dir: Path) -> int:
    removed = 0
    pat = re.compile(r"^seg_\d{2}_chosen\.(png|jpg|jpeg|webp|bmp|ppm)$", re.IGNORECASE)
    for name in os.listdir(plan_dir):
        if not pat.match(name):
            continue
        p = plan_dir / name
        try:
            from PIL import Image

            with Image.open(p) as im:
                rgb = im.convert("RGB")
                ex = rgb.getextrema()
            is_black = bool(ex and len(ex) == 3 and all(lo == hi and lo <= 24 for (lo, hi) in ex))
        except Exception:
            is_black = False

        if is_black:
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Limpia placeholders negros y re-renderiza un custom_plan.")
    parser.add_argument("plan", help="Ruta a carpeta output/custom_... o a custom_plan.json")
    parser.add_argument("--voice", default="es-MX-JorgeNeural")
    parser.add_argument("--speed", default="-10%")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = Path(args.plan).resolve()
    if raw.is_dir():
        plan_file = raw / "custom_plan.json"
        plan_dir = raw
    else:
        plan_file = raw
        plan_dir = raw.parent

    if not plan_file.exists():
        print(f"[RERENDER] ❌ No existe plan: {plan_file}")
        return 2

    os.environ.setdefault("IMG_SOURCES", "flickr,wikimedia,openverse")
    os.environ.setdefault("CUSTOM_COMFY_FALLBACK_ON_MISSING_IMAGES", "1")
    os.environ.setdefault("CUSTOM_ALLOW_PLACEHOLDER_IMAGES", "0")
    os.environ.setdefault("IMG_ANTIBOT_PROXY", "1")

    touched, removed_sel = _sanitize_plan(plan_file)
    removed_files = _remove_placeholder_files(plan_dir)
    print(f"[RERENDER] Plan saneado: segmentos={touched}, selections_removidas={removed_sel}, archivos_placeholder_borrados={removed_files}")

    if args.dry_run:
        print("[RERENDER] Dry-run OK. No se ejecutó render.")
        return 0

    from core import custom_video

    ok = custom_video.renderizar_video_personalizado_desde_plan(
        str(plan_dir),
        voz=args.voice,
        velocidad=args.speed,
        interactive=False,
    )
    print("[RERENDER] RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
