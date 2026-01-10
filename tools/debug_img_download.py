import os
import shutil
import sys
from pathlib import Path

                                                                                 
                                                              
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.custom_video import (
    _descargar_imagen_a_archivo,
    _is_blocked_image_host,
    _resolve_wikimedia_thumb_via_api,
)


def main() -> int:
    out_dir = Path("output") / "_debug_img"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_thumb = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/"
        "100_trajectories_guided_by_the_wave_function.png/"
        "632px-100_trajectories_guided_by_the_wave_function.png"
    )

    print("UA set:", bool(os.environ.get("WIKIMEDIA_USER_AGENT")))
    print("IMG_DEBUG_HTTP:", os.environ.get("IMG_DEBUG_HTTP") or "")

    resolved = _resolve_wikimedia_thumb_via_api(test_thumb)
    print("resolve_thumb:", resolved)

    saved = _descargar_imagen_a_archivo(test_thumb, str(out_dir / "wiki_test.jpg"))
    print("download_saved:", saved)
    if saved:
        p = Path(saved)
        print("file_exists:", p.exists(), "size:", p.stat().st_size)

    print(
        "blocked_researchgate:",
        _is_blocked_image_host("https://www.researchgate.net/publication/123/figure/fig1.png"),
    )
    print(
        "blocked_rgstatic:",
        _is_blocked_image_host("https://i1.rgstatic.net/publication/123/largepreview.png"),
    )

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
