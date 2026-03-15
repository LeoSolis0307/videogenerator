import os
from core.custom_video import _buscar_bing_imagenes, _descargar_imagen_a_archivo


def main() -> int:
    out_dir = r"c:\Users\leona\Downloads\videogenerator\output\debug_neuralink_chip"
    os.makedirs(out_dir, exist_ok=True)

    found = _buscar_bing_imagenes("neuralink implant chip", max_results=5)
    print("FOUND", len(found))

    saved_paths = []
    for i, (url, title) in enumerate(found[:3], start=1):
        dst = os.path.join(out_dir, f"bing_neuralink_{i:02d}.jpg")
        saved = _descargar_imagen_a_archivo(url, dst)
        print(f"CAND_{i}_TITLE", title)
        print(f"CAND_{i}_URL", url)
        print(f"CAND_{i}_SAVED", saved or "")
        if saved:
            saved_paths.append(saved)

    print("SAVED_COUNT", len(saved_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
