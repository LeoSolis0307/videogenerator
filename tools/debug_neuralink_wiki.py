import os
from core.custom_video import _buscar_wikimedia_imagenes, _descargar_imagen_a_archivo


def main() -> int:
    out_dir = r"c:\Users\leona\Downloads\videogenerator\output\debug_neuralink_chip"
    os.makedirs(out_dir, exist_ok=True)

    found = _buscar_wikimedia_imagenes("neuralink implant", max_results=8)
    print("FOUND", len(found))
    if not found:
        print("NO_URL")
        return 2

    url, title = found[0]
    dst = os.path.join(out_dir, "neuralink_chip_01.jpg")
    saved = _descargar_imagen_a_archivo(url, dst)
    print("TITLE", title)
    print("URL", url)
    print("SAVED", saved or "")
    print("EXISTS", bool(saved and os.path.exists(saved)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
