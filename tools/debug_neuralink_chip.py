import os
from core.custom_video import descargar_mejores_imagenes_ddg


def main() -> int:
    out_dir = r"c:\Users\leona\Downloads\videogenerator\output\debug_neuralink_chip"
    os.makedirs(out_dir, exist_ok=True)

    rutas, metas = descargar_mejores_imagenes_ddg(
        out_dir,
        ["neuralink n1 implant chip close-up"],
        ["only device/chip implant, no elon musk stage presentation, realistic photo"],
        max_per_query=12,
        segment_numbers=[1],
    )

    print("FOUND", len(rutas))
    if not rutas:
        print("NO_IMAGE")
        return 2

    m0 = metas[0] if metas else {}
    sel = m0.get("selected") if isinstance(m0, dict) else None
    print("TITLE", (sel or {}).get("title") if isinstance(sel, dict) else "")
    print("URL", (sel or {}).get("url") if isinstance(sel, dict) else "")
    print("SAVED", rutas[0])
    print("EXISTS", os.path.exists(rutas[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
