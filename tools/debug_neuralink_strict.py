import os
from core.custom_video import descargar_mejores_imagenes_ddg


def main() -> int:
    out_dir = r"c:\Users\leona\Downloads\videogenerator\output\debug_neuralink_chip"
    os.makedirs(out_dir, exist_ok=True)

    rutas, metas = descargar_mejores_imagenes_ddg(
        out_dir,
        ["neuralink n1 implant chip close-up"],
        ["only device/chip implant, no elon musk stage presentation, realistic photo"],
        max_per_query=4,
        segment_numbers=[1],
    )

    print("FOUND", len(rutas))
    if metas:
        m0 = metas[0]
        print("CANDS", len((m0.get("candidates") or [])))
        print("SELECTED", m0.get("selected"))
    print("FILES", sorted(os.listdir(out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
