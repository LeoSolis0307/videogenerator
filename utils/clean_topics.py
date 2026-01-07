import argparse
import os
import re
import shutil
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import topic_db


def _extract_prompts_anywhere(text: str) -> list[str]:
    txt = (text or "")

    prompts: list[str] = []

                                          
    for m in re.finditer(r"`\s*(Escribe\s+un\s+guion[^`]+?)\s*`", txt, flags=re.I | re.S):
        prompts.append(m.group(1))

                                                     
                                                          
    for m in re.finditer(r"(?im)^.*?prompt\s*:\s*(.+)$", txt):
        chunk = m.group(1).strip()
        if not chunk:
            continue
                                                       
        m2 = re.search(r"`\s*(Escribe\s+un\s+guion[^`]+?)\s*`", chunk, flags=re.I | re.S)
        if m2:
            prompts.append(m2.group(1))
        else:
            prompts.append(chunk)

                                                   
    for m in re.finditer(r"(?im)^(?:0\s+)?(?:!\s+)?(Escribe\s+un\s+guion[^\n]+)$", txt):
        prompts.append(m.group(1))

                                                                    
    for line in txt.splitlines():
        if re.search(r"Escribe\s+un\s+guion", line, flags=re.I) is None:
            continue
        idxs = [m.start() for m in re.finditer(r"Escribe\s+un\s+guion", line, flags=re.I)]
        if not idxs:
            continue
        idxs.append(len(line))
        for a, b in zip(idxs, idxs[1:]):
            prompts.append(line[a:b])

    def _clean_one(p: str) -> str:
        s = re.sub(r"\s+", " ", (p or "").strip())
                                  
        s = s.strip("`").rstrip("`")

                                                                       
        low = s.lower()
        gi = low.find("gancho:")
        if gi != -1:
            q1 = s.find('"', gi)
            if q1 != -1:
                q2 = s.find('"', q1 + 1)
                if q2 != -1:
                    s = s[: q2 + 1].rstrip()

                                       
        s = re.sub(r"\b(end)\b\s*$", "", s, flags=re.I).strip()
        return s

                                                                        
    out: list[str] = []
    seen: set[str] = set()
    for p in prompts:
        s = _clean_one(p)
        if not s:
            continue
        if "escribe un guion" not in s.lower():
            continue
        norm = topic_db.normalize_text(re.sub(r"[`\"']", "", s))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(s)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=os.path.join("storage", "temas_custom.txt"))
    ap.add_argument("--out", dest="out_path", default=os.path.join("storage", "temas_custom.cleaned.txt"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    in_path = args.in_path
    out_path = args.out_path

    if not os.path.exists(in_path):
        print(f"[CLEAN] Input not found: {in_path}")
        return 1

    raw = open(in_path, "r", encoding="utf-8").read()
    prompts = _extract_prompts_anywhere(raw)

    header = [
        "# Temas/prompt para videos personalizados (uno por línea)",
        "# - Para marcar un tema como usado, el sistema le pondrá prefijo \"0 \" cuando el video termine OK.",
        "# - Prefijo \"! \" = forzar uso aunque sea repetido (se saltará el bloqueo por DB).",
        "",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + prompts) + "\n")

    print(f"[CLEAN] Extracted prompts: {len(prompts)}")
    print(f"[CLEAN] Wrote: {out_path}")

    if args.overwrite:
        if args.backup:
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = in_path + f".bak_{ts}"
            shutil.copy2(in_path, backup_path)
            print(f"[CLEAN] Backup: {backup_path}")
        shutil.copy2(out_path, in_path)
        print(f"[CLEAN] Overwrote: {in_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
