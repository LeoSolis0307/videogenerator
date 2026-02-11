import re

from core import topic_db
from utils import topic_file


def _clean_prompt_line(p: str) -> str:
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


def _extract_prompts_anywhere(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    prompts: list[str] = []

                                          
    for m in re.finditer(r"`\s*(Escribe\s+un\s+guion[^`]+?)\s*`", raw, flags=re.I | re.S):
        prompts.append(m.group(1))

                                                     
    for m in re.finditer(r"(?im)^.*?prompt\s*:\s*(.+)$", raw):
        chunk = (m.group(1) or "").strip()
        if not chunk:
            continue
        m2 = re.search(r"`\s*(Escribe\s+un\s+guion[^`]+?)\s*`", chunk, flags=re.I | re.S)
        if m2:
            prompts.append(m2.group(1))
        else:
            prompts.append(chunk)

                                                   
    for m in re.finditer(r"(?im)^(?:0\s+)?(?:!\s+)?(Escribe\s+un\s+guion[^\n]+)$", raw):
        prompts.append(m.group(1))

                                                      
    for line in raw.splitlines():
        if re.search(r"Escribe\s+un\s+guion", line, flags=re.I) is None:
            continue
        idxs = [m.start() for m in re.finditer(r"Escribe\s+un\s+guion", line, flags=re.I)]
        if not idxs:
            continue
        idxs.append(len(line))
        for a, b in zip(idxs, idxs[1:]):
            prompts.append(line[a:b])

                             
    out: list[str] = []
    seen: set[str] = set()
    for p in prompts:
        s = _clean_prompt_line(p)
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


def _extract_topics_anywhere(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    topics: list[str] = []

    # Markdown bold numbered items: **1. Title**
    for m in re.finditer(r"(?m)^\s*\*\*\s*\d+\s*\.\s*(.+?)\s*\*\*\s*$", raw):
        topics.append(m.group(1))

    # Plain numbered items: 1. Title
    # Avoid matching lines that are blockquotes or headings.
    for m in re.finditer(r"(?m)^\s*(?![>#])\d+\s*\.\s*(.+?)\s*$", raw):
        topics.append(m.group(1))

    # Blockquotes that include a quoted topic/prompt. We keep the full quoted sentence(s)
    # because the topics file is free-form text.
    for m in re.finditer(r"(?m)^\s*>\s*\"([^\"]{10,})\"\s*$", raw):
        topics.append(m.group(1))

    out: list[str] = []
    seen: set[str] = set()
    for t in topics:
        s = _clean_prompt_line(t)
        if not s:
            continue
        # Remove trailing punctuation from titles like "...**" or stray colons/spaces.
        s = s.strip().strip("-").strip()
        norm = topic_db.normalize_text(re.sub(r"[`\"']", "", s))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(s)

    return out


def parse_prompts_from_blob(text: str) -> list[str]:

    prompts = _extract_prompts_anywhere(text)
    if prompts:
        return prompts
    return _extract_topics_anywhere(text)


def dedupe_prompts(
    prompts: list[str],
    *,
    topics_path: str = topic_file.TOPICS_FILE_DEFAULT,
    db_threshold: float = 0.80,
) -> tuple[list[str], list[tuple[str, str]]]:

    topic_file.ensure_topics_file(topics_path)
    existing_lines = topic_file.load_all_topics_raw(topics_path)

                                                                    
    existing_norm: set[str] = set()
    for line in existing_lines:
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("0"):
            s = s.lstrip("0").strip()
        if s.startswith("!"):
            s = s.lstrip("!").strip()
        n = topic_db.normalize_text(s)
        if n:
            existing_norm.add(n)

    accepted: list[str] = []
    discarded: list[tuple[str, str]] = []

    seen_in_blob: set[str] = set()

    for p in prompts:
        p_clean = _clean_prompt_line(p)
        if not p_clean:
            continue

        pn = topic_db.normalize_text(p_clean)
        if pn in seen_in_blob:
            discarded.append((p_clean, "repetido en el texto pegado"))
            continue
        seen_in_blob.add(pn)

        if pn in existing_norm:
            discarded.append((p_clean, "ya existe en el archivo de temas"))
            continue

        match = topic_db.find_similar_topic(p_clean, kinds=("custom", "custom_pending"), threshold=db_threshold)
        if match is not None:
            discarded.append((p_clean, f"repetido en DB (sim={match.similarity:.2f})"))
            continue

        accepted.append(p_clean)

    return accepted, discarded
