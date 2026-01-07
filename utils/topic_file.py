import os

from core import topic_db


TOPICS_FILE_DEFAULT = os.path.join("storage", "temas_custom.txt")


def ensure_topics_file(path: str = TOPICS_FILE_DEFAULT) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        return path
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Temas/prompt para videos personalizados (uno por línea)\n"
            "# - Para marcar un tema como usado, el sistema le pondrá prefijo \"0 \" cuando el video termine OK.\n"
            "# - No uses el prefijo \"0 \" tú mismo salvo que quieras desactivarlo.\n\n"
        )
    return path


def load_topics_available(path: str = TOPICS_FILE_DEFAULT) -> list[str]:
    ensure_topics_file(path)
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = (line or "").rstrip("\n")
            s = raw.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            if s.startswith("0"):
                                   
                continue
            if s.startswith("!"):
                s2 = s.lstrip("!").strip()
                if s2:
                    out.append(s2)
                continue
            out.append(s)
    return out


def load_topics_available_with_flags(path: str = TOPICS_FILE_DEFAULT) -> list[tuple[str, bool]]:
    \
\
\
\
    ensure_topics_file(path)
    out: list[tuple[str, bool]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = (line or "").rstrip("\n")
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("0"):
                continue
            if s.startswith("!"):
                s2 = s.lstrip("!").strip()
                if s2:
                    out.append((s2, True))
                continue
            out.append((s, False))
    return out


def load_all_topics_raw(path: str = TOPICS_FILE_DEFAULT) -> list[str]:
    \
    ensure_topics_file(path)
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def mark_topic_used(topic: str, path: str = TOPICS_FILE_DEFAULT) -> bool:
\
\
\
\

    ensure_topics_file(path)
    want = topic_db.normalize_text(topic)
    if not want:
        return False

    changed = False
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = (line or "").rstrip("\n")
            s = raw.strip()
            if not changed and s and not s.startswith("#") and not s.startswith("0"):
                cmp = s
                if cmp.startswith("!"):
                    cmp = cmp.lstrip("!").strip()
                if topic_db.normalize_text(cmp) == want:
                                                                       
                    lines.append("0 " + s)
                    changed = True
                    continue
            lines.append(raw)

    if not changed:
        return False

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")

    return True


def append_topics(
    topics: list[tuple[str, bool]],
    *,
    path: str = TOPICS_FILE_DEFAULT,
) -> None:
    \
\
\
\
    ensure_topics_file(path)
    with open(path, "a", encoding="utf-8") as f:
        for t, forced in topics:
            s = (t or "").replace("\r", " ").replace("\n", " ")
            s = " ".join(s.split()).strip()
            s = s.strip("`").rstrip("`")
            if not s:
                continue
            line = ("! " + s) if forced else s
            f.write(line + "\n")
