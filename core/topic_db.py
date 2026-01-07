import json
import os
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


DB_PATH_DEFAULT = os.path.join("storage", "app.db")


_STOPWORDS_ES = {
    "a",
    "acerca",
    "ahi",
    "ahí",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "asi",
    "así",
    "aun",
    "aún",
    "bajo",
    "bien",
    "cada",
    "casi",
    "como",
    "cómo",
    "con",
    "contra",
    "cual",
    "cuál",
    "cuando",
    "cuándo",
    "de",
    "del",
    "desde",
    "donde",
    "dónde",
    "dos",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "eran",
    "eres",
    "es",
    "esa",
    "esas",
    "ese",
    "esos",
    "esta",
    "está",
    "estan",
    "están",
    "estas",
    "este",
    "esto",
    "estos",
    "fue",
    "fueron",
    "ha",
    "han",
    "hasta",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "mas",
    "más",
    "me",
    "mi",
    "mis",
    "mucho",
    "muy",
    "no",
    "nos",
    "o",
    "otra",
    "otras",
    "otro",
    "otros",
    "para",
    "pero",
    "poco",
    "por",
    "porque",
    "que",
    "qué",
    "se",
    "sea",
    "ser",
    "si",
    "sí",
    "sin",
    "sobre",
    "su",
    "sus",
    "tambien",
    "también",
    "te",
    "tengo",
    "tener",
    "tiene",
    "tienen",
    "toda",
    "todas",
    "todo",
    "todos",
    "tu",
    "tú",
    "un",
    "una",
    "unas",
    "uno",
    "unos",
    "y",
    "ya",
}

_STOPWORDS_EN = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "why",
    "with",
}

_STOPWORDS = _STOPWORDS_ES | _STOPWORDS_EN


@dataclass(frozen=True)
class TopicMatch:
    id: int
    kind: str
    topic_key: str
    brief: str
    similarity: float
    created_at: int
    plan_dir: str | None
    video_path: str | None


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)


def _connect(db_path: str) -> sqlite3.Connection:
    _ensure_dir(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def init_db(db_path: str = DB_PATH_DEFAULT) -> None:
    with _connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                topic_key TEXT NOT NULL,
                tokens_json TEXT NOT NULL,
                brief TEXT,
                created_at INTEGER NOT NULL,
                plan_dir TEXT,
                video_path TEXT
            );
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_topics_kind_created ON topics(kind, created_at DESC);"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_topics_kind_key ON topics(kind, topic_key);"
        )


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = _strip_accents(t)
    t = re.sub(r"\s+", " ", t)
    return t


def tokenize(text: str) -> list[str]:
    t = normalize_text(text)
                                                      
    raw = re.findall(r"[a-z0-9]+", t)
    out: list[str] = []

    for w in raw:
        if len(w) <= 1:
            continue
        if w in _STOPWORDS:
            continue

                                                                          
        if len(w) >= 4 and w.endswith("s") and not w.endswith("ss"):
            w2 = w[:-1]
            if len(w2) >= 3 and w2 not in _STOPWORDS:
                w = w2

        out.append(w)

    return out


def topic_key_from_text(text: str, *, max_tokens: int = 8) -> str:
    toks = tokenize(text)
                                                      
    seen: set[str] = set()
    ordered: list[str] = []
    for w in toks:
        if w in seen:
            continue
        seen.add(w)
        ordered.append(w)
        if len(ordered) >= max_tokens:
            break
    return " ".join(ordered).strip()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def find_similar_topic(
    text: str,
    *,
    kind: str = "custom",
    kinds: tuple[str, ...] | None = None,
    db_path: str = DB_PATH_DEFAULT,
    threshold: float = 0.80,
) -> TopicMatch | None:
\
\
\
\
\
\
\
\

    init_db(db_path)

    tokens = tokenize(text)
    tokset = set(tokens)
    if len(tokset) < 2:
        return None

                                                                                                
    min_overlap = 2

                                                                                   
    want_kinds = kinds if kinds else (kind,)
    with _connect(db_path) as con:
        if len(want_kinds) == 1:
            rows = con.execute(
                "SELECT id, kind, topic_key, tokens_json, brief, created_at, plan_dir, video_path "
                "FROM topics WHERE kind = ? ORDER BY created_at DESC LIMIT 500;",
                (want_kinds[0],),
            ).fetchall()
        else:
            placeholders = ",".join(["?"] * len(want_kinds))
            rows = con.execute(
                "SELECT id, kind, topic_key, tokens_json, brief, created_at, plan_dir, video_path "
                f"FROM topics WHERE kind IN ({placeholders}) ORDER BY created_at DESC LIMIT 800;",
                tuple(want_kinds),
            ).fetchall()

    best: TopicMatch | None = None
    for (rid, rkind, rkey, rtokens_json, rbrief, rcreated, rplan, rvideo) in rows:
        try:
            rtoks = set(json.loads(rtokens_json) or [])
        except Exception:
            rtoks = set()
        if len(rtoks) < 2:
            continue

        overlap = len(tokset & rtoks)
        if overlap < min_overlap:
            continue

        sim = _jaccard(tokset, rtoks)
        if sim >= threshold:
            match = TopicMatch(
                id=int(rid),
                kind=str(rkind),
                topic_key=str(rkey),
                brief=str(rbrief or ""),
                similarity=float(sim),
                created_at=int(rcreated),
                plan_dir=str(rplan) if rplan else None,
                video_path=str(rvideo) if rvideo else None,
            )
            if best is None or match.similarity > best.similarity:
                best = match

    return best


def register_topic_if_new(
    text: str,
    *,
    kind: str = "custom",
    db_path: str = DB_PATH_DEFAULT,
    plan_dir: str | None = None,
    video_path: str | None = None,
    threshold: float = 0.98,
) -> int | None:
    \
\
\
\
    existing = find_similar_topic(text, kind=kind, db_path=db_path, threshold=threshold)
    if existing is not None:
        return None
    return register_topic(text, kind=kind, plan_dir=plan_dir, video_path=video_path, db_path=db_path)


def delete_by_plan_dir(plan_dir: str, *, kind: str, db_path: str = DB_PATH_DEFAULT) -> int:
    if not plan_dir:
        return 0
    init_db(db_path)
    with _connect(db_path) as con:
        cur = con.execute("DELETE FROM topics WHERE kind = ? AND plan_dir = ?;", (kind, plan_dir))
        return int(cur.rowcount or 0)


def register_topic(
    text: str,
    *,
    kind: str = "custom",
    plan_dir: str | None = None,
    video_path: str | None = None,
    db_path: str = DB_PATH_DEFAULT,
) -> int:
    init_db(db_path)

    tokens = tokenize(text)
    key = topic_key_from_text(text)
    payload = json.dumps(tokens, ensure_ascii=False)
    now = int(time.time())

    with _connect(db_path) as con:
        cur = con.execute(
            "INSERT INTO topics(kind, topic_key, tokens_json, brief, created_at, plan_dir, video_path) "
            "VALUES(?, ?, ?, ?, ?, ?, ?);",
            (kind, key, payload, text, now, plan_dir, video_path),
        )
        return int(cur.lastrowid)
