import requests
import time
import random
import re
import os

from utils.fs import cargar_historial, guardar_historial

HEADERS = {
    "User-Agent": os.getenv(
        "REDDIT_USER_AGENT",
        "videogenerator/1.0 (script de importación; contacto: local)",
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.reddit.com/",
    "Origin": "https://www.reddit.com",
}


def es_historia_narrativa(texto: str, *, min_chars: int = 900) -> bool:
    t = (texto or "").strip()
    if len(t) < int(min_chars):
        return False

    low = t.lower()

    if "edit:" in low and len(t) < 1400:
        return False

    opinion_markers = (
        "i think",
        "in my opinion",
        "imo",
        "cmv",
        "politics",
        "president",
        "democrats",
        "republicans",
        "left wing",
        "right wing",
    )
    if sum(1 for m in opinion_markers if m in low) >= 2:
        return False

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(sents) < 8:
        return False

    narrative_markers = (
        "when ",
        "then ",
        "after ",
        "before ",
        "later ",
        "that day",
        "years ago",
        "months ago",
        "i was",
        "my mom",
        "my dad",
        "my boyfriend",
        "my girlfriend",
        "my husband",
        "my wife",
    )
    score = sum(1 for m in narrative_markers if m in low)
    return score >= 2


def _request_json_first_ok(urls: list[str], timeout: int = 15):
    last_error = None
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_error = e
            print(f"[REDDIT] Fuente falló ({url}): {e}")
    if last_error is not None:
        raise last_error
    raise RuntimeError("Sin URLs para consultar Reddit")


def _feed_urls() -> list[str]:
    return [
        "https://www.reddit.com/r/AskReddit/top/.json?t=week&limit=50&raw_json=1",
        "https://www.reddit.com/r/AskReddit/top/.json?t=month&limit=50&raw_json=1",
        "https://www.reddit.com/r/AskReddit/top/.json?t=year&limit=50&raw_json=1",
        "https://www.reddit.com/r/AskReddit/hot/.json?limit=50&raw_json=1",
        "https://api.reddit.com/r/AskReddit/top?t=week&limit=50&raw_json=1",
        "https://api.reddit.com/r/AskReddit/top?t=month&limit=50&raw_json=1",
        "https://api.reddit.com/r/AskReddit/top?t=year&limit=50&raw_json=1",
        "https://api.reddit.com/r/AskReddit/hot?limit=50&raw_json=1",
        "https://old.reddit.com/r/AskReddit/top/.json?t=week&limit=50&raw_json=1",
        "https://old.reddit.com/r/AskReddit/top/.json?t=month&limit=50&raw_json=1",
        "https://old.reddit.com/r/AskReddit/top/.json?t=year&limit=50&raw_json=1",
        "https://old.reddit.com/r/AskReddit/hot/.json?limit=50&raw_json=1",
    ]


def _comment_urls(permalink: str) -> list[str]:
    return [
        f"https://www.reddit.com{permalink}.json?raw_json=1",
        f"https://api.reddit.com{permalink}?raw_json=1",
        f"https://old.reddit.com{permalink}.json?raw_json=1",
    ]

def obtener_post():
    print("[REDDIT] Buscando post viral...")
    time.sleep(2)

    historial = cargar_historial()

    fuentes = _feed_urls()

    vistos = set()
    posts_combo = []

    for url in fuentes:
        try:
            data = _request_json_first_ok([url])
            posts = data["data"]["children"]
            for p in posts:
                pid = p["data"].get("id", "")
                if pid in vistos:
                    continue
                vistos.add(pid)
                posts_combo.append(p)
        except Exception as e:
            print(f"[REDDIT] Fuente falló ({url}): {e}")

                                                           
    random.shuffle(posts_combo)

    for p in posts_combo:
        d = p["data"]
        if d.get("stickied"):
            continue
        if d.get("num_comments", 0) <= 50:
            continue

        titulo = (d.get("title") or "").strip()
        post_id = (d.get("id") or "").strip()

                                                                                
        key_titulo = f"reddit_title:{' '.join(titulo.split()).lower()}"
        key_id = f"reddit_post:{post_id.lower()}" if post_id else ""

        if key_titulo in historial or (key_id and key_id in historial):
            print(f"[REDDIT] Saltando repetido: {titulo}")
            continue

        print(f"[REDDIT] Post encontrado: {titulo}")
        guardar_historial([k for k in (key_titulo, key_id) if k])
        return d

    print("[REDDIT] No se encontró post válido")
    return None


def obtener_comentarios(permalink):
    print("[REDDIT] Descargando comentarios...")
    time.sleep(2)

    data = _request_json_first_ok(_comment_urls(permalink))
    comments = data[1]["data"]["children"]
    random.shuffle(comments)

    print(f"[REDDIT] {len(comments)} comentarios obtenidos")
    return comments
