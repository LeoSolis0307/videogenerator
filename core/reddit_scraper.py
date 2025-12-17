import requests
import time
import random

from utils.fs import cargar_historial, guardar_historial

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://old.reddit.com/"
}

def obtener_post():
    print("[REDDIT] Buscando post viral...")
    time.sleep(2)

    historial = cargar_historial()

    fuentes = [
        "https://old.reddit.com/r/AskReddit/top/.json?t=week&limit=50&raw_json=1",
        "https://old.reddit.com/r/AskReddit/top/.json?t=month&limit=50&raw_json=1",
        "https://old.reddit.com/r/AskReddit/top/.json?t=year&limit=50&raw_json=1",
        "https://old.reddit.com/r/AskReddit/hot/.json?limit=50&raw_json=1",
    ]

    vistos = set()
    posts_combo = []

    for url in fuentes:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            posts = r.json()["data"]["children"]
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

    url = f"https://old.reddit.com{permalink}.json?raw_json=1"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    comments = r.json()[1]["data"]["children"]
    random.shuffle(comments)

    print(f"[REDDIT] {len(comments)} comentarios obtenidos")
    return comments
