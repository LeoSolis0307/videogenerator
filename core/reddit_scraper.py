import requests
import time
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://old.reddit.com/"
}

def obtener_post():
    print("[REDDIT] Buscando post viral...")
    time.sleep(2)

    url = "https://old.reddit.com/r/AskReddit/top/.json?t=week&limit=10&raw_json=1"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    posts = r.json()["data"]["children"]

    for p in posts:
        d = p["data"]
        if not d.get("stickied") and d.get("num_comments", 0) > 100:
            print(f"[REDDIT] Post encontrado: {d['title']}")
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
