import requests
import time
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://google.com"
}

def obtener_post():
    url = "https://www.reddit.com/r/AskReddit/top/.json?t=week&limit=10&raw_json=1"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    posts = r.json()["data"]["children"]
    for p in posts:
        d = p["data"]
        if not d["stickied"] and d["num_comments"] > 100:
            return d
    return None


def obtener_comentarios(permalink):
    time.sleep(1.5)
    url = f"https://www.reddit.com{permalink}.json?raw_json=1"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    comments = r.json()[1]["data"]["children"]
    random.shuffle(comments)
    return comments
