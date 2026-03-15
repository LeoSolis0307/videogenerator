import re
import requests

u = "https://www.bing.com/images/search"
q = "neuralink implant chip"
resp = requests.get(u, params={"q": q, "form": "HDRSC2", "first": "1"}, timeout=20)
resp.raise_for_status()
html = resp.text or ""
print("LEN", len(html))
print("HAS_IUSC", "iusc" in html.lower())

hits = re.findall(r'\bm="([^"]{80,2000})"', html)
print("M_ATTR", len(hits))
print(hits[:2])

print("SNIPPET", html[:700].replace("\n", " "))
