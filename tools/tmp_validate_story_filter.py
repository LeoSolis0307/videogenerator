import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from core import reddit_scraper

post = reddit_scraper.obtener_post()
if not post:
    print('NO_POST')
    raise SystemExit(0)

comments = reddit_scraper.obtener_comentarios(post.get('permalink'))
selected = ''
for c in comments:
    if c.get('kind') != 't1':
        continue
    body = ((c.get('data') or {}).get('body') or '').strip()
    if not body:
        continue
    if reddit_scraper.es_historia_narrativa(body, min_chars=900):
        selected = body
        break

print('POST_TITLE:', (post.get('title') or '')[:180])
print('FOUND_STORY:', bool(selected))
if selected:
    print('STORY_BEGIN')
    print(selected[:3200])
    print('STORY_END')
