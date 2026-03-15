import hashlib
import json
import os
import sys
import time
from typing import Any

import requests

from core import topic_db
from core.config import settings
from core.llm.client import llm_client
from core.reddit_scraper import HEADERS, es_historia_narrativa
from utils.fs import asegurar_directorio

KIND_IMPORTED = "reddit_story_imported"

_REDDIT_COOLDOWN_UNTIL = 0.0
_REDDIT_429_COUNT = 0
_REDDIT_429_SOFT_LIMIT = max(1, int(settings.reddit_429_soft_limit))
_REDDIT_429_HARD_LIMIT = max(_REDDIT_429_SOFT_LIMIT + 1, int(settings.reddit_429_hard_limit))
_REDDIT_REQUEST_PAUSE_S = max(0.0, float(settings.reddit_request_pause_s))
_REDDIT_TIMEOUT_SEC = max(5.0, float(settings.reddit_timeout_sec))
_REDDIT_REQUEST_RETRIES = max(0, int(settings.reddit_request_retries))
_REDDIT_RETRY_BACKOFF_S = max(0.0, float(settings.reddit_retry_backoff_s))
_REDDIT_MAX_COOLDOWN_WAIT_S = max(1.0, float(settings.reddit_max_cooldown_wait_s))

_REDDIT_CLIENT_ID = (settings.reddit_client_id or "").strip()
_REDDIT_CLIENT_SECRET = (settings.reddit_client_secret or "").strip()
_REDDIT_OAUTH_TOKEN = ""
_REDDIT_OAUTH_TOKEN_EXP = 0.0

DEFAULT_SUBREDDITS = (
    "AskReddit",
    "NoStupidQuestions",
    "AITA",
    "TrueOffMyChest",
    "relationship_advice",
    "TIFU",
    "confession",
    "AmItheAsshole",
)

_SEARCH_STATE_FILE = os.path.join("storage", "reddit_import_state.json")
_STATE_VERSION = 1
_STATE_RETENTION_DAYS = max(1, int(settings.reddit_state_retention_days))
_STATE_MAX_POST_IDS = max(500, int(settings.reddit_state_max_post_ids))
_STATE_MAX_COMMENT_IDS = max(1000, int(settings.reddit_state_max_comment_ids))


def _now_ts() -> int:
    return int(time.time())


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_timestamp_map(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in raw.items():
        key = (str(k) or "").strip()
        if not key:
            continue
        ts = _as_int(v, 0)
        if ts <= 0:
            continue
        out[key] = ts
    return out


def _state_default() -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "updated_at": _now_ts(),
        "subreddit_cursor": 0,
        "feed_cursor_by_subreddit": {},
        "seen_post_ids": {},
        "seen_comment_ids": {},
    }


def _prune_timestamp_map(values: dict[str, int], *, keep_max: int, min_ts: int) -> dict[str, int]:
    if not values:
        return {}
    filtered = {k: int(v) for k, v in values.items() if int(v) >= int(min_ts)}
    if len(filtered) <= keep_max:
        return filtered
    ordered = sorted(filtered.items(), key=lambda it: it[1], reverse=True)
    trimmed = ordered[:keep_max]
    return {k: int(v) for k, v in trimmed}


def _load_search_state() -> dict[str, Any]:
    default = _state_default()
    try:
        asegurar_directorio(os.path.dirname(_SEARCH_STATE_FILE))
        if not os.path.exists(_SEARCH_STATE_FILE):
            return default
        with open(_SEARCH_STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return default

    if not isinstance(raw, dict):
        return default

    now = _now_ts()
    min_ts = now - (_STATE_RETENTION_DAYS * 86400)

    feed_cursor_raw = raw.get("feed_cursor_by_subreddit")
    feed_cursor: dict[str, int] = {}
    if isinstance(feed_cursor_raw, dict):
        for k, v in feed_cursor_raw.items():
            kk = (str(k) or "").strip().lower()
            if not kk:
                continue
            feed_cursor[kk] = max(0, _as_int(v, 0))

    seen_post_ids = _prune_timestamp_map(
        _normalize_timestamp_map(raw.get("seen_post_ids")),
        keep_max=_STATE_MAX_POST_IDS,
        min_ts=min_ts,
    )
    seen_comment_ids = _prune_timestamp_map(
        _normalize_timestamp_map(raw.get("seen_comment_ids")),
        keep_max=_STATE_MAX_COMMENT_IDS,
        min_ts=min_ts,
    )

    return {
        "version": _STATE_VERSION,
        "updated_at": now,
        "subreddit_cursor": max(0, _as_int(raw.get("subreddit_cursor"), 0)),
        "feed_cursor_by_subreddit": feed_cursor,
        "seen_post_ids": seen_post_ids,
        "seen_comment_ids": seen_comment_ids,
    }


def _save_search_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return

    now = _now_ts()
    min_ts = now - (_STATE_RETENTION_DAYS * 86400)
    state["version"] = _STATE_VERSION
    state["updated_at"] = now

    seen_post_ids = _prune_timestamp_map(
        _normalize_timestamp_map(state.get("seen_post_ids")),
        keep_max=_STATE_MAX_POST_IDS,
        min_ts=min_ts,
    )
    seen_comment_ids = _prune_timestamp_map(
        _normalize_timestamp_map(state.get("seen_comment_ids")),
        keep_max=_STATE_MAX_COMMENT_IDS,
        min_ts=min_ts,
    )
    state["seen_post_ids"] = seen_post_ids
    state["seen_comment_ids"] = seen_comment_ids

    feed_cursor: dict[str, int] = {}
    raw_feed_cursor = state.get("feed_cursor_by_subreddit")
    if isinstance(raw_feed_cursor, dict):
        for k, v in raw_feed_cursor.items():
            kk = (str(k) or "").strip().lower()
            if not kk:
                continue
            feed_cursor[kk] = max(0, _as_int(v, 0))
    state["feed_cursor_by_subreddit"] = feed_cursor

    try:
        asegurar_directorio(os.path.dirname(_SEARCH_STATE_FILE))
        with open(_SEARCH_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _log(f"[STORY-IMPORT] ⚠️ No se pudo guardar estado de búsqueda: {e}")


def _rotate_with_cursor(items: list[str], cursor: int) -> list[str]:
    if not items:
        return []
    c = int(cursor) % len(items)
    if c <= 0:
        return list(items)
    return list(items[c:] + items[:c])


def _log(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
        safe = str(message).encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe)


def _split_csv(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _subreddits_from_env(default_subreddits: tuple[str, ...]) -> tuple[str, ...]:
    raw = (settings.reddit_import_subreddits or "").strip()
    if not raw:
        return default_subreddits
    items = _split_csv(raw)
    if not items:
        return default_subreddits
    return tuple(items)


def _effective_subreddits(default_subreddits: tuple[str, ...], *, light_mode: bool) -> tuple[str, ...]:
    base = _subreddits_from_env(default_subreddits)
    if not light_mode:
        return base
    if len(base) <= 2:
        return base
    return base[:2]


def _feed_profile_from_env() -> str:
    profile = (settings.reddit_feed_profile or "light").strip().lower()
    if profile in {"minimal", "light", "full"}:
        return profile
    return "light"


def _oauth_enabled() -> bool:
    return bool(_REDDIT_CLIENT_ID and _REDDIT_CLIENT_SECRET)


def _get_reddit_oauth_token(timeout: float | None = None) -> str:
    global _REDDIT_OAUTH_TOKEN, _REDDIT_OAUTH_TOKEN_EXP

    if not _oauth_enabled():
        return ""

    now = time.time()
    if _REDDIT_OAUTH_TOKEN and now < (_REDDIT_OAUTH_TOKEN_EXP - 30.0):
        return _REDDIT_OAUTH_TOKEN

    timeout = float(timeout if timeout is not None else _REDDIT_TIMEOUT_SEC)
    try:
        resp = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(_REDDIT_CLIENT_ID, _REDDIT_CLIENT_SECRET),
            headers={"User-Agent": HEADERS.get("User-Agent", "videogenerator/1.0")},
            data={"grant_type": "client_credentials"},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}
        token = str(payload.get("access_token") or "").strip()
        expires_in = float(payload.get("expires_in") or 3600)
        if not token:
            return ""
        _REDDIT_OAUTH_TOKEN = token
        _REDDIT_OAUTH_TOKEN_EXP = time.time() + max(120.0, expires_in)
        return token
    except Exception as e:
        _log(f"[STORY-IMPORT] ⚠️ No se pudo obtener token OAuth de Reddit: {e}")
        return ""


def _oauth_headers(timeout: float | None = None) -> dict[str, str]:
    token = _get_reddit_oauth_token(timeout=timeout)
    if not token:
        return HEADERS
    return {
        "User-Agent": HEADERS.get("User-Agent", "videogenerator/1.0"),
        "Accept": "application/json",
        "Authorization": f"bearer {token}",
    }


def _normalize_space(text: str) -> str:
    return " ".join((text or "").strip().split())


def _text_hash(text: str) -> str:
    clean = _normalize_space(text)
    return hashlib.sha1(clean.encode("utf-8")).hexdigest()


def _request_json(url: str, timeout: float | None = None) -> Any | None:
    global _REDDIT_COOLDOWN_UNTIL, _REDDIT_429_COUNT

    now = time.time()
    if now < _REDDIT_COOLDOWN_UNTIL:
        wait_s = min(_REDDIT_COOLDOWN_UNTIL - now, _REDDIT_MAX_COOLDOWN_WAIT_S)
        if wait_s > 0:
            _log(f"[STORY-IMPORT] ℹ️ Reddit en cooldown; esperando {wait_s:.1f}s...")
            time.sleep(wait_s)

    timeout = float(timeout if timeout is not None else _REDDIT_TIMEOUT_SEC)
    max_attempts = _REDDIT_REQUEST_RETRIES + 1

    for attempt in range(1, max_attempts + 1):
        try:
            if _REDDIT_REQUEST_PAUSE_S > 0:
                time.sleep(_REDDIT_REQUEST_PAUSE_S)
            headers = _oauth_headers(timeout=timeout) if "oauth.reddit.com" in url else HEADERS
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                _REDDIT_429_COUNT += 1
                retry_after = 3.0
                ra = (resp.headers.get("Retry-After") or "").strip()
                if ra:
                    try:
                        retry_after = max(1.0, float(ra))
                    except Exception:
                        retry_after = 3.0
                _REDDIT_COOLDOWN_UNTIL = max(_REDDIT_COOLDOWN_UNTIL, time.time() + retry_after)

                if _REDDIT_429_COUNT <= _REDDIT_429_SOFT_LIMIT or _REDDIT_429_COUNT % 4 == 0:
                    _log(
                        "[STORY-IMPORT] ⚠️ Reddit devolvió 429; "
                        f"enfriando {retry_after:.1f}s (racha={_REDDIT_429_COUNT})."
                    )
                if attempt < max_attempts:
                    wait_s = min(retry_after, _REDDIT_MAX_COOLDOWN_WAIT_S)
                    if wait_s > 0:
                        time.sleep(wait_s)
                    continue
                return None

            resp.raise_for_status()
            _REDDIT_429_COUNT = 0
            return resp.json()
        except requests.exceptions.Timeout:
            if attempt < max_attempts:
                wait_s = _REDDIT_RETRY_BACKOFF_S * attempt
                _log(
                    "[STORY-IMPORT] ⚠️ Timeout consultando Reddit "
                    f"({attempt}/{max_attempts}); reintentando en {wait_s:.1f}s..."
                )
                if wait_s > 0:
                    time.sleep(wait_s)
                continue
            _log(
                "[STORY-IMPORT] ⚠️ Falló fuente Reddit por timeout "
                f"tras {max_attempts} intentos ({url})"
            )
            return None
        except requests.exceptions.ConnectionError as e:
            if attempt < max_attempts:
                wait_s = _REDDIT_RETRY_BACKOFF_S * attempt
                _log(
                    "[STORY-IMPORT] ⚠️ Error de conexión a Reddit "
                    f"({attempt}/{max_attempts}); reintentando en {wait_s:.1f}s..."
                )
                if wait_s > 0:
                    time.sleep(wait_s)
                continue
            _log(f"[STORY-IMPORT] ⚠️ Falló fuente Reddit ({url}): {e}")
            return None
        except Exception as e:
            _log(f"[STORY-IMPORT] ⚠️ Falló fuente Reddit ({url}): {e}")
            return None


def _request_json_first_ok(urls: list[str], timeout: float | None = None) -> Any | None:
    for url in urls:
        if _reddit_is_hard_rate_limited():
            return None
        payload = _request_json(url, timeout=timeout)
        if payload is not None:
            return payload
    return None


def _reddit_is_hard_rate_limited() -> bool:
    return _REDDIT_429_COUNT >= _REDDIT_429_HARD_LIMIT


def _reddit_rate_limit_detected() -> bool:
    return _REDDIT_429_COUNT > 0 or (time.time() < _REDDIT_COOLDOWN_UNTIL)


def _post_feeds(subreddit: str, *, profile: str = "full") -> list[str]:
    s = (subreddit or "").strip()
    if not s:
        return []
    oauth_urls: list[str] = []
    if _oauth_enabled():
        oauth_urls = [
            f"https://oauth.reddit.com/r/{s}/top/.json?t=week&limit=30&raw_json=1",
            f"https://oauth.reddit.com/r/{s}/hot/.json?limit=30&raw_json=1",
        ]
    if profile == "minimal":
        return oauth_urls + [
            f"https://www.reddit.com/r/{s}/top/.json?t=week&limit=20&raw_json=1",
            f"https://old.reddit.com/r/{s}/top/.json?t=week&limit=20&raw_json=1",
        ]
    if profile == "light":
        return oauth_urls + [
            f"https://www.reddit.com/r/{s}/top/.json?t=week&limit=25&raw_json=1",
            f"https://www.reddit.com/r/{s}/hot/.json?limit=25&raw_json=1",
            f"https://api.reddit.com/r/{s}/top?t=week&limit=25&raw_json=1",
            f"https://old.reddit.com/r/{s}/top/.json?t=week&limit=25&raw_json=1",
        ]
    return oauth_urls + [
        f"https://www.reddit.com/r/{s}/top/.json?t=week&limit=30&raw_json=1",
        f"https://www.reddit.com/r/{s}/top/.json?t=month&limit=30&raw_json=1",
        f"https://www.reddit.com/r/{s}/hot/.json?limit=30&raw_json=1",
        f"https://api.reddit.com/r/{s}/top?t=week&limit=30&raw_json=1",
        f"https://api.reddit.com/r/{s}/top?t=month&limit=30&raw_json=1",
        f"https://api.reddit.com/r/{s}/hot?limit=30&raw_json=1",
        f"https://old.reddit.com/r/{s}/top/.json?t=week&limit=30&raw_json=1",
        f"https://old.reddit.com/r/{s}/top/.json?t=month&limit=30&raw_json=1",
        f"https://old.reddit.com/r/{s}/hot/.json?limit=30&raw_json=1",
    ]


def _comment_urls(permalink: str) -> list[str]:
    pp = (permalink or "").strip()
    if not pp:
        return []
    oauth_urls: list[str] = []
    if _oauth_enabled():
        oauth_urls = [f"https://oauth.reddit.com{pp}?raw_json=1"]
    return oauth_urls + [
        f"https://www.reddit.com{pp}.json?raw_json=1",
        f"https://api.reddit.com{pp}?raw_json=1",
        f"https://old.reddit.com{pp}.json?raw_json=1",
    ]


def _clean_comment_body(body: str) -> str:
    text = (body or "").strip()
    if not text:
        return ""
    low = text.lower()
    blocked = (
        "[deleted]",
        "[removed]",
        "i am a bot",
        "this action was performed automatically",
        "edit:",
    )
    if any(b in low for b in blocked):
        return ""
    return text


def _collect_reddit_candidates(
    *,
    target_candidates: int,
    min_chars: int,
    subreddits: tuple[str, ...],
    feed_profile: str = "full",
    state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    state = state if isinstance(state, dict) else _state_default()

    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set((state.get("seen_comment_ids") or {}).keys())
    seen_hashes: set[str] = set()

    seen_post_ids_map = state.get("seen_post_ids") or {}
    if not isinstance(seen_post_ids_map, dict):
        seen_post_ids_map = {}
        state["seen_post_ids"] = seen_post_ids_map

    seen_comment_ids_map = state.get("seen_comment_ids") or {}
    if not isinstance(seen_comment_ids_map, dict):
        seen_comment_ids_map = {}
        state["seen_comment_ids"] = seen_comment_ids_map

    feed_cursor_map = state.get("feed_cursor_by_subreddit") or {}
    if not isinstance(feed_cursor_map, dict):
        feed_cursor_map = {}
        state["feed_cursor_by_subreddit"] = feed_cursor_map

    ordered_subreddits = list(subreddits)
    sub_cursor = _as_int(state.get("subreddit_cursor"), 0)
    ordered_subreddits = _rotate_with_cursor(ordered_subreddits, sub_cursor)
    state["subreddit_cursor"] = sub_cursor + 1

    now = _now_ts()

    _log(
        "[STORY-IMPORT] Estado búsqueda: "
        f"posts_cache={len(seen_post_ids_map)} | comments_cache={len(seen_comment_ids_map)}"
    )

    for subreddit in ordered_subreddits:
        if _reddit_is_hard_rate_limited():
            _log(
                "[STORY-IMPORT] ⚠️ Reddit sigue en rate limit alto; "
                "se corta búsqueda para evitar spam de solicitudes."
            )
            return out

        feeds = _post_feeds(subreddit, profile=feed_profile)
        sub_key = (subreddit or "").strip().lower()
        feed_cursor = _as_int(feed_cursor_map.get(sub_key), 0)
        ordered_feeds = _rotate_with_cursor(feeds, feed_cursor)
        feed_cursor_map[sub_key] = feed_cursor + 1

        for feed in ordered_feeds:
            if _reddit_is_hard_rate_limited():
                _log(
                    "[STORY-IMPORT] ⚠️ Demasiados 429 consecutivos; "
                    "deteniendo scraping por esta ejecución."
                )
                return out

            payload = _request_json(feed)
            if not payload:
                continue

            posts = ((payload.get("data") or {}).get("children") or [])
            for p in posts:
                d = p.get("data") or {}
                if d.get("stickied"):
                    continue

                post_id = (d.get("id") or "").strip()
                if post_id and post_id in seen_post_ids_map:
                    continue

                post_score = int(d.get("score") or 0)
                post_comments = int(d.get("num_comments") or 0)
                if post_score < 400 or post_comments < 80:
                    if post_id:
                        seen_post_ids_map[post_id] = now
                    continue

                permalink = (d.get("permalink") or "").strip()
                if not permalink:
                    if post_id:
                        seen_post_ids_map[post_id] = now
                    continue

                comments_payload = _request_json_first_ok(_comment_urls(permalink))
                if not comments_payload or not isinstance(comments_payload, list) or len(comments_payload) < 2:
                    if post_id:
                        seen_post_ids_map[post_id] = now
                    continue

                comments = (((comments_payload[1] or {}).get("data") or {}).get("children") or [])
                if post_id:
                    seen_post_ids_map[post_id] = now
                for c in comments:
                    if c.get("kind") != "t1":
                        continue
                    cd = c.get("data") or {}
                    cid = (cd.get("id") or "").strip()
                    if cid and cid in seen_comment_ids_map:
                        continue
                    body = _clean_comment_body(cd.get("body") or "")
                    if not body:
                        if cid:
                            seen_comment_ids_map[cid] = now
                        continue
                    if len(body) < min_chars:
                        if cid:
                            seen_comment_ids_map[cid] = now
                        continue
                    if not es_historia_narrativa(body, min_chars=max(min_chars, 900)):
                        if cid:
                            seen_comment_ids_map[cid] = now
                        continue

                    comment_score = int(cd.get("score") or 0)
                    if comment_score < 50:
                        if cid:
                            seen_comment_ids_map[cid] = now
                        continue

                    body_hash = _text_hash(body)
                    if cid and cid in seen_ids:
                        continue
                    if body_hash in seen_hashes:
                        continue

                    seen_ids.add(cid)
                    seen_hashes.add(body_hash)
                    if cid:
                        seen_comment_ids_map[cid] = now
                    out.append(
                        {
                            "comment_id": cid,
                            "comment_body": body,
                            "comment_score": comment_score,
                            "post_id": (d.get("id") or "").strip(),
                            "post_title": (d.get("title") or "").strip(),
                            "post_score": post_score,
                            "post_num_comments": post_comments,
                            "subreddit": (d.get("subreddit") or subreddit).strip(),
                            "permalink": permalink,
                        }
                    )
                    if len(out) >= target_candidates:
                        state["updated_at"] = _now_ts()
                        return out

    state["updated_at"] = _now_ts()
    return out


def _heuristic_priority(item: dict[str, Any]) -> float:
    body_len = len(item.get("comment_body") or "")
    comment_score = int(item.get("comment_score") or 0)
    post_score = int(item.get("post_score") or 0)
    post_comments = int(item.get("post_num_comments") or 0)
    return (body_len * 0.012) + (comment_score * 0.8) + (post_score * 0.08) + (post_comments * 0.22)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _has_climax_and_resolution_markers(text: str) -> tuple[bool, bool]:
    low = (text or "").lower()
    if not low:
        return (False, False)

    climax_markers = (
        "de repente",
        "entonces",
        "en ese momento",
        "me enfrenté",
        "lo confronté",
        "descubrí que",
        "todo explotó",
        "la discusión",
        "el conflicto",
        "lo peor",
        "the moment",
        "suddenly",
        "then everything",
        "i confronted",
        "i found out",
    )
    resolution_markers = (
        "al final",
        "finalmente",
        "desde entonces",
        "desde ese día",
        "esa misma noche",
        "nunca volví",
        "corté todo contacto",
        "puse límites",
        "lo saqué de mi vida",
        "aprendí que",
        "la lección",
        "cerré ese capítulo",
        "ahora",
        "in the end",
        "finally",
        "since then",
        "from that day",
        "that same night",
        "never again",
        "cut ties",
        "i learned",
        "now i",
    )

    has_climax = any(m in low for m in climax_markers)
    has_resolution = any(m in low for m in resolution_markers)
    return (has_climax, has_resolution)


def _has_intensity_markers(text: str) -> bool:
    low = (text or "").lower()
    if not low:
        return False
    intense = (
        "me quedé helado",
        "se me cayó el mundo",
        "traición",
        "me rompió",
        "amenaz",
        "grit",
        "llor",
        "me humill",
        "me destruyó",
        "explot",
        "suddenly",
        "betray",
        "panic",
        "screamed",
        "i broke down",
    )
    hits = sum(1 for m in intense if m in low)
    return hits >= 2


def _has_generic_ending(text: str) -> bool:
    low = (text or "").lower().strip()
    if not low:
        return True
    tail = low[-280:]
    generic = (
        "aprendí que",
        "la lección",
        "moraleja",
        "en conclusión",
        "finalmente entendí",
        "al final entendí",
        "desde entonces todo",
        "y colorín colorado",
        "in the end i learned",
        "the lesson is",
        "moral of the story",
    )
    return any(g in tail for g in generic)


def _style_gate(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _has_generic_ending(t):
        return False
    if not _has_intensity_markers(t):
        return False
    mk_climax, mk_resolution = _has_climax_and_resolution_markers(t)
    return mk_climax and mk_resolution


def _evaluate_story_with_ai(
    candidate: dict[str, Any],
    *,
    min_chars_final: int,
    max_chars_final: int,
) -> dict[str, Any] | None:
    body = (candidate.get("comment_body") or "").strip()
    if not body:
        return None

    prompt = (
        "Eres editor experto en historias ultra virales para videos en español. "
        "Evalúa esta historia real de Reddit con criterio extremo de viralidad, gancho y completitud. "
        "Debe ser una historia completa (inicio-desarrollo-clímax-desenlace), larga y adictiva. "
        "IMPORTANTE: si no tiene clímax fuerte y desenlace contundente, debes rechazarla. "
        "Devuelve SOLO JSON válido con este formato exacto: "
        "{\"aceptar\":bool,\"puntaje_viral\":0-100,\"puntaje_gancho\":0-100,\"puntaje_calidad\":0-100,"
        "\"puntaje_completitud\":0-100,\"es_historia_larga\":bool,\"es_historia_completa\":bool,"
        "\"tiene_climax_fuerte\":bool,\"tiene_desenlace_contundente\":bool,"
        "\"puntaje_climax\":0-100,\"puntaje_desenlace\":0-100,"
        "\"puntaje_intensidad\":0-100,\"puntaje_originalidad_desenlace\":0-100,"
        "\"motivo\":\"...\",\"historia_limpia_es\":\"...\"}. "
        "Reglas estrictas: solo aceptar si la historia puede enganchar desde el primer segundo y retener hasta el final. "
        "historia_limpia_es debe estar en español natural, sin listas, sin encabezados, sin emojis, "
        "sin inventar hechos no presentes, manteniendo coherencia y cierre narrativo. "
        "Prefiere una versión compacta y muy potente para video corto. "
        "El clímax debe ser claramente intenso y el desenlace debe cerrar excelente (sin final tibio). "
        "PROHIBIDO cierre moralina, frase de autoayuda, o final genérico. "
        f"Rango ideal para historia_limpia_es: {min_chars_final} a {max_chars_final} caracteres.\n\n"
        f"SUBREDDIT: {candidate.get('subreddit', '')}\n"
        f"TÍTULO DEL POST: {candidate.get('post_title', '')}\n"
        f"TEXTO ORIGINAL:\n{body}"
    )

    try:
        result = llm_client.generate_json(
            prompt,
            temperature=0.15,
            max_tokens=2600,
            min_ctx=4096,
        )
    except Exception as e:
        _log(f"[STORY-IMPORT] ⚠️ IA no pudo evaluar historia: {e}")
        return None

    if not isinstance(result, dict):
        return None

    story_es = str(result.get("historia_limpia_es") or "").strip()
    pv = _parse_int(result.get("puntaje_viral"), 0)
    pg = _parse_int(result.get("puntaje_gancho"), 0)
    pc = _parse_int(result.get("puntaje_calidad"), 0)
    pcomp = _parse_int(result.get("puntaje_completitud"), 0)
    pclimax = _parse_int(result.get("puntaje_climax"), 0)
    pdesenlace = _parse_int(result.get("puntaje_desenlace"), 0)
    pintensidad = _parse_int(result.get("puntaje_intensidad"), 0)
    porigfinal = _parse_int(result.get("puntaje_originalidad_desenlace"), 0)
    accept = bool(result.get("aceptar"))
    is_long = bool(result.get("es_historia_larga"))
    is_complete = bool(result.get("es_historia_completa"))
    has_climax = bool(result.get("tiene_climax_fuerte"))
    has_resolution = bool(result.get("tiene_desenlace_contundente"))

    if not has_climax or not has_resolution:
        mk_climax, mk_resolution = _has_climax_and_resolution_markers(story_es)
        has_climax = has_climax or mk_climax
        has_resolution = has_resolution or mk_resolution

    if len(story_es) < min_chars_final:
        accept = False
    if len(story_es) > int(max_chars_final * 1.9):
        accept = False
    if pv < 90 or pg < 90 or pc < 88 or pcomp < 90:
        accept = False
    if not is_long or not is_complete:
        accept = False
    if pclimax < 90 or pdesenlace < 90:
        accept = False
    if pintensidad < 90 or porigfinal < 88:
        accept = False
    if not has_climax or not has_resolution:
        accept = False
    if not _style_gate(story_es):
        accept = False

    result["aceptar"] = accept
    result["puntaje_viral"] = pv
    result["puntaje_gancho"] = pg
    result["puntaje_calidad"] = pc
    result["puntaje_completitud"] = pcomp
    result["puntaje_climax"] = pclimax
    result["puntaje_desenlace"] = pdesenlace
    result["puntaje_intensidad"] = pintensidad
    result["puntaje_originalidad_desenlace"] = porigfinal
    result["tiene_climax_fuerte"] = has_climax
    result["tiene_desenlace_contundente"] = has_resolution
    result["historia_limpia_es"] = story_es
    return result


def _compactar_historia_si_larga(
    story_es: str,
    *,
    min_chars_final: int,
    max_chars_final: int,
) -> str:
    texto = (story_es or "").strip()
    if not texto:
        return ""
    if len(texto) <= max_chars_final:
        return texto

    prompt = (
        "Reescribe la historia en español para hacerla más corta y explosiva para video corto. "
        "Conserva hechos reales, clímax fuerte y desenlace contundente. "
        "El final NO puede ser moralina ni reflexión genérica; debe ser específico, impactante y memorable. "
        "No uses listas ni encabezados. "
        f"Entrega solo texto final entre {min_chars_final} y {max_chars_final} caracteres.\n\n"
        f"HISTORIA ORIGINAL:\n{texto}"
    )

    try:
        compacta = llm_client.generate(
            prompt,
            temperature=0.2,
            max_tokens=1100,
            min_ctx=3072,
        )
    except Exception as e:
        _log(f"[STORY-IMPORT] ⚠️ No se pudo compactar historia larga: {e}")
        return texto

    out = (compacta or "").strip()
    if not out:
        return texto
    if len(out) < min_chars_final:
        return texto
    if len(out) > max_chars_final:
        return out[:max_chars_final].rstrip()
    return out


def importar_historias_reddit(
    *,
    total: int,
    output_dir: str = os.path.join("historias", "Reddit Virales"),
    min_chars_origen: int = 700,
    min_chars_final: int = 700,
    max_chars_final: int = 1300,
    max_candidatos: int = 140,
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
    light_mode: bool = False,
) -> dict[str, Any]:
    total = max(1, int(total))
    target_candidates = max(total * 10, int(max_candidatos))

    asegurar_directorio(output_dir)
    topic_db.init_db()
    search_state = _load_search_state()

    active_subreddits = _effective_subreddits(subreddits, light_mode=light_mode)
    feed_profile = _feed_profile_from_env()
    if light_mode and feed_profile == "full":
        feed_profile = "light"

    _log(
        "[STORY-IMPORT] Buscando historias de Reddit candidatas... "
        f"(subreddits={len(active_subreddits)}, perfil={feed_profile}, oauth={'on' if _oauth_enabled() else 'off'})"
    )
    candidates = _collect_reddit_candidates(
        target_candidates=target_candidates,
        min_chars=min_chars_origen,
        subreddits=active_subreddits,
        feed_profile=feed_profile,
        state=search_state,
    )
    _save_search_state(search_state)

    if not candidates:
        if (not light_mode) and _reddit_rate_limit_detected():
            wait_s = max(2.0, float(settings.reddit_fallback_wait_s))
            _log(
                "[STORY-IMPORT] ℹ️ Activando fallback anti-429 "
                f"(espera {wait_s:.1f}s, menos subreddits/feeds)."
            )
            time.sleep(wait_s)
            return importar_historias_reddit(
                total=total,
                output_dir=output_dir,
                min_chars_origen=min_chars_origen,
                min_chars_final=min_chars_final,
                max_chars_final=max_chars_final,
                max_candidatos=max_candidatos,
                subreddits=subreddits,
                light_mode=True,
            )

        reason = "No se encontraron candidatos largos/virales en Reddit."
        if _reddit_rate_limit_detected():
            reason = (
                "Reddit respondió con rate limit (429) incluso en modo ligero. "
                "Reintenta en unos minutos o reduce subreddits vía REDDIT_IMPORT_SUBREDDITS."
            )
        return {
            "requested": total,
            "imported": 0,
            "evaluated": 0,
            "saved": [],
            "reason": reason,
            "cache": {
                "seen_posts": len((search_state.get("seen_post_ids") or {})),
                "seen_comments": len((search_state.get("seen_comment_ids") or {})),
            },
        }

    candidates = sorted(candidates, key=_heuristic_priority, reverse=True)

    imported = 0
    evaluated = 0
    saved: list[str] = []
    seen_comment_ids_map = search_state.get("seen_comment_ids") or {}
    if not isinstance(seen_comment_ids_map, dict):
        seen_comment_ids_map = {}
        search_state["seen_comment_ids"] = seen_comment_ids_map

    for cand in candidates:
        if imported >= total:
            break

        source_text = str(cand.get("comment_body") or "").strip()
        if not source_text:
            continue

        repeated_source = topic_db.find_similar_topic(
            source_text,
            kinds=(KIND_IMPORTED, "custom", "custom_pending"),
            threshold=0.92,
        )
        if repeated_source is not None:
            continue

        ai_eval = _evaluate_story_with_ai(
            cand,
            min_chars_final=min_chars_final,
            max_chars_final=max_chars_final,
        )
        evaluated += 1
        cid_eval = (cand.get("comment_id") or "").strip()
        if cid_eval:
            seen_comment_ids_map[cid_eval] = _now_ts()
        if not ai_eval or not ai_eval.get("aceptar"):
            continue

        historia_final = str(ai_eval.get("historia_limpia_es") or "").strip()
        historia_final = _compactar_historia_si_larga(
            historia_final,
            min_chars_final=min_chars_final,
            max_chars_final=max_chars_final,
        )
        if not historia_final:
            continue
        if len(historia_final) < min_chars_final:
            continue
        if not _style_gate(historia_final):
            continue

        repeated_final = topic_db.find_similar_topic(
            historia_final,
            kinds=(KIND_IMPORTED, "custom", "custom_pending"),
            threshold=0.90,
        )
        if repeated_final is not None:
            continue

        now = int(time.time())
        n = imported + 1
        base = f"reddit_story_{now}_{n}"
        txt_path = os.path.join(output_dir, f"{base}.txt")
        meta_path = os.path.join(output_dir, f"{base}.json")
        while os.path.exists(txt_path) or os.path.exists(meta_path):
            n += 1
            base = f"reddit_story_{now}_{n}"
            txt_path = os.path.join(output_dir, f"{base}.txt")
            meta_path = os.path.join(output_dir, f"{base}.json")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(historia_final.strip() + "\n")

        meta = {
            "source": "reddit",
            "created_at": now,
            "subreddit": cand.get("subreddit"),
            "post_id": cand.get("post_id"),
            "post_title": cand.get("post_title"),
            "post_score": cand.get("post_score"),
            "post_num_comments": cand.get("post_num_comments"),
            "permalink": cand.get("permalink"),
            "comment_id": cand.get("comment_id"),
            "comment_score": cand.get("comment_score"),
            "source_hash": _text_hash(source_text),
            "final_hash": _text_hash(historia_final),
            "ia": {
                "puntaje_viral": ai_eval.get("puntaje_viral"),
                "puntaje_gancho": ai_eval.get("puntaje_gancho"),
                "puntaje_calidad": ai_eval.get("puntaje_calidad"),
                "puntaje_completitud": ai_eval.get("puntaje_completitud"),
                "puntaje_climax": ai_eval.get("puntaje_climax"),
                "puntaje_desenlace": ai_eval.get("puntaje_desenlace"),
                "puntaje_intensidad": ai_eval.get("puntaje_intensidad"),
                "puntaje_originalidad_desenlace": ai_eval.get("puntaje_originalidad_desenlace"),
                "tiene_climax_fuerte": ai_eval.get("tiene_climax_fuerte"),
                "tiene_desenlace_contundente": ai_eval.get("tiene_desenlace_contundente"),
                "motivo": ai_eval.get("motivo"),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        topic_db.register_topic_if_new(
            historia_final,
            kind=KIND_IMPORTED,
            plan_dir=txt_path,
            threshold=0.90,
        )

        imported += 1
        saved.append(txt_path)
        _log(
            "[STORY-IMPORT] ✅ Historia importada "
            f"{imported}/{total} | r/{cand.get('subreddit', '')} | "
            f"viral={ai_eval.get('puntaje_viral', 0)}"
        )

    reason = ""
    if imported < total:
        if _reddit_rate_limit_detected():
            reason = "Reddit limitó solicitudes (429); no se pudieron obtener suficientes candidatos en esta corrida."
        else:
            reason = "No alcanzó suficientes historias que pasaran filtros estrictos de IA/no repetición."

    _save_search_state(search_state)

    return {
        "requested": total,
        "imported": imported,
        "evaluated": evaluated,
        "saved": saved,
        "reason": reason,
        "cache": {
            "seen_posts": len((search_state.get("seen_post_ids") or {})),
            "seen_comments": len((search_state.get("seen_comment_ids") or {})),
        },
    }
