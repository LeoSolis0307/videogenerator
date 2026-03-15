import json
import os
import sys
from typing import Any

import requests


def _mask(value: str, *, head: int = 4, tail: int = 2) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if len(v) <= head + tail:
        return "*" * len(v)
    return f"{v[:head]}...{v[-tail:]}"


def _get_env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _print_header() -> None:
    print("[OAUTH-CHECK] Reddit OAuth self-check")


def _print_config(client_id: str, client_secret: str, user_agent: str, timeout: float) -> None:
    print(f"[OAUTH-CHECK] REDDIT_CLIENT_ID: {'set' if client_id else 'missing'}")
    print(f"[OAUTH-CHECK] REDDIT_CLIENT_SECRET: {'set' if client_secret else 'missing'}")
    if client_id:
        print(f"[OAUTH-CHECK] client_id(masked): {_mask(client_id)}")
    if client_secret:
        print(f"[OAUTH-CHECK] client_secret(masked): {_mask(client_secret)}")
    print(f"[OAUTH-CHECK] user_agent: {user_agent}")
    print(f"[OAUTH-CHECK] timeout: {timeout:.1f}s")


def _request_token(client_id: str, client_secret: str, user_agent: str, timeout: float) -> dict[str, Any]:
    resp = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(client_id, client_secret),
        headers={"User-Agent": user_agent},
        data={"grant_type": "client_credentials"},
        timeout=timeout,
    )
    payload = {}
    try:
        payload = resp.json() if resp.content else {}
    except Exception:
        payload = {}
    return {"status": resp.status_code, "payload": payload, "text": resp.text}


def _probe_me(token: str, user_agent: str, timeout: float) -> dict[str, Any]:
    resp = requests.get(
        "https://oauth.reddit.com/api/v1/me",
        headers={"Authorization": f"bearer {token}", "User-Agent": user_agent, "Accept": "application/json"},
        timeout=timeout,
    )
    payload = {}
    try:
        payload = resp.json() if resp.content else {}
    except Exception:
        payload = {}
    return {"status": resp.status_code, "payload": payload, "text": resp.text}


def main() -> int:
    _print_header()
    client_id = _get_env("REDDIT_CLIENT_ID")
    client_secret = _get_env("REDDIT_CLIENT_SECRET")
    user_agent = _get_env("REDDIT_USER_AGENT") or "videogenerator/1.0 (oauth check; contacto: local)"
    timeout = float(_get_env("REDDIT_TIMEOUT_SEC") or "20")

    _print_config(client_id, client_secret, user_agent, timeout)

    if not client_id or not client_secret:
        print("[OAUTH-CHECK] ❌ Faltan variables REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET")
        return 2

    try:
        token_res = _request_token(client_id, client_secret, user_agent, timeout)
    except Exception as e:
        print(f"[OAUTH-CHECK] ❌ Error solicitando token: {e}")
        return 3

    status = int(token_res.get("status") or 0)
    payload = token_res.get("payload") or {}

    if status != 200:
        print(f"[OAUTH-CHECK] ❌ Token request HTTP {status}")
        if payload:
            print("[OAUTH-CHECK] payload:")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 4

    token = str(payload.get("access_token") or "").strip()
    token_type = str(payload.get("token_type") or "").strip()
    scope = str(payload.get("scope") or "").strip()
    expires_in = payload.get("expires_in")

    if not token:
        print("[OAUTH-CHECK] ❌ No llegó access_token en la respuesta")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 5

    print("[OAUTH-CHECK] ✅ Token OAuth obtenido")
    print(f"[OAUTH-CHECK] token_type: {token_type or 'unknown'} | scope: {scope or '(none)'} | expires_in: {expires_in}")

    try:
        probe = _probe_me(token, user_agent, timeout)
    except Exception as e:
        print(f"[OAUTH-CHECK] ⚠️ No se pudo probar oauth endpoint: {e}")
        return 0

    pstatus = int(probe.get("status") or 0)
    if pstatus in (200, 401, 403):
        print(f"[OAUTH-CHECK] ✅ oauth.reddit.com responde (HTTP {pstatus})")
        return 0

    print(f"[OAUTH-CHECK] ⚠️ oauth.reddit.com devolvió HTTP {pstatus}")
    ppayload = probe.get("payload") or {}
    if ppayload:
        print(json.dumps(ppayload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
