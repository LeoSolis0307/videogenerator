import os
from typing import Any


def _env_flag(name: str) -> bool:
    val = (os.environ.get(name) or "").strip().lower()
    return val in {"1", "true", "yes", "y", "si", "sí", "on"}


def _env_false(name: str) -> bool:
    val = (os.environ.get(name) or "").strip().lower()
    return val in {"0", "false", "no", "n", "off"}


def _print_speed_default_enabled() -> bool:
    if "OLLAMA_PRINT_SPEED" in os.environ and _env_false("OLLAMA_PRINT_SPEED"):
        return False
    if "OLLAMA_SHOW_SPEED" in os.environ and _env_false("OLLAMA_SHOW_SPEED"):
        return False
    if "OLLAMA_PRINT_SPEED" in os.environ:
        return _env_flag("OLLAMA_PRINT_SPEED")
    if "OLLAMA_SHOW_SPEED" in os.environ:
        return _env_flag("OLLAMA_SHOW_SPEED")
    return True


_PRINT_SPEED = _print_speed_default_enabled()


def ollama_tokens_per_second(response_json: dict[str, Any]) -> float | None:
    if not isinstance(response_json, dict):
        return None

    try:
        total_tokens = int(response_json.get("eval_count") or 0)
        duration_ns = int(response_json.get("eval_duration") or 0)
    except Exception:
        return None

    if total_tokens <= 0 or duration_ns <= 0:
        return None

    seconds = duration_ns / 1_000_000_000
    if seconds <= 0:
        return None

    return total_tokens / seconds


def _diagnostic_for_speed(tps: float) -> str:
    if tps > 20:
        return "OK: GPU fast (likely VRAM)"
    if tps > 8:
        return "WARN: GPU ok (may be using some RAM)"
    return "BAD: Bottleneck (GPU waiting on RAM/CPU)"


def maybe_print_ollama_speed(response_json: dict[str, Any], *, tag: str = "OLLAMA", enabled: bool | None = None) -> None:
    if enabled is None:
        enabled = _PRINT_SPEED
    if not enabled:
        return

    tps = ollama_tokens_per_second(response_json)
    if not tps:
        return

    try:
        total_tokens = int(response_json.get("eval_count") or 0)
        duration_ns = int(response_json.get("eval_duration") or 0)
        seconds = duration_ns / 1_000_000_000
    except Exception:
        total_tokens = 0
        seconds = 0.0

    print(f"[{tag}] Speed: {tps:.2f} tokens/sec")
    if total_tokens and seconds:
        print(f"[{tag}]   eval_count={total_tokens} eval_seconds={seconds:.2f}")
    print(f"[{tag}]   {_diagnostic_for_speed(tps)}")
