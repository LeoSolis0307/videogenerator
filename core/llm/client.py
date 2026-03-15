import json
import re
import ast
import requests
import time
from typing import Any
from core.config import settings
from core.ollama_metrics import maybe_print_ollama_speed

class OllamaClient:
    def __init__(self, session: requests.Session | None = None):
        self._session = session or requests.Session()

    def _get_extra_options(self) -> dict:
        return settings.ollama_options

    def _raise_http_error(self, resp: requests.Response, model: str):
        body = (resp.text or "").strip()[:1500]
        hint = ""
        low = body.lower()
        if "model" in low and ("not found" in low or "no such" in low):
            hint = f"\n[LLM] 💡 Hint: Model '{model}' not found. Run `ollama pull {model}`."
        elif "out of memory" in low or "oom" in low or "cuda" in low:
             hint = "\n[LLM] 💡 Hint: OOM / VRAM error. Try a smaller model."
        
        msg = f"Ollama HTTP {resp.status_code} with model '{model}'. {body} {hint}"
        raise RuntimeError(msg)

    def generate(self, prompt: str, *, temperature: float = 0.65, max_tokens: int = 900,
                 timeout_sec: float | None = None, model: str | None = None,
                 min_ctx: int | None = None, json_mode: bool = False) -> str:
        
        model_name = model or settings.ollama_text_model_short
        timeout = timeout_sec or settings.ollama_timeout
        
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        
        # Merge extra options
        extra = self._get_extra_options()
        if "num_ctx" not in extra:
            default_ctx = min_ctx if min_ctx else settings.ollama_text_num_ctx
            options["num_ctx"] = max(256, default_ctx)
        options.update(extra)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        if int(settings.ollama_keep_alive) > 0:
            payload["keep_alive"] = int(settings.ollama_keep_alive)

        try:
            resp = self._session.post(settings.ollama_url, json=payload, timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Could not connect to Ollama at {settings.ollama_url}") from e

        if resp.status_code >= 400:
            self._raise_http_error(resp, model_name)
        
        try:
            data = resp.json()
        except Exception as e:
            body = (resp.text or "").strip()[:1500]
            raise RuntimeError(f"Ollama returned non-JSON response for model '{model_name}'. {body}") from e
        maybe_print_ollama_speed(data, tag="LLM")
        return (data.get("response") or "").strip()

    def generate_json(self, prompt: str, *, temperature: float = 0.65, max_tokens: int = 900,
                      timeout_sec: float | None = None, model: str | None = None, min_ctx: int | None = None) -> Any:
        # Retry logic for JSON
        last_err = None
        current_model = model or settings.ollama_text_model_short
        last_raw = ""
        
        for attempt in range(3):
            try:
                raw = self.generate(prompt, temperature=temperature, max_tokens=max_tokens, 
                                  timeout_sec=timeout_sec, model=current_model, min_ctx=min_ctx, json_mode=True)
                last_raw = raw
                return self._extract_json_value(raw)
            except Exception as e:
                last_err = e
                # Simple retry logic, could be more sophisticated like in original code
                print(f"[LLM] ⚠️ JSON generation failed (attempt {attempt+1}): {e}")

                low = str(e).lower()
                is_timeout = ("timed out" in low) or ("read timeout" in low) or ("timeout" in low)
                short_model = (settings.ollama_text_model_short or "").strip()
                if is_timeout and short_model and current_model != short_model:
                    print(f"[LLM] ↩️ Fallback de modelo por timeout: {current_model} -> {short_model}")
                    current_model = short_model
                    time.sleep(0.2)
                    continue
                
                # If it's a parse error, try to fix it
                if "JSON" in str(e) or "parse" in str(e):
                    fix_prompt = (
                        f"You returned INVALID JSON. Return ONLY valid JSON, no prose, no markdown fences.\n"
                        f"ERROR: {e}\n"
                        f"INVALID_OUTPUT:\n{(last_raw or str(last_err))[:4000]}\n"
                    )
                    try:
                        raw = self.generate(fix_prompt, temperature=0.2, max_tokens=max_tokens,
                                          timeout_sec=timeout_sec, model=current_model, min_ctx=min_ctx, json_mode=True)
                        last_raw = raw
                        return self._extract_json_value(raw)
                    except Exception:
                        pass

                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
                        
        raise last_err or RuntimeError("Failed to generate valid JSON")

    def _extract_json_value(self, raw: str) -> Any:
        raw = (raw or "").strip()
        # Remove markdown
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
        
        if not raw:
            raise ValueError("Empty LLM response")
            
        try:
            return json.loads(raw)
        except Exception:
            pass

        fenced_chunks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
        for chunk in fenced_chunks:
            parsed = self._try_parse_json_chunk(chunk)
            if parsed is not None:
                return parsed
            
        parsed_full = self._try_parse_json_chunk(raw)
        if parsed_full is not None:
            return parsed_full

        for chunk in self._iter_balanced_json_candidates(raw):
            parsed = self._try_parse_json_chunk(chunk)
            if parsed is not None:
                return parsed
                
        raise ValueError("Could not parse JSON from response")

    def _try_parse_json_chunk(self, text: str) -> Any | None:
        chunk = (text or "").strip()
        if not chunk:
            return None
        chunk = self._normalize_json_like_text(chunk)
        chunk = re.sub(r",\s*([\]}])", r"\1", chunk)
        try:
            return json.loads(chunk)
        except Exception:
            pass

        py_like = re.sub(r"\btrue\b", "True", chunk, flags=re.IGNORECASE)
        py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
        py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
        try:
            val = ast.literal_eval(py_like)
        except Exception:
            return None
        if isinstance(val, (dict, list)):
            return val
        return None

    def _normalize_json_like_text(self, text: str) -> str:
        t = (text or "").strip().lstrip("\ufeff")
        if not t:
            return t

        t = t.replace("\u201c", '"').replace("\u201d", '"')
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = re.sub(r"^\s*json\s*[:\-]?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"/\*.*?\*/", "", t, flags=re.DOTALL)
        t = re.sub(r"(?m)^\s*//.*$", "", t)
        return t.strip()

    def _iter_balanced_json_candidates(self, text: str):
        starts = []
        for idx, char in enumerate(text):
            if char in "[{":
                starts.append(idx)

        for start in starts:
            opener = text[start]
            closer = "}" if opener == "{" else "]"
            depth = 0
            in_string = False
            escaped = False
            for end in range(start, len(text)):
                ch = text[end]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue

                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        yield text[start:end + 1]
                        break

llm_client = OllamaClient()
