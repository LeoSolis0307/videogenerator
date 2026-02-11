import json
import re
import requests
import time
from typing import Any, Dict, List, Optional
from core.config import settings
from core.ollama_metrics import maybe_print_ollama_speed

class OllamaClient:
    def __init__(self):
        pass

    def _get_extra_options(self) -> dict:
        if not settings.ollama_options_json:
            return {}
        try:
            return json.loads(settings.ollama_options_json)
        except Exception:
            return {}

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
                 timeout_sec: float | None = None, model: str | None = None, min_ctx: int | None = None) -> str:
        
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

        try:
            resp = requests.post(settings.ollama_url, json=payload, timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Could not connect to Ollama at {settings.ollama_url}") from e

        if resp.status_code >= 400:
            self._raise_http_error(resp, model_name)
        
        data = resp.json()
        maybe_print_ollama_speed(data, tag="LLM")
        return (data.get("response") or "").strip()

    def generate_json(self, prompt: str, *, temperature: float = 0.65, max_tokens: int = 900,
                      timeout_sec: float | None = None, model: str | None = None, min_ctx: int | None = None) -> Any:
        # Retry logic for JSON
        last_err = None
        current_model = model or settings.ollama_text_model_short
        
        for attempt in range(3):
            try:
                raw = self.generate(prompt, temperature=temperature, max_tokens=max_tokens, 
                                  timeout_sec=timeout_sec, model=current_model, min_ctx=min_ctx)
                return self._extract_json_value(raw)
            except Exception as e:
                last_err = e
                # Simple retry logic, could be more sophisticated like in original code
                print(f"[LLM] ⚠️ JSON generation failed (attempt {attempt+1}): {e}")
                
                # If it's a parse error, try to fix it
                if "JSON" in str(e) or "parse" in str(e):
                    fix_prompt = (
                        f"You returned INVALID JSON. Fix it and return ONLY valid JSON.\n"
                        f"ERROR: {e}\n"
                        f"INVALID_OUTPUT: {str(last_err)[:2000]}\n"
                    )
                    try:
                        raw = self.generate(fix_prompt, temperature=0.2, max_tokens=max_tokens,
                                          timeout_sec=timeout_sec, model=current_model, min_ctx=min_ctx)
                        return self._extract_json_value(raw)
                    except Exception:
                        pass
                        
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
            
        # Try to find { ... } or [ ... ]
        o_s = raw.find("{")
        o_e = raw.rfind("}")
        if o_s != -1 and o_e > o_s:
            try:
                chunk = raw[o_s : o_e + 1]
                # Common fix: remove trailing commas before closing braces
                chunk = re.sub(r",\s*([\]}])", r"\1", chunk)
                return json.loads(chunk)
            except Exception:
                pass

        a_s = raw.find("[")
        a_e = raw.rfind("]")
        if a_s != -1 and a_e > a_s:
            try:
                chunk = raw[a_s : a_e + 1]
                chunk = re.sub(r",\s*([\]}])", r"\1", chunk)
                return json.loads(chunk)
            except Exception:
                pass
                
        raise ValueError("Could not parse JSON from response")

llm_client = OllamaClient()
