"""Provider-agnostic JSON LLM client: validation, retries, rate limiting, disk cache.

Providers (set in .env):
  LLM_PROVIDER=gemini         (default)  GEMINI_API_KEY, GEMINI_MODEL
  LLM_PROVIDER=openai_compat             LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
  LLM_PROVIDER=offline                   no key; heuristic stand-in (promo/offline_llm.py), lower quality
      (any OpenAI-compatible /chat/completions endpoint: OpenAI, Groq, OpenRouter, Together, local Ollama)

Every successful response is cached under cache/llm/<sha256>.json, keyed by
(model, prompt version, system prompt, user prompt; plus provider for non-Gemini). Re-running a step skips cached calls,
so an interrupted run resumes where it stopped and no successful call is ever repeated.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Type

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class LLMError(RuntimeError):
    pass


class TransientError(RuntimeError):
    def __init__(self, code, msg, retry_after=None):
        super().__init__(f"HTTP {code}: {msg}")
        self.code, self.retry_after = code, retry_after


TRANSIENT = {408, 429, 500, 502, 503, 504}


def _extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    return text[start:]


class LLMJSON:
    def __init__(self, model: str | None = None, cache_dir: Path | None = None,
                 min_interval: float | None = None, max_retries: int = 5, client=None,
                 provider: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
        if self.provider == "gemini":
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        else:
            self.model = model or os.getenv("LLM_MODEL", "heuristic" if self.provider == "offline" else "")
        self.cache_dir = Path(cache_dir or ROOT / "cache" / "llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL", "4") if min_interval is None else min_interval)
        if self.provider == "offline" and min_interval is None:
            self.min_interval = 0.0
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", max_retries))
        self._last = 0.0
        self._client = client  # injectable for tests (Gemini-shaped: client.models.generate_content)
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0, "in_tokens": 0, "out_tokens": 0,
                      "consecutive_quota": 0}

    @property
    def name(self):
        return f"{self.provider}:{self.model}"

    # ---------------------------------------------------------- providers
    def _gemini(self, system, user, max_tokens):
        from google.genai import types, errors
        if self._client is None:
            key = os.getenv("GEMINI_API_KEY", "").strip()
            if not key:
                raise LLMError(f"GEMINI_API_KEY missing in {ROOT / '.env'}")
            from google import genai
            self._client = genai.Client(api_key=key)
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=0,
                                          response_mime_type="application/json", max_output_tokens=max_tokens)
        try:
            r = self._client.models.generate_content(model=self.model, contents=user, config=cfg)
        except errors.APIError as e:
            code = getattr(e, "code", 0) or 0
            if code in TRANSIENT:
                m = re.search(r"retry in ([\d.]+)s", str(e), re.I)
                raise TransientError(code, str(e)[:300], float(m.group(1)) + 1 if m else None)
            raise LLMError(f"HTTP {code}: {e}") from e
        u = getattr(r, "usage_metadata", None)
        if u:
            self.stats["in_tokens"] += u.prompt_token_count or 0
            self.stats["out_tokens"] += u.candidates_token_count or 0
        return r.text or ""

    def _openai_compat(self, system, user, max_tokens):
        base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.getenv("LLM_API_KEY", "").strip()
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {key}"} if key else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")[:300]
            if e.code in TRANSIENT:
                ra = e.headers.get("retry-after")
                raise TransientError(e.code, msg, float(ra) + 1 if ra and ra.replace(".", "").isdigit() else None)
            raise LLMError(f"HTTP {e.code}: {msg}") from e
        except urllib.error.URLError as e:
            raise TransientError(0, str(e))
        u = data.get("usage") or {}
        self.stats["in_tokens"] += u.get("prompt_tokens", 0)
        self.stats["out_tokens"] += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"] or ""

    def _call(self, system, user, max_tokens):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        self.stats["calls"] += 1
        if self.provider == "gemini":
            return self._gemini(system, user, max_tokens)
        if self.provider == "openai_compat":
            return self._openai_compat(system, user, max_tokens)
        if self.provider == "offline":
            from .offline_llm import answer
            return answer(system, user)
        raise LLMError(f"unknown LLM_PROVIDER {self.provider!r}")

    # ---------------------------------------------------------- public
    def _key(self, version, system, user):
        parts = [self.model, version, system, user]          # Gemini: original key format (keeps old cache valid)
        if self.provider != "gemini":
            parts = [self.provider] + parts
        h = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def cached(self, system, user, version) -> bool:
        return self._key(version, system, user).exists()

    def ask(self, system: str, user: str, schema: Type[BaseModel], version: str,
            max_tokens: int = 8192, validate=None, tag: str = "") -> BaseModel:
        """Return a validated `schema` instance. `validate(obj) -> list[str]` adds semantic checks;
        problems are sent back to the model on retry."""
        path = self._key(version, system, user)
        if path.exists():
            try:
                obj = schema.model_validate(json.loads(path.read_text(encoding="utf-8"))["parsed"])
                self.stats["cache_hits"] += 1
                return obj
            except Exception:
                path.unlink()  # corrupt / schema changed -> recompute

        prompt, last_err = user, None
        for attempt in range(self.max_retries):
            try:
                text = self._call(system, prompt, max_tokens)
                self.stats["consecutive_quota"] = 0
                obj = schema.model_validate_json(_extract_json(text))
                problems = validate(obj) if validate else []
                if problems and attempt < self.max_retries - 1:
                    raise ValueError("; ".join(problems[:20]))
                path.write_text(json.dumps({"provider": self.provider, "model": self.model, "version": version,
                                            "tag": tag, "problems": problems, "raw": text,
                                            "parsed": obj.model_dump()}, ensure_ascii=False, indent=1),
                                encoding="utf-8")
                return obj
            except TransientError as e:
                last_err = e
                if e.code == 429:
                    self.stats["consecutive_quota"] += 1
                delay = e.retry_after or min(60, 5 * 2 ** attempt)
                print(f"    [{tag}] {e.code or 'network'} error, waiting {delay:.0f}s")
                time.sleep(delay)
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                last_err = e
                msg = str(e).replace("\n", " ")[:1500]
                print(f"    [{tag}] invalid output (attempt {attempt+1}): {msg[:160]}")
                prompt = user + f"\n\nYOUR PREVIOUS ANSWER WAS INVALID: {msg}\nReturn corrected JSON only."
            self.stats["retries"] += 1
        raise LLMError(f"[{tag}] failed after {self.max_retries} attempts: {last_err}")


GeminiJSON = LLMJSON  # backwards-compatible name used by Pass A

