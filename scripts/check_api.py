"""Verify the Gemini key with ONE tiny API call. Never prints the key."""
import os, sys
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
key, model = os.getenv("GEMINI_API_KEY", "").strip(), os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
if not key:
    sys.exit(f"✗ GEMINI_API_KEY is empty in {ROOT/'.env'}")
if key.startswith(('"', "'")) or " " in key:
    sys.exit("✗ Remove quotes/spaces around the key in .env")
print(f"key loaded ({len(key)} chars) · model = {model}")

from google import genai
from google.genai import types, errors
client = genai.Client(api_key=key)
try:
    r = client.models.generate_content(
        model=model, contents='Reply with exactly this JSON: {"ok": true}',
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
    print("✓ API call succeeded:", r.text.strip())
    u = r.usage_metadata
    if u: print(f"  tokens in/out: {u.prompt_token_count}/{u.candidates_token_count}")
except errors.ClientError as e:
    msg = {400: "bad request / key invalid", 401: "key rejected", 403: "key lacks permission or API not enabled",
           404: f"model '{model}' not found — set GEMINI_MODEL in .env to a Flash model listed in AI Studio",
           429: "rate limit / quota hit — wait a minute and retry"}.get(e.code, "")
    sys.exit(f"✗ HTTP {e.code}: {msg}\n  {e.message}")
except Exception as e:
    sys.exit(f"✗ {type(e).__name__}: {e}")
