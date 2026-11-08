"""Build public_deploy/: the folder to push to a PUBLIC GitHub repo for Streamlit Community Cloud.

  python scripts/export_app_data.py --all
  python scripts/make_public_bundle.py
Contains: app.py, promo/ (code only), app_data/ (sanitized), requirements.txt, .streamlit/config.toml, README.
Never contains: raw screenplays, outputs/ (full excerpts), Pass A / Cold Viewer files, gold files, feedback.jsonl, .env, cache.
Fails if any screenplay line other than a promo's hook or cliffhanger line appears anywhere in the bundle.
"""
import json, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from promo.show import load_registry, _abs

OUT = ROOT / "public_deploy"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
shutil.copy2(ROOT / "app.py", OUT / "app.py")
shutil.copytree(ROOT / "promo", OUT / "promo", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(ROOT / "app_data", OUT / "app_data")
shutil.copytree(ROOT / "scripts", OUT / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.html"))   # live pipeline code
(OUT / "config" / "weights").mkdir(parents=True)
shutil.copytree(ROOT / "config" / "weights" / "global", OUT / "config" / "weights" / "global")
shutil.copy2(ROOT / "config" / "weights" / "registry.json", OUT / "config" / "weights" / "registry.json")
(OUT / "requirements.txt").write_text("streamlit>=1.37\naltair>=5\npandas\npydantic>=2\nnumpy\npython-dotenv\ngoogle-genai\n", encoding="utf-8")
(OUT / ".streamlit").mkdir()
(OUT / ".streamlit" / "config.toml").write_text('[server]\nheadless = true\nmaxUploadSize = 100\n\n[theme]\nbase = "light"\nprimaryColor = "#2a78d6"\n', encoding="utf-8")
(OUT / ".public_mode").write_text("This bundle always runs in public mode (no screenplay excerpts).\n", encoding="utf-8")
(OUT / ".gitignore").write_text(".env\n__pycache__/\ndata/\n", encoding="utf-8")
(OUT / "README.md").write_text("""# Promo Finder (public demo)

Streamlit app over precomputed promo rankings for microdrama shows.
Set the environment variable / Streamlit secret `PUBLIC_MODE=1`: full screenplay excerpts are not included in this repo
and are hidden; timestamps, hook and cliffhanger lines, scores, rankings, explanations, evaluation, failure analysis and
feedback-learning status are shown. Reviews made here go to a separate demo file and never change the ranking.
Run locally: `pip install -r requirements.txt` then `PUBLIC_MODE=1 streamlit run app.py`.

**Analyze New Input** runs the real pipeline on uploaded `.txt` screenplays or a `.zip` of them, in a temporary workspace
outside this repository. It needs the server-side environment variable `GEMINI_API_KEY` (optionally `GEMINI_MODEL`,
`LLM_MIN_INTERVAL`); without it the pre-computed demo still works and the live page explains what is missing.
""", encoding="utf-8")

# ---- leak check: no screenplay line except hook / cliffhanger lines may appear
allowed = set()
for sid in json.loads((ROOT / "app_data" / "shows.json").read_text(encoding="utf-8"))["shows"]:
    for p in json.loads((ROOT / "app_data" / sid / "promos.json").read_text(encoding="utf-8")):
        allowed |= {p["hook"]["text"].strip(), p["cliffhanger"]["text"].strip()}
bundle = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in OUT.rglob("*") if f.is_file() and f.suffix in (".py", ".json", ".md", ".toml", ".txt"))
bundle_n = re.sub(r"\\u[0-9a-f]{4}|\\\"", " ", bundle)
leaks, checked = [], 0
for s in load_registry()["shows"].values():
    for e in s["episodes"]:
        raw = _abs(e["source_file"]).read_text(encoding="utf-8").splitlines()
        for i, l in enumerate(raw):
            l = l.strip()
            if len(l) < 40 or l.startswith(("**", "Start:", "EPISODE", "Scene", "SCENE", "---", "CONTINUITY")) or l in allowed:
                continue
            checked += 1
            probe = l[:60]
            if probe in bundle or json.dumps(probe)[1:-1] in bundle:
                leaks.append(f"{e['episode_id']} line {i + 1}: {probe}")
if leaks:
    shutil.rmtree(OUT)
    sys.exit(f"LEAK CHECK FAILED ({len(leaks)} screenplay lines found in the bundle), bundle removed:\n" + "\n".join(leaks[:10]))
size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
print(f"public_deploy/ ready ({size // 1024} KB). Leak check passed: {checked} screenplay lines checked, 0 found "
      f"(only {len(allowed)} hook/cliffhanger lines are shown, by design).")
