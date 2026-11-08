"""Live new input: run the EXISTING pipeline on uploaded screenplays, isolated from the demo.

    files = [("ep1.txt", b"..."), ("batch.zip", b"...")]
    job = start_job(files, mode="new", show_name="My Show")      # returns immediately; work runs in a thread
    job.status()                                                   # progress for the UI
    job.results()                                                  # promos / manifest / report when done

Isolation: every job gets a throwaway workspace in the system temp folder (never inside the project or Git):
  <tmp>/promo_live/<job_id>/ {promo/, scripts/, config/weights/, data/shows.json, uploads/<show folder>/*.txt, shows/...}
The workspace has its own show registry, feedback store and LLM cache, so a live job can never modify the demo's
app_data, outputs, registry, feedback or weights. Steps are the unchanged project scripts, run as subprocesses:
ingest -> parse_all -> run_pass_a --all -> run_candidates -> run_cold_viewer [--top-per-episode K] -> run_final -> evaluate
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = Path(tempfile.gettempdir()) / "promo_live"
# upload limits (generous enough for a ~100-episode ZIP; small enough for a free-tier server)
MAX_TXT_FILES = 300
MAX_FILE_BYTES = 2 * 1024 * 1024            # per screenplay
MAX_TOTAL_BYTES = 80 * 1024 * 1024          # all extracted text
MAX_ZIP_MEMBERS = 3000
MAX_RATIO = 200                             # zip-bomb guard (uncompressed / compressed)
JOB_TTL_S = 3 * 3600
_JOBS: dict[str, "LiveJob"] = {}
_LOCK = threading.Lock()


class UploadError(ValueError):
    pass


# ---------------------------------------------------------------- safe input handling
def _decode(name: str, data: bytes) -> tuple[str | None, str]:
    if b"\x00" in data[:4096]:
        return None, "binary content, not a text screenplay"
    for enc in ("utf-8-sig", "cp1252"):
        try:
            txt = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return None, "not valid UTF-8/Windows text"
    if not txt.strip():
        return None, "empty file"
    return txt, ""


def _safe_member_name(name: str) -> str | None:
    """Return a safe relative POSIX path, or None if the member is unsafe (absolute, traversal, drive letter)."""
    n = name.replace("\\", "/")
    if n.startswith("/") or re.match(r"^[A-Za-z]:", n):
        return None
    parts = [p for p in PurePosixPath(n).parts if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


def collect_inputs(files: list[tuple[str, bytes]]) -> dict:
    """Turn uploaded .txt files and/or .zip archives into {basename: text}. Never writes or executes anything.
    Returns {"texts": {name: text}, "folder_hint": str|None, "skipped": [(name, why)], "rejected": [(name, why)]}."""
    texts, skipped, rejected, folders = {}, [], [], set()
    total = 0

    def add(name, data, folder=None):
        nonlocal total
        base = PurePosixPath(name).name
        if len(data) > MAX_FILE_BYTES:
            rejected.append((name, f"larger than {MAX_FILE_BYTES // 1024} KB")); return
        txt, why = _decode(name, data)
        if txt is None:
            rejected.append((name, why)); return
        if base in texts:
            rejected.append((name, f"duplicate file name '{base}' (kept the first one)")); return
        if len(texts) >= MAX_TXT_FILES:
            rejected.append((name, f"more than {MAX_TXT_FILES} screenplays")); return
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise UploadError(f"uploaded text exceeds {MAX_TOTAL_BYTES // (1024 * 1024)} MB")
        texts[base] = txt
        if folder:
            folders.add(folder)

    for fname, data in files:
        low = fname.lower()
        if low.endswith(".txt"):
            add(fname, data)
        elif low.endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                rejected.append((fname, "not a valid ZIP file")); continue
            infos = zf.infolist()
            if len(infos) > MAX_ZIP_MEMBERS:
                raise UploadError(f"{fname}: more than {MAX_ZIP_MEMBERS} entries")
            for info in infos:
                if info.is_dir():
                    continue
                safe = _safe_member_name(info.filename)
                if safe is None:
                    rejected.append((info.filename, "unsafe path in ZIP (absolute or '..'); ignored")); continue
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    rejected.append((safe, "symbolic link in ZIP; ignored")); continue
                parts = safe.split("/")
                if parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts):
                    skipped.append((safe, "system/hidden file")); continue
                if not safe.lower().endswith(".txt"):
                    skipped.append((safe, "not a .txt file")); continue
                if info.file_size > MAX_FILE_BYTES:
                    rejected.append((safe, f"larger than {MAX_FILE_BYTES // 1024} KB")); continue
                if info.compress_size and info.file_size / max(1, info.compress_size) > MAX_RATIO and info.file_size > 1_000_000:
                    rejected.append((safe, "suspicious compression ratio; ignored")); continue
                with zf.open(info) as fh:
                    blob = fh.read(MAX_FILE_BYTES + 1)
                folder = parts[-2] if len(parts) > 1 else PurePosixPath(fname).stem
                add(safe, blob, folder)
        else:
            skipped.append((fname, "only .txt and .zip are accepted"))
    if not texts:
        raise UploadError("no usable .txt screenplays found" + (f" (rejected: {'; '.join(f'{n}: {w}' for n, w in rejected[:5])})" if rejected else ""))
    folder_hint = next(iter(folders)) if len(folders) == 1 else None
    return {"texts": texts, "folder_hint": folder_hint, "skipped": skipped, "rejected": rejected}


# ---------------------------------------------------------------- workspace + job
def _workspace(job_id: str, existing_show: dict | None) -> Path:
    ws = LIVE_ROOT / job_id
    shutil.copytree(ROOT / "promo", ws / "promo", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "scripts", ws / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    wsrc = ROOT / "config" / "weights"
    (ws / "config" / "weights").mkdir(parents=True)
    if (wsrc / "global").exists():
        shutil.copytree(wsrc / "global", ws / "config" / "weights" / "global")
    reg = {"global": "v0", "shows": {}}
    if (wsrc / "registry.json").exists():
        full = json.loads((wsrc / "registry.json").read_text(encoding="utf-8"))
        reg["global"] = full.get("global", "v0")
        if existing_show and existing_show["show_id"] in full.get("shows", {}):     # that show's own promoted weights only
            sid = existing_show["show_id"]
            reg["shows"][sid] = full["shows"][sid]
            shutil.copytree(wsrc / sid, ws / "config" / "weights" / sid)
    (ws / "config" / "weights" / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
    (ws / "data").mkdir()
    shows = {"default_show": None, "shows": {}}
    if existing_show:          # continuation: same show identity, no earlier screenplays needed
        sid = existing_show["show_id"]
        shows = {"default_show": sid, "shows": {sid: {"show_id": sid, "show_name": existing_show["show_name"],
                                                      "work_dir": f"shows/{sid}/work", "outputs_dir": f"shows/{sid}/outputs",
                                                      "held_out_episodes": [], "episodes": []}}}
    (ws / "data" / "shows.json").write_text(json.dumps(shows), encoding="utf-8")
    return ws


@dataclass
class LiveJob:
    job_id: str
    workspace: Path
    n_files: int
    mode: str
    show_name: str | None
    existing_show: dict | None
    cold_viewer_top: int | None
    provider: str
    state: str = "queued"                  # queued | running | done | error
    step: str = ""
    done_eps: int = 0
    total_eps: int = 0
    log: list = field(default_factory=list)
    error: str = ""
    show_id: str | None = None
    ingest_report: dict = field(default_factory=dict)
    upload_report: dict = field(default_factory=dict)
    failed_eps: list = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float | None = None

    # ------------------------------------------------------------
    def _env(self):
        env = {**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONIOENCODING": "utf-8", "LLM_PROVIDER": self.provider}
        env.setdefault("LLM_MAX_RETRIES", "3")
        return env

    def _run(self, *args, on_line=None, ok_codes=(0,)):
        p = subprocess.Popen([sys.executable, "-u", *args], cwd=self.workspace, env=self._env(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        tail = []
        for line in p.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-40:]
            if on_line:
                on_line(line)
        code = p.wait()
        if code not in ok_codes:
            raise RuntimeError(f"{args[0]} failed (exit {code}): " + " | ".join(l for l in tail[-6:] if l))
        return code

    def _say(self, msg):
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")

    def run(self):
        self.state = "running"
        try:
            up = self.workspace / "uploads"
            folder = up / (self.upload_report.get("folder") or "upload")
            # ---- 1. identify show + episodes (existing ingest logic)
            self.step = "Identifying show and episodes"
            args = ["scripts/ingest.py", str(folder)]
            if self.existing_show:
                args += ["--show", self.existing_show["show_id"]]
            elif self.show_name:
                args += ["--show-name", self.show_name]
            out = []
            code = self._run(*args, on_line=out.append, ok_codes=(0, 1))
            text = "\n".join(out)
            if code != 0 or "NOT INGESTED" in text:
                msg = (text.split("NOT INGESTED:", 1)[-1].strip() or text)
                raise RuntimeError(msg.replace("Pass --show-name.", "Enter the show name in the 'Show name' box and analyze again.")
                                      .replace("pass --show-name", "enter the show name"))
            reg = json.loads((self.workspace / "data" / "shows.json").read_text(encoding="utf-8"))
            self.show_id = reg["default_show"]
            s = reg["shows"][self.show_id]
            self.ingest_report = {"show_id": self.show_id, "show_name": s["show_name"],
                                  "episodes": [e["episode_number"] for e in s["episodes"]],
                                  "flags": [l.strip() for l in out if "FLAG" in l]}
            self.total_eps = len(s["episodes"])
            if not self.total_eps:
                raise RuntimeError("no episode could be identified. " + " ".join(self.ingest_report["flags"]))
            self._say(f"{s['show_name']}: {self.total_eps} episodes identified {self.ingest_report['episodes'][:20]}")
            for f in self.ingest_report["flags"]:
                self._say(f)
            sh = ["--show", self.show_id]
            self.step = "Parsing screenplays"
            self._run("scripts/parse_all.py", *sh)
            # ---- 2. Pass A, one episode at a time (progress per episode)
            self.step = "Pass A: reading episodes"

            def pa(line):
                m = re.match(r"\s+ep(\d+): ", line)
                if m:
                    self.done_eps += 1
                    self.step = f"Analyzing {self.done_eps}/{self.total_eps}"
                elif re.match(r"\s+ep(\d+) FAILED", line):
                    self.done_eps += 1
                    self.failed_eps.append(int(re.match(r"\s+ep(\d+)", line).group(1)))
                    self._say(line.strip()[:240])
                elif "show map FAILED" in line or "error, waiting" in line:
                    self._say(line.strip()[:200])
            self._run("scripts/run_pass_a.py", "--all", *sh, on_line=pa, ok_codes=(0, 1))
            if len(self.failed_eps) == self.total_eps:
                raise RuntimeError("Pass A failed for every episode (the LLM did not answer). " +
                                   next((l for l in reversed(self.log) if "FAILED" in l), ""))
            if self.failed_eps:
                self._say(f"continuing without episodes {self.failed_eps} (Pass A failed)")
            # ---- 3. candidates
            self.step = "Generating candidates"
            self._run("scripts/run_candidates.py", *sh)
            # ---- 4. Cold Viewer (optional, limited for big batches)
            if self.cold_viewer_top != 0:
                self.step = "Cold Viewer"
                extra = ["--top-per-episode", str(self.cold_viewer_top)] if self.cold_viewer_top else []

                def cvl(line):
                    m = re.match(r"\s+(\d+)/(\d+) ", line)
                    if m:
                        self.step = f"Cold Viewer {m.group(1)}/{m.group(2)}"
                try:
                    self._run("scripts/run_cold_viewer.py", *sh, *extra, on_line=cvl, ok_codes=(0, 1))
                except RuntimeError as e:      # Cold Viewer is a penalty only: ranking still works without it
                    self._say(f"Cold Viewer skipped: {str(e)[:200]}")
            # ---- 5. final global ranking + checks
            self.step = "Ranking"
            self._run("scripts/run_final.py", *sh)
            self.step = "Checking"
            self._run("scripts/evaluate.py", *sh, ok_codes=(0, 1))
            self.state, self.step = "done", "Done"
            self._say("done")
        except Exception as e:          # noqa: BLE001 - every failure is reported to the UI, never raised into Streamlit
            self.state, self.error = "error", str(e)[:1500]
            self._say(f"ERROR: {self.error[:300]}")
        finally:
            self.finished = time.time()

    # ------------------------------------------------------------
    def status(self) -> dict:
        return {"state": self.state, "step": self.step, "done_eps": self.done_eps, "total_eps": self.total_eps,
                "log": list(self.log[-30:]), "error": self.error, "elapsed_s": round((self.finished or time.time()) - self.started)}

    def results(self) -> dict | None:
        if self.state != "done":
            return None
        out = self.workspace / "shows" / self.show_id / "outputs"
        j = lambda n: json.loads((out / n).read_text(encoding="utf-8")) if (out / n).exists() else None
        rep = (out / "eval_report.md").read_text(encoding="utf-8") if (out / "eval_report.md").exists() else ""
        return {"promos": j("promos_ranked.json") or [], "manifest": j("run_manifest.json") or {}, "report": rep,
                "ingest": self.ingest_report, "upload": self.upload_report, "failed_eps": self.failed_eps}

    def cleanup(self):
        shutil.rmtree(self.workspace, ignore_errors=True)


def preflight(provider: str) -> str | None:
    """None if the live pipeline can call its LLM, else a user-facing reason."""
    if provider == "offline":
        return None
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY", "").strip():
        return ("Live analysis needs a Gemini API key on the server (environment variable GEMINI_API_KEY). "
                "It is not configured, so only the pre-computed demo is available.")
    if provider == "openai_compat" and not os.getenv("LLM_BASE_URL"):
        return "LLM_PROVIDER=openai_compat needs LLM_BASE_URL (and usually LLM_API_KEY) on the server."
    return None


def cleanup_old():
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with _LOCK:
        for jid, j in list(_JOBS.items()):
            if j.finished and now - j.finished > JOB_TTL_S:
                j.cleanup(); _JOBS.pop(jid, None)
    for d in LIVE_ROOT.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > JOB_TTL_S and d.name not in _JOBS:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def start_job(files: list[tuple[str, bytes]], *, mode: str = "new", show_name: str | None = None,
              existing_show: dict | None = None, cold_viewer_top: int | None = 2, provider: str | None = None,
              background: bool = True) -> LiveJob:
    """Validate uploads, build an isolated workspace, and start the pipeline. Raises UploadError for bad input."""
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    why = preflight(provider)
    if why:
        raise UploadError(why)
    if mode == "continuation" and not existing_show:
        raise UploadError("choose which existing show these episodes continue")
    col = collect_inputs(files)
    cleanup_old()
    job_id = uuid.uuid4().hex[:12]
    ws = _workspace(job_id, existing_show if mode == "continuation" else None)
    folder = re.sub(r"[^\w .'\-]", "_", col["folder_hint"] or "upload").strip() or "upload"
    up = ws / "uploads" / folder
    up.mkdir(parents=True)
    for name, txt in col["texts"].items():
        (up / re.sub(r"[^\w .'\-]", "_", name)).write_text(txt, encoding="utf-8")
    job = LiveJob(job_id, ws, len(col["texts"]), mode, (show_name or "").strip() or None,
                  existing_show if mode == "continuation" else None, cold_viewer_top, provider,
                  upload_report={"files": sorted(col["texts"]), "skipped": col["skipped"], "rejected": col["rejected"],
                                 "folder": folder})
    with _LOCK:
        _JOBS[job_id] = job
    if background:
        threading.Thread(target=job.run, name=f"live-{job_id}", daemon=True).start()
    else:
        job.run()
    return job


def get_job(job_id: str) -> LiveJob | None:
    return _JOBS.get(job_id)
