"""Show / episode registry: multiple shows, later episodes of the same show, no show-specific code.

Registry file: data/shows.json
  {"default_show": "<show_id>",
   "shows": {"<show_id>": {"show_id", "show_name", "work_dir", "outputs_dir", "held_out_episodes": [...],
                           "episodes": [{"episode_number", "episode_id", "source_file", "sha256", "flags"}]}}}

Identification (never by episode number alone):
  show   = --show-name if given, else a "SHOW:/SERIES:/TITLE:" header in the screenplays, else the folder name.
           A generic folder name (raw, episodes, scripts, ...) with no header is AMBIGUOUS -> refused, ask for --show-name.
  number = "EPISODE NUMBER:" header, else a filename pattern (episode-12, ep_12, e12, or the only number in the name).
           Header vs filename disagreement, missing numbers and duplicate numbers are flagged, not guessed.
  episode_id = "<show_id>:e<number>"
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .parser import parse_episode, Episode

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "shows.json"
GENERIC_FOLDERS = {"raw", "data", "episodes", "episode", "scripts", "script", "input", "inputs", "txt", "new", "files",
                   "screenplays", "screenplay", "uploads", "download", "downloads", "tmp", "temp"}
RE_HDR_NUM = re.compile(r"^\s*EPISODE\s+NUMBER\s*:\s*(?:Episode\s*)?(\d+)", re.I | re.M)
RE_HDR_SHOW = re.compile(r"^\s*(?:SHOW|SERIES|SHOW\s+TITLE|SERIES\s+TITLE)\s*:\s*(.+)$", re.I | re.M)
RE_FILE_NUM = [re.compile(p, re.I) for p in (r"episode[\s_-]*(\d+)", r"\bep[\s_.-]*(\d+)", r"(?:^|[^a-z])e(\d+)(?:[^0-9]|$)")]


class ShowError(ValueError):
    pass


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:48].strip("_") or "show"


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"default_show": None, "shows": {}}


def save_registry(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- identification
def identify_episode_number(path: Path) -> tuple[int | None, list[str]]:
    flags = []
    head = path.read_text(encoding="utf-8", errors="ignore")[:3000]
    m = RE_HDR_NUM.search(head)
    hdr = int(m.group(1)) if m else None
    fn = None
    for rx in RE_FILE_NUM:
        mm = rx.search(path.stem)
        if mm:
            fn = int(mm.group(1)); break
    if fn is None:
        nums = re.findall(r"\d+", path.stem)
        if len(nums) == 1:
            fn = int(nums[0])
        elif len(nums) > 1:
            flags.append(f"filename has several numbers {nums}; not used")
    if hdr is not None and fn is not None and hdr != fn:
        flags.append(f"number_conflict: header says {hdr}, filename says {fn}")
        return None, flags
    n = hdr if hdr is not None else fn
    if n is None:
        flags.append("no_episode_number: neither an 'EPISODE NUMBER:' header nor a number in the filename")
    return n, flags


def identify_show_name(folder: Path, files: list[Path], show_name: str | None) -> tuple[str, list[str]]:
    flags = []
    hdrs = set()
    for f in files:
        m = RE_HDR_SHOW.search(f.read_text(encoding="utf-8", errors="ignore")[:3000])
        if m:
            hdrs.add(m.group(1).strip())
    if len(hdrs) > 1:
        raise ShowError(f"files in {folder} name different shows: {sorted(hdrs)}; split them or pass --show-name")
    if show_name:
        if hdrs and slug(next(iter(hdrs))) != slug(show_name):
            flags.append(f"--show-name '{show_name}' differs from screenplay header '{next(iter(hdrs))}' (using --show-name)")
        return show_name.strip(), flags
    if hdrs:
        return next(iter(hdrs)), flags
    name = folder.name.strip()
    if name.lower() in GENERIC_FOLDERS or re.fullmatch(r"[\d_\-\s]+", name) or len(name) < 3:
        raise ShowError(f"cannot identify the show: folder '{name}' is generic and the screenplays have no SHOW:/SERIES: "
                        f"header. Pass --show-name.")
    return name, flags


# ---------------------------------------------------------------- show object
@dataclass
class Show:
    show_id: str
    show_name: str
    work_dir: Path
    outputs_dir: Path
    held_out_episodes: list = field(default_factory=list)
    episodes: list = field(default_factory=list)       # registry dicts, sorted by number

    @property
    def parsed_dir(self): return self.work_dir / "parsed"
    @property
    def pass_a_dir(self): return self.work_dir / "pass_a"
    @property
    def candidates_dir(self): return self.work_dir / "candidates"
    @property
    def cold_viewer_dir(self): return self.work_dir / "cold_viewer"
    @property
    def gold_dir(self): return self.work_dir / "gold"
    @property
    def blind_dir(self): return self.work_dir / "blind"

    def episode_id(self, number: int) -> str:
        return f"{self.show_id}:e{number:03d}"

    def load_episodes(self, numbers=None) -> list[Episode]:
        out = []
        for e in self.episodes:
            if e.get("excluded") or (numbers is not None and e["episode_number"] not in numbers):
                continue
            out.append(parse_episode(_abs(e["source_file"]), ep=e["episode_number"]))
        return out

    def is_held_out(self, number: int) -> bool:
        return number in set(self.held_out_episodes)


def get_show(show_id: str | None = None) -> Show:
    reg = load_registry()
    sid = show_id or reg.get("default_show")
    if not sid:
        raise ShowError("no show registered; run scripts/ingest.py first")
    if sid not in reg["shows"]:
        raise ShowError(f"unknown show '{sid}'. Registered: {sorted(reg['shows'])}")
    s = reg["shows"][sid]
    return Show(s["show_id"], s["show_name"], _abs(s["work_dir"]), _abs(s["outputs_dir"]), s.get("held_out_episodes", []),
                sorted(s["episodes"], key=lambda x: x["episode_number"]))


def add_show_arg(ap):
    ap.add_argument("--show", default=None, help="show_id from data/shows.json (default: the registry's default show)")
    return ap


# ---------------------------------------------------------------- ingest
def ingest(folder: str | Path, show_name: str | None = None, show_id: str | None = None, copy_raw: bool = True,
           work_dir: str | None = None, outputs_dir: str | None = None, held_out=None, make_default=False) -> dict:
    """Register a folder of episode screenplays as a new show, or as later episodes of an existing show."""
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*.txt") if p.is_file())
    if not files:
        raise ShowError(f"no .txt screenplays in {folder}")
    reg = load_registry()
    if show_id and show_id in reg["shows"]:
        name, flags = reg["shows"][show_id]["show_name"], []
        if show_name and slug(show_name) != slug(name):
            raise ShowError(f"show '{show_id}' is registered as '{name}', not '{show_name}'")
    else:
        name, flags = identify_show_name(folder, files, show_name)
        show_id = show_id or slug(name)
    new_show = show_id not in reg["shows"]
    if new_show:
        reg["shows"][show_id] = {"show_id": show_id, "show_name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "work_dir": work_dir or f"shows/{show_id}/work",
                                 "outputs_dir": outputs_dir or f"shows/{show_id}/outputs",
                                 "held_out_episodes": list(held_out or []), "episodes": []}
    s = reg["shows"][show_id]
    known = {e["episode_number"]: e for e in s["episodes"]}
    report = {"show_id": show_id, "show_name": name, "new_show": new_show, "added": [], "unchanged": [], "flagged": [],
              "show_flags": flags}
    seen_here = {}
    for f in files:
        n, fl = identify_episode_number(f)
        if n is None:
            report["flagged"].append({"file": f.name, "flags": fl}); continue
        if n in seen_here:
            report["flagged"].append({"file": f.name, "flags": [f"duplicate episode {n} in this folder (also {seen_here[n]})"]}); continue
        seen_here[n] = f.name
        h = _sha(f)
        if n in known:
            if known[n]["sha256"] == h:
                report["unchanged"].append(n)
            else:
                report["flagged"].append({"file": f.name, "flags": [f"episode {n} already registered from "
                                          f"{known[n]['source_file']} with different content; not overwritten"]})
            continue
        try:
            e = parse_episode(f, ep=n)
        except Exception as ex:          # noqa: BLE001
            report["flagged"].append({"file": f.name, "flags": [f"unparseable: {ex}"]}); continue
        if not e.units or not e.shots:
            report["flagged"].append({"file": f.name, "flags": ["no timed units found"]}); continue
        dest = f
        if copy_raw:
            raw_dir = _abs(s["work_dir"]).parent / "raw" if s["work_dir"] != "data" else ROOT / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            dest = raw_dir / f.name
            if dest.resolve() != f.resolve():
                if dest.exists():
                    report["flagged"].append({"file": f.name, "flags": [f"{dest} exists; not overwritten"]}); continue
                shutil.copy2(f, dest)
        s["episodes"].append({"episode_number": n, "episode_id": f"{show_id}:e{n:03d}", "source_file": _rel(dest),
                              "original_file": f.name, "sha256": h, "flags": fl,
                              "duration_s": round(e.duration, 2), "ingested": time.strftime("%Y-%m-%d %H:%M:%S")})
        report["added"].append(n)
    s["episodes"].sort(key=lambda x: x["episode_number"])
    if new_show and s["episodes"] and s["episodes"][0]["episode_number"] > 1:
        report["show_flags"].append(f"new show starts at episode {s['episodes'][0]['episode_number']}: if these are later episodes "
                                    f"of a registered show, re-run with --show <show_id> (registered: {sorted(k for k in reg['shows'] if k != show_id)})")
    nums = [e["episode_number"] for e in s["episodes"]]
    gaps = sorted(set(range(min(nums), max(nums) + 1)) - set(nums)) if nums else []
    if gaps:
        report["show_flags"].append(f"missing episode numbers: {gaps[:20]}{'…' if len(gaps) > 20 else ''}")
    if make_default or not reg.get("default_show"):
        reg["default_show"] = show_id
    save_registry(reg)
    return report
