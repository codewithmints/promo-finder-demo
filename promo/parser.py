"""Screenplay parser: TXT -> timed units, shots, legal cut points.

File format (verified on all 50 episodes):
    header:  EPISODE NUMBER / EPISODE LOGLINE / CONTINUITY NOTES
    script:  --- SCRIPT START --- ... --- SCRIPT END ---
    scene:   SCENE 01 / Scene heading: / Scene start: MM:SS.ss / Scene end: / Actions and dialogue:
    unit:    **LABEL (cue, cue):**  /  Start: (MM:SS.ss) | End: (MM:SS.ss)  /  text line
Tracks: CAMERA shots tile the timeline; dialogue / ACTION / SOUND FX overlay them.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

TRACKS = {"CAMERA": "shot", "ACTION": "action", "SOUND FX": "sfx"}
SPEAKER_TAGS = {"O.S.", "V.O.", "O.C.", "CONT'D", "CONT’D"}
EPS = 0.05  # tolerance (s) when comparing boundaries

RE_FILE = re.compile(r"episode-(\d+)", re.I)
RE_LABEL = re.compile(r"^\*\*(.+?):\*\*\s*$")
RE_TIME = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")
RE_SPAN = re.compile(r"^Start:\s*\(([^)]*)\)\s*\|\s*End:\s*\(([^)]*)\)\s*$", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
RE_FIELD = re.compile(r"^(scene heading|scene start|scene end|episode number|episode logline|continuity notes)\s*:\s*(.*)$", re.I)
RE_PAREN = re.compile(r"\(([^)]*)\)")


def parse_time(s: str) -> float:
    m = RE_TIME.search(s)
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    h, mnt, sec = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + float(sec)


def fmt_time(t: float) -> str:
    m, s = divmod(round(t, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Unit:
    ep: int
    unit_id: str
    track: str              # shot | dialogue | action | sfx
    t_start: float
    t_end: float
    text: str
    speaker_raw: str = ""
    speaker: str = ""       # canonical name, filled later (Pass A alias map)
    delivery: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    scene_id: int = 0
    scene_heading: str = ""
    shot_id: str = ""       # shot containing t_start
    src_line: int = 0       # 1-based line number of the label in the source file


@dataclass
class Shot:
    ep: int
    shot_id: str
    t_start: float
    t_end: float
    scene_id: int
    text: str
    unit_ids: list = field(default_factory=list)


@dataclass
class Episode:
    ep: int
    path: str
    logline: str
    continuity: str
    duration: float
    scenes: list
    units: list
    shots: list
    legal_starts: list
    legal_ends: list
    warnings: list

    def lines(self):
        return [u for u in self.units if u.track != "shot"]


def _split_label(label: str):
    groups = RE_PAREN.findall(label)
    name = RE_PAREN.sub("", label).strip()
    delivery, tags = [], []
    for g in groups:
        g = g.strip()
        if g.upper() in SPEAKER_TAGS:
            tags.append(g.upper())
        else:
            delivery += [c.strip() for c in g.split(",") if c.strip()]
    return name, delivery, tags


def parse_episode(path: str | Path, ep: int | None = None) -> Episode:
    path = Path(path)
    raw = path.read_text(encoding="utf-8").splitlines()
    if ep is None:                      # legacy fallback; the show registry passes the identified number
        m = RE_FILE.search(path.name)
        ep = int(m.group(1)) if m else 0
    meta, scenes, units, warns = {}, [], [], []
    scene = {"scene_id": 0, "heading": "", "t_start": None, "t_end": None}
    in_script = not any(l.strip().startswith("--- SCRIPT START") for l in raw)   # files without markers: whole file
    i = 0
    while i < len(raw):
        s = raw[i].strip()
        if s.startswith("--- SCRIPT START"):
            in_script = True
        elif s.startswith("--- SCRIPT END"):
            in_script = False
        elif (fm := RE_FIELD.match(s)):
            key, val = fm.group(1).lower(), fm.group(2).strip()
            if key == "scene heading":
                scene["heading"] = val
            elif key == "scene start":
                scene["t_start"] = parse_time(val)
            elif key == "scene end":
                scene["t_end"] = parse_time(val)
            else:
                meta[key] = val
        elif in_script and (sm := RE_SCENE.match(s)):
            scene = {"scene_id": int(sm.group(1)), "heading": "", "t_start": None, "t_end": None}
            scenes.append(scene)
        elif in_script and (lm := RE_LABEL.match(s)):
            label = lm.group(1).strip()
            span = RE_SPAN.match(raw[i + 1].strip()) if i + 1 < len(raw) else None
            if not span:
                warns.append(f"line {i+1}: label without timestamp: {s}")
                i += 1
                continue
            t0, t1 = parse_time(span.group(1)), parse_time(span.group(2))
            # text = following non-empty lines until blank/next label
            j, text = i + 2, []
            while j < len(raw) and raw[j].strip() and not RE_LABEL.match(raw[j].strip()) \
                    and not raw[j].strip().startswith("---"):
                text.append(raw[j].strip())
                j += 1
            name, delivery, tags = _split_label(label)
            track = TRACKS.get(name.upper(), "dialogue")
            if t1 <= t0:
                warns.append(f"line {i+1}: end<=start ({t0}->{t1})")
                t1 = t0 + 0.1
            units.append(Unit(ep, "", track, t0, t1, " ".join(text),
                              speaker_raw=name if track == "dialogue" else "",
                              delivery=delivery, tags=tags,
                              scene_id=scene["scene_id"], scene_heading=scene["heading"],
                              src_line=i + 1))
            i = j
            continue
        i += 1

    # stable sort by start (fixes the small <1s inversions in 6 episodes); keep src order on ties
    units.sort(key=lambda u: (u.t_start, u.track != "shot", u.src_line))
    for k, u in enumerate(units):
        u.unit_id = f"e{ep:02d}_u{k:03d}"

    # Shot track: CAMERA units, plus synthetic 'uncovered' blocks where the camera track has a gap
    # (86 such gaps across 40 episodes; up to 25s in ep18) so the blocks tile [0, duration].
    duration = max(u.t_end for u in units)
    blocks, cursor = [], 0.0
    for u in (u for u in units if u.track == "shot"):
        if u.t_start - cursor > EPS:
            blocks.append((cursor, u.t_start, None))
        blocks.append((u.t_start, u.t_end, u))
        cursor = max(cursor, u.t_end)
    if duration - cursor > EPS:
        blocks.append((cursor, duration, None))
    shots = []
    for t0, t1, cam in blocks:
        sid = f"e{ep:02d}_s{len(shots):03d}"
        if cam is not None:
            cam.shot_id = sid
            shots.append(Shot(ep, sid, t0, t1, cam.scene_id, cam.text, [cam.unit_id]))
        else:
            sc = next((x["scene_id"] for x in scenes if x["t_start"] is not None and x["t_start"] - EPS <= t0 < (x["t_end"] or 1e9)), 0)
            shots.append(Shot(ep, sid, t0, t1, sc, "[no camera note — dialogue/action only]", []))
    for u in units:
        if u.track == "shot":
            continue
        owner = next((s for s in shots if s.t_start - EPS <= u.t_start < s.t_end - EPS), None)
        if owner is None:  # fallback: nearest shot start
            owner = min(shots, key=lambda s: abs(s.t_start - u.t_start))
        u.shot_id = owner.shot_id
        owner.unit_ids.append(u.unit_id)

    # validation (camera gaps are expected and handled; only report silence across ALL tracks)
    prev_end = 0.0
    for u in units:
        if u.t_start - prev_end > 1.0:
            warns.append(f"silent gap (no unit on any track) {fmt_time(prev_end)}-{fmt_time(u.t_start)}")
        prev_end = max(prev_end, u.t_end)
    for sc in scenes:
        if sc["t_start"] is None or sc["t_end"] is None:
            warns.append(f"scene {sc['scene_id']} missing start/end")

    legal_starts, legal_ends = legal_cut_points(units, shots, duration)
    for w in warns:
        warnings.warn(f"ep{ep}: {w}")
    return Episode(ep, str(path), meta.get("episode logline", ""), meta.get("continuity notes", ""),
                   duration, scenes, units, shots, legal_starts, legal_ends, warns)


def _split_sentence_joins(units):
    """Times between two dialogue halves of one sentence ('...' / '...')."""
    dl = [u for u in units if u.track == "dialogue"]
    bad = []
    for a, b in zip(dl, dl[1:]):
        if a.text.rstrip('"” ').endswith(("...", "…")) and b.text.lstrip('"“ ').startswith(("...", "…")):
            bad.append((a.t_end - EPS, b.t_start + EPS))
    return bad


def legal_cut_points(units, shots, duration):
    """Times where a promo may start / end without cutting a line or a split sentence."""
    dialogue = [u for u in units if u.track == "dialogue"]
    joins = _split_sentence_joins(units)

    def inside_line(t):
        return any(d.t_start + EPS < t < d.t_end - EPS for d in dialogue) or \
               any(a <= t <= b for a, b in joins)

    # never end on a trailing 'cut to black' shot
    last_real_end = duration
    if shots and re.search(r"cut to black", shots[-1].text, re.I) and shots[-1].t_end - shots[-1].t_start <= 1.5:
        last_real_end = shots[-1].t_start
    starts = {round(s.t_start, 2) for s in shots}
    starts |= {round(u.t_start, 2) for u in units if u.track in ("dialogue", "action")}
    ends = {round(s.t_end, 2) for s in shots}
    ends |= {round(u.t_end, 2) for u in units if u.track in ("dialogue", "action")}
    starts = sorted(t for t in starts if not inside_line(t) and t < last_real_end)
    ends = sorted(t for t in ends if not inside_line(t) and t <= last_real_end + EPS)
    return starts, ends


def units_in_span(episode: Episode, t0: float, t1: float, include_shots=True):
    """Reconstruct a promo span: every unit whose midpoint is in [t0, t1), in source order."""
    out = [u for u in episode.units
           if t0 - EPS <= (u.t_start + u.t_end) / 2 < t1 + EPS and (include_shots or u.track != "shot")]
    return sorted(out, key=lambda u: u.src_line)


def load_show(raw_dir: str | Path) -> list[Episode]:
    files = sorted(Path(raw_dir).glob("*.txt"), key=lambda p: int(RE_FILE.search(p.name).group(1)))
    return [parse_episode(p) for p in files]


def episode_to_records(e: Episode):
    return [asdict(u) for u in e.units], [asdict(s) for s in e.shots]
