"""Selective same-episode multi-segment promos: an earlier HOOK segment + an existing MAIN candidate.

A composite is created only when ALL hold (fixed rules, not tuned on gold):
  * main candidate has a strong cliffhanger         (cliff feature >= 2/3)
  * main candidate has a weak opening               (hook feature <= 1/3)
  * an earlier block in the same episode is a substantially better hook
        (hook >= 2 on the 0-3 scale AND at least one full level above the main's hook)
  * that hook block is not a reveal (reveal < 2) and does not resolve tension
  * hook segment: 3-10s, on legal cuts, hook line within 1s of the segment start,
    ends >= 1s before the main starts (no overlap, chronological)
  * total duration 20-60s
  * the two segments share at least one character
  * the composite scores higher than the main alone, after a 0.05 join penalty
At most one composite per main candidate.
"""
from __future__ import annotations

import re

from .parser import Episode, EPS
from .candidates import features_for_segments, block_heuristics, MIN_DUR, MAX_DUR

JOIN_PENALTY = 0.05
SEG_MIN, SEG_MAX, GAP_MIN = 3.0, 10.0, 1.0


def _characters(e: Episode, alias: dict, s: float, t: float) -> set:
    names = set(alias.values())
    out = set()
    for u in e.units:
        if s - EPS <= u.t_start < t:
            if u.track == "dialogue":
                out.add(alias.get(u.speaker_raw.upper(), u.speaker_raw.title()))
            for n in names:
                first = n.split()[-1]
                if len(first) >= 4 and re.search(rf"\b{re.escape(first)}\b", u.text):
                    out.add(n)
    return out


def generate_composites(e: Episode, tags: dict, cands: list[dict], spoiler_ep: bool) -> tuple[list[dict], dict]:
    T = {b["block_id"]: b for b in tags["blocks"]}
    H = block_heuristics(e)
    hook_of = lambda b: max(T[b.shot_id]["hook"], H[b.shot_id]["hook_h"])
    blocks = sorted(e.shots, key=lambda s: s.t_start)
    content = sorted([u for u in e.units if u.track in ("dialogue", "action")], key=lambda u: u.t_start)
    alias = tags.get("alias", {})
    LS, LE = e.legal_starts, e.legal_ends
    stats = {"mains_considered": 0, "eligible_mains": 0, "built": 0}
    out = []
    for m in cands:
        stats["mains_considered"] += 1
        f = m["features"]
        if f["cliff"] < 0.66 or f["hook"] > 0.34:
            continue
        stats["eligible_mains"] += 1
        main_hook_level = f["hook"] * 3
        best = None
        hooks = [b for b in blocks if b.t_end <= m["start"] - GAP_MIN and hook_of(b) >= 2
                 and hook_of(b) >= main_hook_level + 1 and T[b.shot_id]["reveal"] < 2 and not T[b.shot_id]["resolves"]]
        hooks.sort(key=lambda b: (-hook_of(b), -T[b.shot_id]["intensity"], -b.t_start))
        main_chars = _characters(e, alias, m["start"], m["end"])
        for b in hooks:
            starts = [x for x in LS if b.t_start - 1.0 - EPS <= x <= b.t_start + EPS]
            if not starts:
                continue
            s = max(starts)
            line = next((u for u in content if u.t_start >= s - EPS and u.t_start < b.t_end), None)
            need = line.t_end if line else b.t_end
            ends = [x for x in LE if x >= max(s + SEG_MIN, need) - EPS and x <= min(s + SEG_MAX, m["start"] - GAP_MIN) + EPS]
            if not ends:
                continue
            t = min(ends)
            total = (t - s) + m["duration"]
            if not (MIN_DUR - EPS <= total <= MAX_DUR + EPS):
                continue
            shared = _characters(e, alias, s, t) & main_chars
            if not shared:
                continue
            r = features_for_segments(e, tags, [(s, t), (m["start"], m["end"])], spoiler_ep)
            if r["pre_score"] - JOIN_PENALTY <= m["pre_score"]:
                continue
            best = (s, t, r, shared)
            break
        if not best:
            continue
        s, t, r, shared = best
        stats["built"] += 1
        out.append({
            "cand_id": f"m{e.ep:02d}_{len(out)+1:02d}", "ep": e.ep, "type": "multi_segment", "main_id": m["cand_id"],
            "segments": [[round(s, 2), round(t, 2)], [m["start"], m["end"]]],
            "start": s, "end": m["end"], "duration": round((t - s) + m["duration"], 2),
            "start_block": r["hook_block"], "end_block": m["end_block"], "hook_block": r["hook_block"],
            "hook_delay": r["hook_delay"], "cliff_block": r["cliff_block"],
            "features": r["features"], "contrib": {**r["contrib"], "join": -JOIN_PENALTY},
            "pre_score": round(r["pre_score"] - JOIN_PENALTY, 4),
            "reasons": [f"multi-segment: earlier hook ({T[r['hook_block']]['hook_type']}) joined to {m['cand_id']}, "
                        f"which had a weak opening (hook {f['hook']:.2f} -> {r['features']['hook']:.2f})",
                        f"shared character(s): {', '.join(sorted(shared))}"] +
                       [x for x in m["reasons"] if not x.startswith("strong opening")],
        })
    return out, stats
