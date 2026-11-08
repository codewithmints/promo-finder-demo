"""Final scoring, global ranking helpers, tiers and raw-output assembly.

final_score = pre_score (generator features x v0 weights; composites include a 0.05 join penalty)
              - 0.30 if Cold Viewer verdict == "reject"
              - 0.10 * (context_dependency - 2) if context_dependency >= 3
Cold Viewer never adds points (on dev gold its scores did not separate good from bad promos).
Multi-segment promos inherit the Cold Viewer result of their main clip (they were not judged separately).
Weights are v0 (results/ranking_decision.json: no re-weighting passed the pre-registered rule).

All text in the outputs is the ORIGINAL screenplay text with absolute episode timestamps.
"""
from __future__ import annotations

import re

from .parser import Episode, units_in_span, fmt_time, EPS

RANKING_VERSION = "v1_global"      # v0 weights; global selection (no per-episode quota) + same-episode multi-segment
WEIGHTS_VERSION = "v0"
CV_REJECT_PENALTY = 0.30
CV_CONTEXT_PENALTY = 0.10          # per point above 2
HOOK_TYPE_WORDS = {"violence": "a violent moment", "shocking_line": "a shocking line", "reveal": "a reveal",
                   "accusation": "an accusation", "threat": "a threat", "humiliation": "a public humiliation",
                   "betrayal": "a betrayal", "romance": "a charged romantic moment", "rescue": "a rescue",
                   "ultimatum": "an ultimatum", "breakdown": "an emotional breakdown", "mystery": "a mystery",
                   "none": "a tense moment"}


def segments_of(c: dict) -> list[tuple[float, float]]:
    return [tuple(s) for s in c.get("segments") or [[c["start"], c["end"]]]]


def final_score(cand: dict, cv: dict | None) -> tuple[float, dict]:
    pen = {"cv_reject": 0.0, "cv_context": 0.0}
    if cv:
        if cv["verdict"] == "reject":
            pen["cv_reject"] = -CV_REJECT_PENALTY
        if cv["context_dependency"] >= 3:
            pen["cv_context"] = -CV_CONTEXT_PENALTY * (cv["context_dependency"] - 2)
    return round(cand["pre_score"] + sum(pen.values()), 4), pen


def _short(t: str, n: int = 110) -> str:
    t = " ".join(t.split())
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"


def _units(e: Episode, cand: dict):
    out = []
    for s, t in segments_of(cand):
        out += units_in_span(e, s, t)
    return out


def hook_and_cliff(e: Episode, cand: dict) -> tuple[dict, dict]:
    """Hook = first spoken/acted line of the hook block (original text); cliffhanger = last line before the final cut."""
    segs = segments_of(cand)
    s0, (sl, tl) = segs[0][0], segs[-1]
    shots = {s.shot_id: s for s in e.shots}
    first = [u for u in units_in_span(e, *segs[0]) if u.track in ("dialogue", "action")]
    hb = cand["hook_block"]
    cand_lines = [u for u in first if u.shot_id == hb and u.t_start >= s0 - EPS] or [u for u in first if u.t_start >= s0 - EPS]
    h = cand_lines[0] if cand_lines else None
    ht = h.t_start if h else shots[hb].t_start
    hook = {"time": fmt_time(ht), "offset_s": round(max(0.0, ht - s0), 2),
            "speaker": h.speaker_raw if h and h.track == "dialogue" else "", "text": h.text if h else shots[hb].text}
    lastu = units_in_span(e, sl, tl)
    dl = [u for u in lastu if u.track == "dialogue" and u.t_end <= tl + EPS]
    tail = [u for u in lastu if u.track in ("dialogue", "action") and u.t_end <= tl + EPS]
    c = dl[-1] if dl and dl[-1].t_end > tl - 6 else (tail[-1] if tail else None)
    cliff = {"time": fmt_time(c.t_start) if c else fmt_time(tl), "speaker": c.speaker_raw if c and c.track == "dialogue" else "",
             "text": c.text if c else shots[cand["cliff_block"]].text}
    return hook, cliff


def why_line(cand: dict, tags: dict, hook: dict, cv: dict | None) -> str:
    T = {b["block_id"]: b for b in tags["blocks"]}
    ht = HOOK_TYPE_WORDS.get(T[cand["hook_block"]]["hook_type"], "a tense moment")
    reasons = " ".join(cand["reasons"])
    if "unanswered question" in reasons:
        end = "cuts on an unanswered question"
    elif "plea" in reasons:
        end = "cuts on a desperate plea"
    elif "episode cliffhanger" in reasons:
        end = "cuts on the episode's own cliffhanger"
    else:
        end = "cuts before the tension resolves"
    lead = (f"Opens on {ht} from earlier in the episode, then jumps to the build-up and "
            if cand.get("type") == "multi_segment" else f"Opens on {ht} {hook['offset_s']:.1f}s in and ")
    q = (cv or {}).get("open_question", "")
    q = f"; leaves the viewer asking: {_short(q, 90)}" if q and q.lower() != "none" and cand.get("type") != "multi_segment" else ""
    return f"{lead}{end}{q}"


def screenplay_excerpt(e: Episode, cand: dict) -> list[dict]:
    """Original screenplay units per segment, original labels, absolute timestamps, source order."""
    out = []
    for i, (s, t) in enumerate(segments_of(cand), 1):
        for u in units_in_span(e, s, t):
            label = {"shot": "CAMERA", "action": "ACTION", "sfx": "SOUND FX"}.get(u.track)
            if not label:
                cues = ", ".join(u.delivery + u.tags)
                label = f"{u.speaker_raw} ({cues})" if cues else u.speaker_raw
            out.append({"segment": i, "start": fmt_time(u.t_start), "end": fmt_time(u.t_end), "label": label,
                        "text": u.text, "unit_id": u.unit_id, "src_line": u.src_line})
    return out


def excerpt_text(ex: list[dict]) -> str:
    lines, seg = [], None
    for x in ex:
        if x["segment"] != seg and (seg is not None or any(y["segment"] > 1 for y in ex)):
            lines.append(f"--- segment {x['segment']} ---")
        seg = x["segment"]
        lines.append(f"[{x['start']}-{x['end']}] {x['label']}: {x['text']}")
    return "\n".join(lines)


def flags(cand: dict, cv: dict | None) -> list[str]:
    f, F = [], cand["features"]
    if F["hook"] < 0.5:
        f.append("weak_hook")
    if cand["hook_delay"] > 3:
        f.append("hook_after_3s")
    if F["slow_open"] > 0:
        f.append("slow_open")
    if F["cliff"] < 0.66:
        f.append("weak_cliffhanger")
    if F["resolution"]:
        f.append("resolves_near_end")
    if F["context"] >= 0.5 or (cv and cv["context_dependency"] >= 3):
        f.append("needs_context")
    if F["spoiler"] >= 0.5:
        f.append("spoiler_risk")
    if cv and cv["verdict"] == "reject":
        f.append("cold_viewer_reject")
    if cand.get("type") == "multi_segment":
        f.append("multi_segment_join_unverified")
    return f


def same_moment(a: dict, b: dict) -> bool:
    """Within an episode: two promos built on the same hook block or the same cliffhanger block."""
    return a["ep"] == b["ep"] and (a["hook_block"] == b["hook_block"] or a["cliff_block"] == b["cliff_block"])


def _norm(t: str) -> str:
    return re.sub(r"[^a-z ]", "", t.lower()).strip()


def cross_episode_duplicate(a: dict, b: dict) -> bool:
    """Adjacent episodes that re-open on the same scene: identical hook line or identical cliffhanger line."""
    if abs(a["ep"] - b["ep"]) != 1:
        return False
    return (len(_norm(a["_hook_text"])) > 12 and _norm(a["_hook_text"]) == _norm(b["_hook_text"])) or \
           (len(_norm(a["_cliff_text"])) > 12 and _norm(a["_cliff_text"]) == _norm(b["_cliff_text"]))


# ---------------------------------------------------------------- posting tiers (classification only; rank unchanged)
TIERING_VERSION = "tiers_v1"
SCORE_FLOOR = 0.40          # hook and cliff both >= 2/3 alone contribute 0.40 at the default weights
TIER_A_BLOCKERS = ("needs_context", "spoiler_risk", "resolves_near_end", "cold_viewer_reject", "slow_open")
TIER_C_TRIGGERS = ("weak_hook", "weak_cliffhanger", "cold_viewer_reject")


def tier(features: dict, hook_offset: float, final: float, flag_list: list[str]) -> tuple[str, str]:
    """A (post first): hook >= 2/3 within 3.0s, cliffhanger >= 2/3, score >= floor, none of TIER_A_BLOCKERS.
    C (post last / needs human check): score < floor or any of TIER_C_TRIGGERS.  B: everything else."""
    c_hits = [f for f in flag_list if f in TIER_C_TRIGGERS]
    if final < SCORE_FLOOR or c_hits:
        why = ([f"score {final:.2f} < floor {SCORE_FLOOR}"] if final < SCORE_FLOOR else []) + c_hits
        return "C", "post last / needs review: " + ", ".join(why)
    miss = []
    if hook_offset > 3.0:
        miss.append(f"hook at +{hook_offset:.1f}s")
    if features["hook"] < 0.66:
        miss.append("hook below 2/3")
    if features["cliff"] < 0.66:
        miss.append("cliffhanger below 2/3")
    miss += [f for f in flag_list if f in TIER_A_BLOCKERS]
    if miss:
        return "B", "not tier A: " + ", ".join(miss)
    return "A", f"strong hook within {hook_offset:.1f}s, strong cliffhanger, no context/spoiler/resolution flags, score {final:.2f}"
