"""Persistent human feedback: data/feedback/feedback.jsonl (append-only JSON lines, all shows in one file).

Each record is scoped by show_id and episode_id, stores the ORIGINAL segments and (optionally) the reviewer's
CORRECTED segments snapped to legal cuts, plus feature vectors for both, so learning never depends on re-running
the pipeline. Re-reviewing the same promo appends a new record; the latest one wins (supersedes).

Held-out guard: a record on an episode in the show's held_out_episodes is stored with evaluation_only=True and is
never returned for training (load_training refuses them; the learner checks again).
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator

from .candidates import features_for_segments, MIN_DUR, MAX_DUR
from .parser import Episode, EPS, fmt_time
from . import weights as Wt

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_FILE = ROOT / "data" / "feedback" / "feedback.jsonl"
REASONS = ["weak_hook", "no_cliffhanger", "resolved", "confusing", "needs_context", "spoiler", "bad_boundary",
           "low_emotion", "boring", "duplicate", "other"]
SNAP_TOLERANCE = 2.0          # seconds: corrected boundaries snap to the nearest legal cut within this distance


class Feedback(BaseModel):
    feedback_id: str
    created_at: str
    reviewer: str
    source: str                                   # review | blind_import | dev_gold_import
    show_id: str
    episode_id: str
    episode_number: int
    promo_id: str = ""
    candidate_id: str = ""
    ranking_version: str = ""
    weights_version: str = ""
    accepted: bool
    quality: Optional[int] = None                 # 1 best .. 3 weak
    reason: str = ""
    note: str = ""
    original_segments: list[list[float]]
    corrected_segments: Optional[list[list[float]]] = None
    boundary_shift_s: Optional[list[list[float]]] = None     # per segment [start shift, end shift]
    features: dict                                # Wt.vector of the original segments
    corrected_features: Optional[dict] = None
    evaluation_only: bool = False
    evaluation_only_reason: str = ""
    supersedes: Optional[str] = None

    @field_validator("quality")
    @classmethod
    def _q(cls, v):
        if v is not None and v not in (1, 2, 3):
            raise ValueError("quality must be 1, 2 or 3")
        return v

    @field_validator("reason")
    @classmethod
    def _r(cls, v):
        if v and v not in REASONS:
            raise ValueError(f"reason must be one of {REASONS}")
        return v


class ReviewError(ValueError):
    pass


def _snap(t: float, legal: list[float], what: str) -> float:
    best = min(legal, key=lambda x: abs(x - t))
    if abs(best - t) > SNAP_TOLERANCE:
        raise ReviewError(f"{what} {fmt_time(t)} is not within {SNAP_TOLERANCE}s of a legal cut (nearest {fmt_time(best)})")
    return round(best, 2)


def snap_segments(e: Episode, segments: list[list[float]]) -> list[list[float]]:
    """Snap corrected boundaries to legal cuts and validate order and total duration."""
    out = [[_snap(s, e.legal_starts, "start"), _snap(t, e.legal_ends, "end")] for s, t in segments]
    for i, (s, t) in enumerate(out):
        if t <= s:
            raise ReviewError(f"segment {i + 1}: end must be after start")
        if i and s < out[i - 1][1] - EPS:
            raise ReviewError("segments must be in chronological order without overlap")
    total = sum(t - s for s, t in out)
    if not (MIN_DUR - EPS <= total <= MAX_DUR + EPS):
        raise ReviewError(f"total duration {total:.1f}s is outside 20–60s")
    return out


def _append(rec: Feedback) -> Feedback:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    prev = latest().get(_key(rec.model_dump()))
    if prev:
        rec.supersedes = prev["feedback_id"]
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(rec.model_dump_json() + "\n")
    return rec


def _key(r: dict) -> tuple:
    return (r["show_id"], r["episode_id"], r["candidate_id"] or json.dumps(r["original_segments"]), r["reviewer"], r["source"])


def add_review(show, episode: Episode, tags: dict, spoiler_ep: bool, promo: dict, *, accepted: bool,
               quality: int | None = None, reason: str = "", note: str = "", corrected_segments=None,
               reviewer: str = "reviewer", source: str = "review", feedback_id: str | None = None) -> Feedback:
    """Record one human review of a ranked promo (promo = a record from promos_ranked.json, or any dict with
    segments/features/candidate_id). Rejections need a reason. Corrected boundaries are snapped to legal cuts and
    re-featurised so they can be learned from."""
    if not accepted and not reason:
        raise ReviewError("a rejection needs a reason")
    if accepted and quality is None:
        raise ReviewError("an acceptance needs a quality rating (1–3)")
    orig = [[s["start_s"], s["end_s"]] if isinstance(s, dict) else list(s) for s in promo["segments"]]
    multi = len(orig) > 1
    cv = promo.get("cold_viewer")
    x = promo.get("feature_vector") or Wt.vector(promo["features"], cv, multi)
    corr = corr_x = shift = None
    if corrected_segments:
        corr = snap_segments(episode, corrected_segments)
        if corr != [[round(s, 2), round(t, 2)] for s, t in orig]:
            r = features_for_segments(episode, tags, [tuple(c) for c in corr], spoiler_ep)
            corr_x = Wt.vector(r["features"], None, len(corr) > 1)
            corr_x["cv_reject"], corr_x["cv_context"] = x["cv_reject"], x["cv_context"]   # same clip, same judge signal
            shift = [[round(c[0] - o[0], 2), round(c[1] - o[1], 2)] for c, o in zip(corr, orig)] if len(corr) == len(orig) else None
        else:
            corr = None
    ev = show.is_held_out(episode.ep)
    rec = Feedback(feedback_id=feedback_id or f"{source}:{uuid.uuid4().hex[:12]}", created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   reviewer=reviewer, source=source, show_id=show.show_id, episode_id=show.episode_id(episode.ep),
                   episode_number=episode.ep, promo_id=promo.get("promo_id", ""), candidate_id=promo.get("candidate_id", ""),
                   ranking_version=promo.get("ranking_version", ""), weights_version=promo.get("weights_version", ""),
                   accepted=accepted, quality=quality, reason=reason, note=note, original_segments=orig,
                   corrected_segments=corr, boundary_shift_s=shift, features=x, corrected_features=corr_x,
                   evaluation_only=ev, evaluation_only_reason="held-out episode for this show" if ev else "")
    return _append(rec)


def load_all() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    return [json.loads(l) for l in FEEDBACK_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def latest(show_id: str | None = None) -> dict:
    """Latest record per (show, episode, promo/window, reviewer, source)."""
    out = {}
    for r in load_all():
        if show_id and r["show_id"] != show_id:
            continue
        out[_key(r)] = r
    return out


def load_training(show_id: str | None = None) -> list[dict]:
    """Records usable for learning: latest per key, NEVER evaluation-only (held-out) ones."""
    return [r for r in latest(show_id).values() if not r["evaluation_only"]]


def feedback_hash(records: list[dict]) -> str:
    return hashlib.sha256(json.dumps(sorted(r["feedback_id"] for r in records)).encode()).hexdigest()[:12]
