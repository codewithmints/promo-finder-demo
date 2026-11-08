"""Review store for the PUBLIC deployment (PUBLIC_MODE=1), which has no raw screenplays.

Same review fields and validation as promo/feedback.py, using only the exported bundle:
  * corrected boundaries snap to the exported legal cuts (cuts.json) and must total 20-60s
  * held-out episodes are marked evaluation-only
  * features of the ORIGINAL promo come from the bundle; corrected boundaries cannot be re-featurised without the
    screenplay, so corrected_features stays empty (flag needs_local_refeaturize)
Records go to data/feedback/feedback_public_demo.jsonl, a separate file the learner never reads, so a public
demo can never change the ranking. Locally, the real store (promo/feedback.py) is used instead.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .feedback import Feedback, REASONS, ReviewError, SNAP_TOLERANCE

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILE = ROOT / "data" / "feedback" / "feedback_public_demo.jsonl"
MIN_DUR, MAX_DUR, EPS = 20.0, 60.0, 0.05


def _snap(t, legal, what):
    best = min(legal, key=lambda x: abs(x - t))
    if abs(best - t) > SNAP_TOLERANCE:
        raise ReviewError(f"{what} {t:.2f}s is not within {SNAP_TOLERANCE}s of a legal cut (nearest {best:.2f}s)")
    return round(best, 2)


def snap_with_cuts(cuts_ep: dict, segments):
    out = [[_snap(s, cuts_ep["legal_starts"], "start"), _snap(t, cuts_ep["legal_ends"], "end")] for s, t in segments]
    for i, (s, t) in enumerate(out):
        if t <= s:
            raise ReviewError(f"segment {i + 1}: end must be after start")
        if i and s < out[i - 1][1] - EPS:
            raise ReviewError("segments must be in chronological order without overlap")
    total = sum(t - s for s, t in out)
    if not (MIN_DUR - EPS <= total <= MAX_DUR + EPS):
        raise ReviewError(f"total duration {total:.1f}s is outside 20–60s")
    return out


def add_public_review(show_id: str, promo: dict, cuts: dict, *, accepted: bool, quality=None, reason="", note="",
                      corrected_segments=None, reviewer="public_demo") -> dict:
    if not accepted and not reason:
        raise ReviewError("a rejection needs a reason")
    if accepted and quality is None:
        raise ReviewError("an acceptance needs a quality rating (1–3)")
    ep = promo["episode"]
    ce = cuts[str(ep)]
    orig = [[s["start_s"], s["end_s"]] for s in promo["segments"]]
    corr = shift = None
    if corrected_segments:
        corr = snap_with_cuts(ce, corrected_segments)
        if corr == [[round(s, 2), round(t, 2)] for s, t in orig]:
            corr = None
        elif len(corr) == len(orig):
            shift = [[round(c[0] - o[0], 2), round(c[1] - o[1], 2)] for c, o in zip(corr, orig)]
    rec = Feedback(feedback_id=f"public:{uuid.uuid4().hex[:12]}", created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   reviewer=reviewer, source="review_public_demo", show_id=show_id, episode_id=ce["episode_id"],
                   episode_number=ep, promo_id=promo["promo_id"], candidate_id=promo["candidate_id"],
                   ranking_version=promo.get("ranking_version", ""), weights_version=promo.get("weights_version", ""),
                   accepted=accepted, quality=quality, reason=reason, note=note, original_segments=orig,
                   corrected_segments=corr, boundary_shift_s=shift, features=promo["feature_vector"],
                   corrected_features=None, evaluation_only=bool(ce.get("held_out")),
                   evaluation_only_reason="held-out episode for this show" if ce.get("held_out") else
                   ("needs_local_refeaturize" if corr else ""))
    PUBLIC_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PUBLIC_FILE, "a", encoding="utf-8") as f:
        f.write(rec.model_dump_json() + "\n")
    return rec.model_dump()


def load_public(show_id: str) -> list[dict]:
    if not PUBLIC_FILE.exists():
        return []
    return [r for r in (json.loads(l) for l in PUBLIC_FILE.read_text(encoding="utf-8").splitlines() if l.strip())
            if r["show_id"] == show_id]
