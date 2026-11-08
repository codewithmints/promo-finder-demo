"""Gold-set helpers shared by evaluation scripts."""
from __future__ import annotations

import pandas as pd

from .parser import Episode
from .candidates import iou


def gold_spans(gold: pd.DataFrame, eps: dict[int, Episode]) -> pd.DataFrame:
    """Resolve start/end/hook refs (shot or unit ids) to exact seconds."""
    rows = []
    for r in gold.itertuples():
        e = eps[r.ep]
        U = {u.unit_id: u for u in e.units}
        S = {s.shot_id: s for s in e.shots}

        def t(ref, which):
            if ref in S:
                return S[ref].t_start if which == "s" else S[ref].t_end
            return U[ref].t_start if which == "s" else U[ref].t_end
        rows.append({**r._asdict(), "t0": t(r.start_ref, "s"), "t1": t(r.end_ref, "e")})
    return pd.DataFrame(rows).drop(columns=["Index"])


def match(gold_row, cands: list[dict], thr: float = 0.5):
    """Best-overlapping candidate (dict) for a gold row, or None if IoU < thr."""
    best, bi = None, 0.0
    for c in cands:
        v = iou((gold_row.t0, gold_row.t1), (c["start"], c["end"]))
        if v > bi:
            best, bi = c, v
    return (best, round(bi, 2)) if bi >= thr else (None, round(bi, 2))
