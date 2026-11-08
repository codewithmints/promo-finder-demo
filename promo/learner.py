"""Interpretable feedback learner at the scoring layer (no model fine-tuning).

Training data = preference pairs built from human feedback (never from held-out records):
  1. accepted  >  rejected          (within the same show)
  2. quality 1 >  quality 3         (among accepted)
  3. corrected boundaries > original boundaries   (a reviewer's correction is a direct preference)
Model: score = w · x on the same 13 interpretable features the ranker already uses.
Fit:   pairwise logistic loss + lambda * ||w - w_current||^2, signs fixed (bonuses stay >= 0, penalties <= 0),
       so with little data the weights barely move from the current ones.
Gate:  enough data (min accepted / rejected / episodes), then grouped-by-episode cross-validation:
       the candidate must beat the CURRENT weights on held-out folds by min_auc_gain, and win in at least
       min_bootstrap_win_rate of episode-bootstrap resamples. Otherwise status = not_promoted.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from . import weights as Wt

GATE = {"min_accepted": 60, "min_rejected": 20, "min_episodes": 8, "min_auc_gain": 0.02, "min_bootstrap_win_rate": 0.8}
LAMBDA = 1.0          # pull toward the current weights at ~50 records; fades as feedback grows (lambda * 50 / n)
FEATS = Wt.KEYS


def _vec(d: dict) -> np.ndarray:
    return np.array([float(d.get(k, 0.0)) for k in FEATS])


def build_pairs(records: list[dict]) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """(better, worse, episode_of_better, episode_of_worse) pairs."""
    assert not any(r["evaluation_only"] for r in records), "held-out records must never reach the learner"
    acc = [r for r in records if r["accepted"]]
    rej = [r for r in records if not r["accepted"]]
    pairs = [(_vec(a["features"]), _vec(b["features"]), a["episode_id"], b["episode_id"]) for a in acc for b in rej]
    q1 = [r for r in acc if r.get("quality") == 1]; q3 = [r for r in acc if r.get("quality") == 3]
    pairs += [(_vec(a["features"]), _vec(b["features"]), a["episode_id"], b["episode_id"]) for a in q1 for b in q3]
    for r in records:
        if r.get("corrected_features"):
            pairs.append((_vec(r["corrected_features"]), _vec(r["features"]), r["episode_id"], r["episode_id"]))
    return pairs


def fit(pairs, w0: dict, n_records: int = 50, lam: float = LAMBDA, iters: int = 800, lr: float = 0.1) -> dict:
    w0v = _vec(w0)
    if not pairs:
        return dict(w0)
    D = np.array([a - b for a, b, *_ in pairs])
    sign = np.array([Wt.SIGN[k] for k in FEATS])
    lam = lam * min(1.0, 50.0 / max(1, n_records))
    w = w0v.copy()
    for _ in range(iters):
        z = D @ w
        g = -(D * (1 / (1 + np.exp(np.clip(z, -30, 30))))[:, None]).mean(0) + 2 * lam * (w - w0v)
        w -= lr * g
        w = np.where(sign > 0, np.maximum(w, 0), np.minimum(w, 0))     # keep interpretable signs
    return {k: round(float(v), 4) for k, v in zip(FEATS, w)}


def pair_accuracy(pairs, w: dict) -> float:
    if not pairs:
        return float("nan")
    wv = _vec(w)
    s = [(a @ wv > b @ wv) + 0.5 * (a @ wv == b @ wv) for a, b, *_ in pairs]
    return float(np.mean(s))


def grouped_cv(records: list[dict], w_current: dict, folds: int = 5, seed: int = 0) -> dict:
    """Train on some episodes, test on pairs whose items are both in unseen episodes."""
    eps = sorted({r["episode_id"] for r in records})
    random.Random(seed).shuffle(eps)
    k = min(folds, len(eps)) or 1
    groups = [set(eps[i::k]) for i in range(k)]
    cur, cand, per_fold, oof = [], [], [], []      # oof: (episode, correct_current, correct_candidate) per test pair
    for g in groups:
        train = [r for r in records if r["episode_id"] not in g]
        test_pairs = [p for p in build_pairs([r for r in records if r["episode_id"] in g])]
        if not test_pairs:
            continue
        w = fit(build_pairs(train), w_current, n_records=len(train))
        a_cur, a_cand = pair_accuracy(test_pairs, w_current), pair_accuracy(test_pairs, w)
        cur.append((a_cur, len(test_pairs))); cand.append((a_cand, len(test_pairs)))
        wc, wn = _vec(w_current), _vec(w)
        for a_, b_, ea, eb in test_pairs:
            oof.append((ea, (a_ @ wc > b_ @ wc) + 0.5 * (a_ @ wc == b_ @ wc), (a_ @ wn > b_ @ wn) + 0.5 * (a_ @ wn == b_ @ wn)))
        per_fold.append({"episodes": len(g), "test_pairs": len(test_pairs), "current": round(a_cur, 3), "candidate": round(a_cand, 3)})
    wavg = lambda xs: sum(a * n for a, n in xs) / max(1, sum(n for _, n in xs))
    return {"current": round(wavg(cur), 4) if cur else None, "candidate": round(wavg(cand), 4) if cand else None,
            "folds": per_fold, "_oof": oof}


def bootstrap_win_rate(oof, n: int = 500, seed: int = 0) -> float:
    """Share of episode-bootstrap resamples of the OUT-OF-FOLD test pairs where the candidate beats current."""
    by = defaultdict(list)
    for ep, c0, c1 in oof:
        by[ep].append((c0, c1))
    eps = sorted(by)
    if not eps:
        return 0.0
    rng = random.Random(seed); wins = 0
    for _ in range(n):
        smp = [x for e in (rng.choice(eps) for _ in eps) for x in by[e]]
        wins += sum(c1 for _, c1 in smp) > sum(c0 for c0, _ in smp)
    return wins / n


def data_summary(records: list[dict]) -> dict:
    return {"records": len(records), "accepted": sum(r["accepted"] for r in records),
            "rejected": sum(not r["accepted"] for r in records), "episodes": len({r["episode_id"] for r in records}),
            "boundary_corrections": sum(1 for r in records if r.get("corrected_features")),
            "sources": dict(Counter(r["source"] for r in records))}


def gate(records: list[dict], w_current: dict, gate_cfg: dict = GATE) -> dict:
    """Fit a candidate and decide promotion. Returns everything needed for the version file and report."""
    d = data_summary(records)
    pairs = build_pairs(records)
    w_cand = fit(pairs, w_current, n_records=len(records))
    reasons = []
    if d["accepted"] < gate_cfg["min_accepted"]:
        reasons.append(f"accepted {d['accepted']} < {gate_cfg['min_accepted']}")
    if d["rejected"] < gate_cfg["min_rejected"]:
        reasons.append(f"rejected {d['rejected']} < {gate_cfg['min_rejected']}")
    if d["episodes"] < gate_cfg["min_episodes"]:
        reasons.append(f"episodes {d['episodes']} < {gate_cfg['min_episodes']}")
    cv = grouped_cv(records, w_current) if records else {"current": None, "candidate": None, "folds": [], "_oof": []}
    win = bootstrap_win_rate(cv.pop("_oof"))
    if cv["current"] is None or cv["candidate"] is None:
        reasons.append("no held-out-fold pairs to validate on")
    else:
        if cv["candidate"] - cv["current"] < gate_cfg["min_auc_gain"]:
            reasons.append(f"cross-validated pair accuracy gain {cv['candidate'] - cv['current']:+.3f} < {gate_cfg['min_auc_gain']}")
        if win < gate_cfg["min_bootstrap_win_rate"]:
            reasons.append(f"bootstrap win rate {win:.2f} < {gate_cfg['min_bootstrap_win_rate']}")
    return {"weights": w_cand, "promote": not reasons, "reasons": reasons, "data": d, "pairs": len(pairs),
            "train_pair_accuracy": {"current": round(pair_accuracy(pairs, w_current), 4) if pairs else None,
                                    "candidate": round(pair_accuracy(pairs, w_cand), 4) if pairs else None},
            "cv": cv, "bootstrap_win_rate": round(win, 3), "gate": gate_cfg,
            "weight_change": {k: round(w_cand[k] - w_current[k], 4) for k in FEATS if abs(w_cand[k] - w_current[k]) >= 0.0005}}


def failure_patterns(records: list[dict]) -> dict:
    """Rejection reasons and how rejected promos differ from accepted ones, feature by feature."""
    acc = [r for r in records if r["accepted"]]
    rej = [r for r in records if not r["accepted"]]
    mean = lambda rs, k: float(np.mean([r["features"].get(k, 0) for r in rs])) if rs else None
    by_reason = defaultdict(list)
    for r in rej:
        by_reason[r["reason"] or "unspecified"].append(r)
    feats = {k: {"accepted": mean(acc, k), "rejected": mean(rej, k)} for k in FEATS}
    reasons = {}
    for reason, rs in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        diffs = {k: round(mean(rs, k) - mean(acc, k), 3) for k in FEATS if acc and mean(rs, k) is not None}
        top = sorted(diffs.items(), key=lambda kv: -abs(kv[1]))[:3]
        reasons[reason] = {"count": len(rs), "largest_feature_differences_vs_accepted": top}
    shifts = [s for r in records for s in (r.get("boundary_shift_s") or [])]
    return {"reasons": reasons, "feature_means": feats,
            "boundary_corrections": {"count": len(shifts),
                                     "mean_start_shift_s": round(float(np.mean([s[0] for s in shifts])), 2) if shifts else None,
                                     "mean_end_shift_s": round(float(np.mean([s[1] for s in shifts])), 2) if shifts else None}}
