"""Generate promo candidates for every episode with Pass A output (no API calls).

  python scripts/run_candidates.py [--top-k 8]
Outputs data/candidates/{candidates.json, candidates.csv}; prints recall vs the dev gold set
and two baselines (last-45s, random valid windows).
"""
import argparse, json, random, sys, time, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from promo.show import get_show, add_show_arg
from promo.candidates import generate, iou, MIN_DUR, MAX_DUR

ap = argparse.ArgumentParser()
ap.add_argument("--top-k", type=int, default=8)
ap.add_argument("--gold", default=None, help="dev gold CSV (default: <work>/gold/gold_dev_*.csv if present)")
add_show_arg(ap)
a = ap.parse_args()
SH = get_show(a.show)
if a.gold is None:
    g = sorted(SH.gold_dir.glob("gold_dev_*.csv"))
    a.gold = str(g[0]) if g else None
warnings.simplefilter("ignore")

eps = {e.ep: e for e in SH.load_episodes()}
show_f = SH.pass_a_dir / "show.json"
spoil = {r["episode"] for r in json.loads(show_f.read_text(encoding="utf-8"))["reveals"]
         if r["kind"] != "premise"} if show_f.exists() else set()

t0 = time.time(); allc, n_windows = [], 0
for ep, e in sorted(eps.items()):
    f = SH.pass_a_dir / f"ep{ep:02d}.json"
    if not f.exists():
        continue
    cands, n = generate(e, json.loads(f.read_text(encoding="utf-8")), spoil, top_k=a.top_k)
    allc += cands; n_windows += n
out = SH.candidates_dir; out.mkdir(parents=True, exist_ok=True)
recs = [c.asdict() for c in allc]
(out / "candidates.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
flat = [{k: v for k, v in r.items() if k not in ("features", "contrib", "reasons")}
        | {f"f_{k}": v for k, v in r["features"].items()} | {"reasons": " | ".join(r["reasons"])} for r in recs]
pd.DataFrame(flat).to_csv(out / "candidates.csv", index=False)
print(f"{len(eps)} episodes · {n_windows:,} legal windows scored · {len(allc)} candidates kept "
      f"(top {a.top_k}/episode, IoU<=0.5) · {time.time()-t0:.1f}s")

# ---------------------------------------------------------------- dev-gold recall + baselines
if not a.gold:
    print("no dev gold for this show: skipping recall check"); sys.exit()
gold = pd.read_csv(a.gold)
pos = gold[gold.label == "pos"]
if pos.empty:
    sys.exit()

def span(ep, r):
    e = eps[ep]; U = {u.unit_id: u for u in e.units}; S = {s.shot_id: s for s in e.shots}
    g = lambda ref, w: (S[ref].t_start if w == "s" else S[ref].t_end) if ref in S else (U[ref].t_start if w == "s" else U[ref].t_end)
    return g(r.start_ref, "s"), g(r.end_ref, "e")

def recall(pred_by_ep, k, thr=0.5):
    hit = [max([iou(span(r.ep, r), p) for p in pred_by_ep.get(r.ep, [])[:k]] or [0]) >= thr for r in pos.itertuples()]
    return sum(hit), len(hit)

ours = {}
for c in allc:
    ours.setdefault(c.ep, []).append((c.start, c.end))
last45 = {}
for ep in pos.ep.unique():
    e = eps[ep]; end = max(e.legal_ends); st = min(e.legal_starts, key=lambda x: abs((end - x) - 45))
    last45[ep] = [(st, end)]
rng = random.Random(0); rand_runs = []
for _ in range(200):
    rp = {}
    for ep in pos.ep.unique():
        e = eps[ep]; pairs = [(s, t) for s in e.legal_starts for t in e.legal_ends if MIN_DUR <= t - s <= MAX_DUR]
        rp[ep] = rng.sample(pairs, a.top_k)
    rand_runs.append(rp)

print(f"\nDev gold recall (positives = {len(pos)}, match = temporal IoU >= 0.5)")
print(f"{'method':<22}" + "".join(f"@{k:<6}" for k in (1, 3, 5, a.top_k)))
print(f"{'ours':<22}" + "".join(f"{recall(ours, k)[0]}/{len(pos):<4} " for k in (1, 3, 5, a.top_k)))
print(f"{'last 45s':<22}" + f"{recall(last45, 1)[0]}/{len(pos):<4} " + "(single window)")
rr = {k: sum(recall(r, k)[0] for r in rand_runs) / len(rand_runs) for k in (1, 3, 5, a.top_k)}
print(f"{'random valid (avg)':<22}" + "".join(f"{rr[k]:.1f}/{len(pos):<2} " for k in (1, 3, 5, a.top_k)))
print("\nper gold promo: best IoU among our top-K (rank of that candidate)")
for r in pos.itertuples():
    g = span(r.ep, r); P = ours.get(r.ep, [])
    best = max(range(len(P)), key=lambda i: iou(g, P[i])) if P else None
    if best is not None:
        print(f"  {r.promo_id}  gold {g[0]:6.1f}-{g[1]:6.1f}  best #{best+1} {P[best][0]:6.1f}-{P[best][1]:6.1f}  IoU {iou(g, P[best]):.2f}")
