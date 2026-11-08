"""Compare Pass A tags with the gold promos (dev episodes). No API calls.

  python scripts/check_pass_a.py [gold.csv] [--show id]   # default: the show's first gold/gold_dev_*.csv
For each gold promo: hook tag at the hook block, cliff tag near the end (max over the last 4s,
since the cliffhanger line often sits a beat before the cut), and calibration stats per episode.
"""
import json, sys, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from promo.show import get_show

warnings.simplefilter("ignore")
_a = [x for x in sys.argv[1:] if not x.startswith("--")]
SH = get_show(sys.argv[sys.argv.index("--show") + 1] if "--show" in sys.argv else None)
if "--show" in sys.argv: _a = [x for x in _a if x != sys.argv[sys.argv.index("--show") + 1]]
gold = pd.read_csv(_a[0] if _a else sorted(SH.gold_dir.glob("gold_dev_*.csv"))[0])
EPS_ = {e.ep: e for e in SH.load_episodes(numbers=set(gold.ep))}
hits = {"hook": [], "cliff": []}
for ep in sorted(gold.ep.unique()):
    f = SH.pass_a_dir / f"ep{ep:02d}.json"
    if not f.exists():
        print(f"ep{ep}: no Pass A output"); continue
    rec = json.loads(f.read_text(encoding="utf-8")); A = rec["blocks"]
    e = EPS_[ep]
    U = {u.unit_id: u for u in e.units}; S = {s.shot_id: s for s in e.shots}
    T = lambda r, w: (S[r].t_start if w == "s" else S[r].t_end) if r in S else (U[r].t_start if w == "s" else U[r].t_end)
    n = len(A)
    print(f"ep{ep:02d} [{rec['prompt_version']}] blocks {n} · hook>=2 {sum(b['hook']>=2 for b in A)/n:.0%} · "
          f"hook=3 {sum(b['hook']==3 for b in A)} · cliff>=2 {sum(b['cliff']>=2 for b in A)} · resolves {sum(b['resolves'] for b in A)}")
    for r in gold[gold.ep == ep].itertuples():
        th, t1 = T(r.hook_ref, "s"), T(r.end_ref, "e")
        hb = next(b for b in A if b["t_start"] - .01 <= th < b["t_end"])
        tail = [b for b in A if b["t_end"] > t1 - 4 and b["t_start"] < t1 - .01]
        cl = max(b["cliff"] for b in tail)
        res = any(b["resolves"] for b in tail)
        ok = "✓" if (hb["hook"] >= 2 and cl >= 2) else "·"
        if r.label == "pos":
            hits["hook"].append(hb["hook"] >= 2); hits["cliff"].append(cl >= 2)
        print(f"  {ok} {r.promo_id} {r.label}  hook {hb['block_id'][4:]}={hb['hook']}  cliff(last 4s)={cl}"
              f"{'  RESOLVES' if res else ''}  {'' if r.label=='pos' else '(' + str(r.neg_reason) + ')'}")
if hits["hook"]:
    print(f"\npositives: hook>=2 at gold hook {sum(hits['hook'])}/{len(hits['hook'])} · "
          f"cliff>=2 at gold ending {sum(hits['cliff'])}/{len(hits['cliff'])}")
