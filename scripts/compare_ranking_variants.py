"""Pre-registered ranking comparison (no API calls). Writes results/ranking_decision.json.

Data allowed for this decision: dev gold (episodes 3, 27) and blind ratings of OUR promos outside the
held-out episodes 1, 12, 34, 45. Held-out gold is never read here.

Rule (fixed before running): a variant is adopted only if, versus v0,
  dev MRR improves AND dev positive-above-negative stays >= 8/9 AND blind AUC (accepted vs rejected) improves.
Candidates (principled, from the diagnosis, not fitted to labels):
  dur    duration preference 22-40s with weight 0.15 (was 25-45s, 0.05)
  midres -0.10 if a conflict resolves before the cliffhanger window
  spoil  spoiler penalty ignores reveals Pass A marks as premise-level
  dur+spoil
"""
import csv, glob, json, sys, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from promo.show import get_show
from promo.evaluation import gold_spans, match
from promo.candidates import W

warnings.simplefilter("ignore")
SH = get_show(sys.argv[1] if len(sys.argv) > 1 else None)
HELD_OUT = set(SH.held_out_episodes)
eps = {e.ep: e for e in SH.load_episodes()}
C = json.loads((SH.candidates_dir / "candidates.json").read_text(encoding="utf-8"))
tags = {json.loads(Path(f).read_text(encoding="utf-8"))["ep"]: json.loads(Path(f).read_text(encoding="utf-8"))
        for f in glob.glob(str(SH.pass_a_dir / "ep*.json"))}
show = json.loads((SH.pass_a_dir / "show.json").read_text(encoding="utf-8"))
spoil_eps = {r["episode"] for r in show["reveals"] if r["kind"] != "premise"}
cv = {r["cand_id"]: r for r in json.loads((SH.cold_viewer_dir / "cold_viewer.json").read_text(encoding="utf-8"))["results"]}
gold = gold_spans(pd.read_csv(sorted(SH.gold_dir.glob("gold_dev_*.csv"))[0]), eps)
assert not set(gold.ep) & HELD_OUT

for c in C:
    T = tags[c["ep"]]
    B = [b for b in T["blocks"] if b["t_end"] > c["start"] + .05 and b["t_start"] < c["end"] - .05]
    premise = {r["block"] for r in T["key_reveals"] if r["level"] == "premise"}
    rev = max([b["reveal"] for b in B if b["block_id"] not in premise] or [0]) / 3
    c["_midres"] = float(any(b["resolves"] for b in B if b["t_end"] <= c["end"] - 4))
    c["_spoil_premise"] = rev * (1.0 if c["ep"] in spoil_eps else 0.5)


def durfit(d, lo=22, hi=40):
    return 1.0 if lo <= d <= hi else max(0, 1 - (lo - d) / 10) if d < lo else max(0, 1 - (d - hi) / (60 - hi))


def score(c, v):
    f, w = dict(c["features"]), dict(W)
    if "dur" in v:
        f["dur"], w["dur"] = durfit(c["duration"]), 0.15
    if "spoil" in v:
        f["spoiler"] = c["_spoil_premise"]
    s = sum(w[k] * f[k] for k in w) - (0.10 * c["_midres"] if "midres" in v else 0)
    r = cv.get(c["cand_id"])
    if r:
        s -= 0.30 * (r["verdict"] == "reject") + (0.10 * (r["context_dependency"] - 2) if r["context_dependency"] >= 3 else 0)
    return s


def dev_metrics(v):
    ranked = {ep: sorted([c for c in C if c["ep"] == ep], key=lambda c: -score(c, v)) for ep in set(gold.ep)}
    rk = {}
    for r in gold.itertuples():
        m, _ = match(r, ranked[r.ep]); rk[r.promo_id] = ranked[r.ep].index(m) + 1 if m else None
    pos = [r for r in gold.itertuples() if r.label == "pos"]; neg = [r for r in gold.itertuples() if r.label == "neg"]
    P = [rk[r.promo_id] for r in pos]
    pr = [rk[a.promo_id] < rk[b.promo_id] for a in pos for b in neg if a.ep == b.ep and rk[a.promo_id] and rk[b.promo_id]]
    return {"top1": sum(p == 1 for p in P if p), "top3": sum(p <= 3 for p in P if p),
            "mrr": round(sum(1 / p for p in P if p) / len(P), 3), "pairs_ok": sum(pr), "pairs": len(pr)}


key = json.loads((SH.blind_dir / "blind_key.json").read_text(encoding="utf-8"))
rat = {r["item_id"]: r for r in csv.DictReader(open(SH.blind_dir / "blind_ratings.csv", encoding="utf-8"))}
prom = {p["promo_id"]: p["candidate_id"] for p in json.loads((ROOT / "results/v0_default/promos_ranked.json").read_text(encoding="utf-8"))}
byid = {c["cand_id"]: c for c in C}
items = [(byid[prom[k["ref"]]], rat[b]["post"] == "Y") for b, k in key.items()
         if k["src"] == "ours" and k["ep"] not in HELD_OUT and rat.get(b, {}).get("post") in ("Y", "N")]
acc = [c for c, y in items if y]; rej = [c for c, y in items if not y]


def blind_auc(v):
    s = {c["cand_id"]: score(c, v) for c, _ in items}
    pr = [(s[a["cand_id"]] > s[b["cand_id"]]) + 0.5 * (s[a["cand_id"]] == s[b["cand_id"]]) for a in acc for b in rej]
    return round(sum(pr) / len(pr), 3)


base = {**dev_metrics(set()), "blind_auc": blind_auc(set())}
rows = [{"variant": "v0", **base, "passed": None}]
for name, v in [("dur", {"dur"}), ("midres", {"midres"}), ("spoil", {"spoil"}), ("dur+spoil", {"dur", "spoil"})]:
    d = {**dev_metrics(v), "blind_auc": blind_auc(v)}
    d["passed"] = d["mrr"] > base["mrr"] and d["pairs_ok"] >= 8 and d["blind_auc"] > base["blind_auc"]
    rows.append({"variant": name, **d})
selected = next((r["variant"] for r in rows[1:] if r["passed"]), None)
decision = {"rule": "adopt only if dev MRR improves AND dev pos>neg >= 8/9 AND blind AUC improves vs v0",
            "data": {"dev_gold": "episodes 3, 27", "blind_items": f"{len(acc)} accepted / {len(rej)} rejected (ours, held-out episodes excluded)"},
            "results": rows, "selected": selected or "v0 (no variant passed)"}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/ranking_decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
print(f"blind items: {len(acc)} accepted / {len(rej)} rejected")
print(f"{'variant':<11}{'top1':>5}{'top3':>5}{'MRR':>7}{'pos>neg':>9}{'blindAUC':>10}  result")
for r in rows:
    print(f"{r['variant']:<11}{r['top1']:>5}{r['top3']:>5}{r['mrr']:>7}{str(r['pairs_ok'])+'/'+str(r['pairs']):>9}{r['blind_auc']:>10}  "
          f"{'baseline' if r['passed'] is None else 'PASS' if r['passed'] else 'fail'}")
print("selected:", decision["selected"])
