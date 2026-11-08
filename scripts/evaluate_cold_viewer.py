"""Before vs after Cold Viewer, on the existing candidates (no API calls).

  python scripts/evaluate_cold_viewer.py
BEFORE = candidate-generator pre_score.   AFTER = 0.5 * pre_norm + 0.5 * cv_norm.
Both the cv_score formula and the 50/50 blend were fixed BEFORE looking at results and are NOT tuned
on the gold set (episodes 3 and 27 are a development check, not training data).
Writes data/cold_viewer/eval_report.md and data/cold_viewer/ranked_candidates.json.
"""
import json, sys, warnings
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from promo.parser import fmt_time
from promo.show import get_show
from promo.evaluation import gold_spans, match

warnings.simplefilter("ignore")
SH = get_show(sys.argv[1] if len(sys.argv) > 1 else None)
eps = {e.ep: e for e in SH.load_episodes()}
cands = json.loads((SH.candidates_dir / "candidates.json").read_text(encoding="utf-8"))
cvd = json.loads((SH.cold_viewer_dir / "cold_viewer.json").read_text(encoding="utf-8"))
cv = {r["cand_id"]: r for r in cvd["results"]}
gold = gold_spans(pd.read_csv(sorted(SH.gold_dir.glob("gold_dev_*.csv"))[0]), eps)

lines = []
def out(s=""):
    print(s); lines.append(s)

# ---------------------------------------------------------------- scores and ranks
judged = [c for c in cands if c["cand_id"] in cv]
pre = [c["pre_score"] for c in judged]; cvs = [cv[c["cand_id"]]["cv_score"] for c in judged]
mm = lambda x, lo, hi: (x - lo) / (hi - lo) if hi > lo else 0.5
for c in judged:
    r = cv[c["cand_id"]]
    c["cv"] = r
    c["pre_norm"] = mm(c["pre_score"], min(pre), max(pre))
    c["cv_norm"] = mm(r["cv_score"], min(cvs), max(cvs))
    c["after"] = round(0.5 * c["pre_norm"] + 0.5 * c["cv_norm"], 4)
by_ep = {}
for c in judged:
    by_ep.setdefault(c["ep"], []).append(c)
for ep, L in by_ep.items():
    for i, c in enumerate(sorted(L, key=lambda c: -c["pre_score"]), 1):
        c["rank_before"] = i
    for i, c in enumerate(sorted(L, key=lambda c: -c["after"]), 1):
        c["rank_after"] = i

m = cvd["meta"]; s = m["stats"]
out(f"# Cold Viewer evaluation\n\nmodel `{m['model']}` · prompt {m['prompt_version']} · judged {m['n_judged']}/"
    f"{m['n_candidates']} · API calls {s['calls']} · cache hits {s['cache_hits']} · retries {s['retries']} · "
    f"failed {len(m['failed'])}")
V = Counter(c["cv"]["verdict"] for c in judged)
out(f"\nverdicts: post {V['post']} · maybe {V['maybe']} · reject {V['reject']}")
df = pd.DataFrame([c["cv"] for c in judged])
cols = ["clarity", "hook_3s", "context_dependency", "curiosity", "stakes_score", "ending_tension", "spoiler_risk"]
out("mean scores (1-5): " + " · ".join(f"{k} {df[k].mean():.2f}" for k in cols))
out(f"unresolved_ending true {df.unresolved_ending.mean():.0%} · would_keep_watching true {df.would_keep_watching.mean():.0%}")
corr = pd.Series(pre).rank().corr(pd.Series(cvs).rank())  # Spearman without scipy
out(f"Spearman(pre_score, cv_score) across all candidates = {corr:.2f}  (low = Cold Viewer adds independent signal)")

# ---------------------------------------------------------------- gold before vs after
out("\n## Gold set (dev episodes) — matched candidate = best IoU >= 0.5\n")
out("| gold | label | cand | IoU | rank before | rank after | verdict | clar | hook | ctx | cur | end | spoil |")
out("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
pairs_b = pairs_a = n_pairs = 0
matched = {}
for g in gold.itertuples():
    c, v = match(g, by_ep.get(g.ep, []))
    matched[g.promo_id] = c
    if c is None:
        out(f"| {g.promo_id} | {g.label} | — | {v} | not in candidates | | | | | | | | |"); continue
    r = c["cv"]
    out(f"| {g.promo_id} | {g.label} | {c['cand_id']} | {v} | {c['rank_before']} | {c['rank_after']} | {r['verdict']} | "
        f"{r['clarity']} | {r['hook_3s']} | {r['context_dependency']} | {r['curiosity']} | {r['ending_tension']} | {r['spoiler_risk']} |")
for ep in gold.ep.unique():
    P = [matched[g.promo_id] for g in gold[(gold.ep == ep) & (gold.label == "pos")].itertuples() if matched[g.promo_id]]
    N = [matched[g.promo_id] for g in gold[(gold.ep == ep) & (gold.label == "neg")].itertuples() if matched[g.promo_id]]
    for p in P:
        for n in N:
            if p is n:
                continue
            n_pairs += 1; pairs_b += p["rank_before"] < n["rank_before"]; pairs_a += p["rank_after"] < n["rank_after"]
pos = gold[gold.label == "pos"]
def rec(k, key):
    return sum(1 for g in pos.itertuples() if matched[g.promo_id] and matched[g.promo_id][key] <= k)
out(f"\nrecall of gold positives within top-k per episode (n={len(pos)}):")
out("| k | before | after |\n|---|---|---|")
for k in (1, 2, 3, 5):
    out(f"| {k} | {rec(k, 'rank_before')}/{len(pos)} | {rec(k, 'rank_after')}/{len(pos)} |")
if n_pairs:
    out(f"\npositive ranked above negative (same episode, both matched): before {pairs_b}/{n_pairs} · after {pairs_a}/{n_pairs}")
mrr = lambda key: sum(1 / matched[g.promo_id][key] for g in pos.itertuples() if matched[g.promo_id]) / len(pos)
out(f"mean reciprocal rank of gold positives: before {mrr('rank_before'):.2f} · after {mrr('rank_after'):.2f}")

# ---------------------------------------------------------------- failure cases
out("\n## Failure cases to inspect\n")
for g in gold.itertuples():
    c = matched[g.promo_id]
    if not c:
        continue
    r = c["cv"]
    if g.label == "neg" and (r["verdict"] == "post" or c["rank_after"] <= 2):
        out(f"- ACCEPTED WEAK: {g.promo_id} (human: {g.neg_reason}) → {c['cand_id']} verdict {r['verdict']}, "
            f"rank {c['rank_after']}, ctx {r['context_dependency']}. CV: \"{r['reason']}\"")
    if g.label == "pos" and (r["verdict"] == "reject" or c["rank_after"] > c["rank_before"] + 2):
        kind = "REJECTED STRONG" if r["verdict"] == "reject" else "DEMOTED STRONG"
        out(f"- {kind}: {g.promo_id} (human q{g.quality}) → {c['cand_id']} verdict {r['verdict']}, "
            f"rank {c['rank_before']}→{c['rank_after']}. CV: \"{r['reason']}\"")
big_down = sorted([c for c in judged if c["rank_before"] <= 2 and c["cv"]["verdict"] == "reject"], key=lambda c: c["ep"])
big_up = sorted([c for c in judged if c["rank_before"] >= 6 and c["cv"]["verdict"] == "post"], key=lambda c: c["ep"])
out(f"\ngenerator top-2 that Cold Viewer rejects: {len(big_down)}")
for c in big_down[:8]:
    out(f"- {c['cand_id']} {fmt_time(c['start'])}-{fmt_time(c['end'])}: \"{c['cv']['reason']}\"")
out(f"\ngenerator rank >= 6 that Cold Viewer would post: {len(big_up)}")
for c in big_up[:8]:
    out(f"- {c['cand_id']} {fmt_time(c['start'])}-{fmt_time(c['end'])}: \"{c['cv']['reason']}\"")
contra = [c for c in judged if c["cv"]["context_dependency"] >= 4 and c["cv"]["clarity"] >= 4]
cliff_miss = [c for c in judged if "episode cliffhanger" in " ".join(c["reasons"]) and c["cv"]["ending_tension"] <= 2]
out(f"\nconsistency flags: clarity>=4 but context_dependency>=4: {len(contra)} · "
    f"ends on the episode cliffhanger but ending_tension<=2: {len(cliff_miss)}")

# ---------------------------------------------------------------- top per episode
out("\n## Top 3 per episode after Cold Viewer\n")
for ep in sorted(by_ep):
    top = sorted(by_ep[ep], key=lambda c: c["rank_after"])[:3]
    out(f"ep{ep:02d}: " + " · ".join(f"{c['cand_id']} {fmt_time(c['start'])}-{fmt_time(c['end'])} "
                                       f"[{c['cv']['verdict']}, {c['after']:.2f}]" for c in top))

(SH.cold_viewer_dir / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
keep = ("cand_id", "ep", "start", "end", "start_ts", "end_ts", "duration", "pre_score", "pre_norm", "cv_norm",
        "after", "rank_before", "rank_after", "reasons", "features", "contrib")
(SH.cold_viewer_dir / "ranked_candidates.json").write_text(json.dumps(
    [{k: c[k] for k in keep} | {"cold_viewer": {k: v for k, v in c["cv"].items() if k != "clip_text"}}
     for c in sorted(judged, key=lambda c: (c["ep"], c["rank_after"]))], indent=1, ensure_ascii=False), encoding="utf-8")
print("\nwrote data/cold_viewer/eval_report.md and ranked_candidates.json")
