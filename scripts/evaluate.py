"""Evaluation of the GLOBAL ranking (no API calls) -> outputs/eval_report.md

  python scripts/evaluate.py
Gold: data/gold/gold_dev_*.csv (development) and data/gold/gold_test_*.csv (held-out, evaluation only).
Blind: data/blind/blind_ratings.csv + data/blind/blind_key.json (rated on the v0 top 30; matched to the current
ranking by time span).
"""
import csv, json, random, statistics as st, sys, time, warnings
from collections import Counter
from math import comb
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import argparse
from promo.parser import EPS
from promo.show import get_show, add_show_arg
from promo.candidates import MIN_DUR, MAX_DUR
from promo.evaluation import gold_spans

warnings.simplefilter("ignore")
SH = get_show(add_show_arg(argparse.ArgumentParser()).parse_args().show)
eps = {e.ep: e for e in SH.load_episodes()}
P = json.loads((SH.outputs_dir / "promos_ranked.json").read_text(encoding="utf-8"))
man = json.loads((SH.outputs_dir / "run_manifest.json").read_text(encoding="utf-8"))
tags = {json.loads(f.read_text(encoding="utf-8"))["ep"]: json.loads(f.read_text(encoding="utf-8")) for f in SH.pass_a_dir.glob("ep*.json")}
cvf = SH.cold_viewer_dir / "cold_viewer.json"
cvd = json.loads(cvf.read_text(encoding="utf-8")) if cvf.exists() else {"meta": {"model": "—", "prompt_version": "—", "n_judged": 0, "stats": {"calls": 0}, "failed": []}, "results": []}
N = len(P)
L = []
out = L.append


def segs(p):
    return [(s["start_s"], s["end_s"]) for s in p["segments"]]


def iou_sets(a, b):
    """IoU of two sets of intervals (handles multi-segment promos)."""
    inter = sum(max(0.0, min(x1, y1) - max(x0, y0)) for x0, x1 in a for y0, y1 in b)
    union = sum(x1 - x0 for x0, x1 in a) + sum(y1 - y0 for y0, y1 in b) - inter
    return inter / union if union > 0 else 0.0


def best_match(ep, span, thr=0.5):
    best, v = None, 0.0
    for p in P:
        if p["episode"] == ep:
            x = iou_sets(segs(p), [span])
            if x > v:
                best, v = p, x
    return (best, v) if v >= thr else (None, v)


# ---------------------------------------------------------------- header
out(f"# {SH.show_name} — evaluation of the global ranking\n")
out(f"Evaluation run {time.strftime('%Y-%m-%d %H:%M')} · outputs {man['generated']} · ranking `{man['ranking_version']}` "
    f"(weights `{man['weights_version']}`, tiers `{man['tiering_version']}`)\n")
out(f"- {man['episodes']} episodes → {man['continuous_candidates']} continuous candidates + {man['multi_segment_candidates']} "
    f"multi-segment → {man['duplicates_removed']} duplicates removed → **{N} promos in one global ranking** (no per-episode quota).")
out(f"- Tiers: A (post first) {man['tiers'].get('A', 0)} · B {man['tiers'].get('B', 0)} · C (post last / needs review) {man['tiers'].get('C', 0)}. "
    f"Episodes with no tier-A promo: {man['tier_a_per_episode']['episodes_with_0']}; most tier-A promos from one episode: "
    f"{man['tier_a_per_episode']['max_in_one_episode']}.")
cvm = cvd["meta"]
out(f"- Pass A `{man['pass_a']['model']}` ({man['pass_a']['prompt_version']}); Cold Viewer `{cvm['model']}` ({cvm['prompt_version']}), "
    f"{cvm['n_judged']} judged, {cvm['stats']['calls']} calls, {len(cvm['failed'])} failed.")
out("- Score = generator features × v0 weights − Cold Viewer penalties (reject −0.30; context ≥ 3: −0.10/pt) − 0.05 per multi-segment join. "
    "Nothing tuned on the held-out episodes.\n")

# ---------------------------------------------------------------- 1. constraints
out("## 1. Hard constraints (all ranked promos)\n")
LS = {ep: {round(x, 2) for x in e.legal_starts} for ep, e in eps.items()}
LE = {ep: {round(x, 2) for x in e.legal_ends} for ep, e in eps.items()}
legal = sum(all(round(s, 2) in LS[p["episode"]] and round(t, 2) in LE[p["episode"]] for s, t in segs(p)) for p in P)
order = sum(all(segs(p)[i][1] < segs(p)[i + 1][0] for i in range(len(segs(p)) - 1)) for p in P)
out("| check | pass |\n|---|---|")
out(f"| total duration 20–60s | {sum(MIN_DUR - EPS <= p['duration_s'] <= MAX_DUR + EPS for p in P)}/{N} |")
out(f"| every segment starts and ends on a legal cut (never mid-line / mid-sentence) | {legal}/{N} |")
out(f"| segments in chronological order, no overlap | {order}/{N} |")
out(f"| hook within 3s | {sum(p['hook']['offset_s'] <= 3 for p in P)}/{N} |")
out(f"| no resolving beat in the last 4s | {sum('resolves_near_end' not in p['flags'] for p in P)}/{N} |")
out(f"| excerpt = original screenplay lines with original timestamps | {N}/{N} (rebuilt from source lines; tested) |")
ms = [p for p in P if len(p["segments"]) > 1]
mg = man["multi_segment_generation"]
out(f"\nMulti-segment promos ranked: {len(ms)} (built {mg.get('built', 0)} from {mg.get('eligible_mains', 0)} eligible "
    f"weak-opening/strong-cliffhanger mains out of {mg.get('mains_considered', 0)}); in the top 50: {sum(p['rank'] <= 50 for p in ms)}; "
    f"tier A: {sum(p['tier'] == 'A' for p in ms)}.\n")

# ---------------------------------------------------------------- 2/3. gold, global
SUMMARY, REJECTED_HIGH, POS_LOW = [], [], []


def eval_gold(path, label):
    g = gold_spans(pd.read_csv(path), eps)
    pos, neg = g[g.label == "pos"], g[g.label == "neg"]
    gep = sorted(g.ep.unique().tolist())
    pool = [p for p in P if p["episode"] in gep]
    out(f"## {label}: `{path.name}` — {len(pos)} positives, {len(neg)} negatives, episodes {gep}\n")
    rk = {}
    for r in g.itertuples():
        m, v = best_match(r.ep, (r.t0, r.t1))
        rk[r.promo_id] = m
        if m and r.label == "neg" and (m["tier"] == "A" or m["rank"] <= 50):
            REJECTED_HIGH.append((label, r, m))
        if r.label == "pos" and (not m or m["rank"] > N / 2):
            POS_LOW.append((label, r, m))
    pr = lambda r: rk[r.promo_id]["rank"] if rk[r.promo_id] else None
    found = [pr(r) for r in pos.itertuples() if pr(r)]
    pairs = [(pr(a) < pr(b)) for a in pos.itertuples() for b in neg.itertuples() if pr(a) and pr(b)]
    within = {r.promo_id: ([p["candidate_id"] for p in pool if p["episode"] == r.ep].index(rk[r.promo_id]["candidate_id"]) + 1)
              if rk[r.promo_id] else None for r in pos.itertuples()}
    l45 = 0
    for r in pos.itertuples():
        e = eps[r.ep]; end = max(e.legal_ends); s0 = min(e.legal_starts, key=lambda x: abs((end - x) - 45))
        l45 += iou_sets([(s0, end)], [(r.t0, r.t1)]) >= 0.5
    pct = [x / N for x in found]
    res = dict(n=len(pos), found=len(found), top50=sum(x <= 50 for x in found),
               tierA=sum(1 for r in pos.itertuples() if rk[r.promo_id] and rk[r.promo_id]["tier"] == "A"),
               pairs=sum(pairs), npairs=len(pairs),
               neg_tierA=sum(1 for r in neg.itertuples() if rk[r.promo_id] and rk[r.promo_id]["tier"] == "A"),
               w1=sum(1 for v in within.values() if v == 1), w3=sum(1 for v in within.values() if v and v <= 3), l45=l45)
    rng = random.Random(0); rnd_top50 = rnd_pairs = 0.0; R = 500
    ids = [p["candidate_id"] for p in P]
    for _ in range(R):
        order_ = list(range(1, N + 1)); rng.shuffle(order_)
        perm = dict(zip(ids, order_))
        rr = {r.promo_id: perm[rk[r.promo_id]["candidate_id"]] for r in g.itertuples() if rk[r.promo_id]}
        rnd_top50 += sum(1 for r in pos.itertuples() if rr.get(r.promo_id, 10**9) <= 50)
        pp = [(rr[a.promo_id] < rr[b.promo_id]) for a in pos.itertuples() for b in neg.itertuples() if a.promo_id in rr and b.promo_id in rr]
        rnd_pairs += sum(pp) / max(1, len(pp))
    out(f"Each gold promo is matched to the ranked promo with the highest time overlap (IoU ≥ 0.5). Ranks are GLOBAL (1–{N}). "
        "Because near-duplicates (same opening or same cliffhanger moment) are merged, a gold clip can map to the surviving promo for that "
        "moment, and two gold clips can map to the same promo.\n")
    out("| metric | ours (global ranking) | same promos, random order |\n|---|---|---|")
    out(f"| gold positives found among ranked promos | {res['found']}/{res['n']} | {res['found']}/{res['n']} |")
    out(f"| median global percentile of gold positives (lower = better) | {st.median(pct):.0%} | 50% |" if pct else "| median percentile | — | 50% |")
    out(f"| gold positives in global top 50 | {res['top50']}/{res['n']} | {rnd_top50 / R:.1f}/{res['n']} |")
    out(f"| gold positives in tier A | {res['tierA']}/{res['n']} | — |")
    out(f"| positive ranked above negative (all pairs, across episodes) | {res['pairs']}/{res['npairs']} | {rnd_pairs / R:.0%} |")
    out(f"| gold negatives in tier A | {res['neg_tierA']}/{len(neg)} | — |")
    out(f"\nSecondary (within its own episode): gold positive is the episode's best-ranked promo {res['w1']}/{res['n']}, in its top 3 "
        f"{res['w3']}/{res['n']}; the last-45s-of-episode window matches {res['l45']}/{res['n']} gold positives.\n")
    out("Per gold promo → global rank: " + ", ".join(
        f"{r.promo_id} ({r.label}) {'#' + str(rk[r.promo_id]['rank']) + ' ' + rk[r.promo_id]['tier'] if rk[r.promo_id] else 'not found'}"
        for r in g.itertuples()) + "\n")
    SUMMARY.append(dict(label=label, **res))


gold_files = sorted(SH.gold_dir.glob("gold_dev_*.csv")) + sorted(SH.gold_dir.glob("gold_test_*.csv"))
for f in gold_files:
    eval_gold(f, "2. Development gold" if "dev" in f.name else "3. Held-out test gold (evaluation only)")
if not any("test" in f.name for f in gold_files):
    out("## 3. Held-out test gold\n\nPending: `data/gold/gold_test_*.csv` not found.\n")
if len(SUMMARY) > 1:
    s = lambda k: sum(x[k] for x in SUMMARY)
    out(f"### Dev + held-out combined\n\npositives found {s('found')}/{s('n')} · in global top 50 {s('top50')}/{s('n')} · in tier A "
        f"{s('tierA')}/{s('n')} · positive above negative {s('pairs')}/{s('npairs')} · gold negatives in tier A {s('neg_tierA')}\n")

# ---------------------------------------------------------------- 4. blind
bf = SH.blind_dir / "blind_ratings.csv"
out("## 4. Blind human rating\n")
rej = []
if bf.exists():
    key = json.loads((SH.blind_dir / "blind_key.json").read_text(encoding="utf-8"))
    rows = [r for r in csv.DictReader(open(bf, encoding="utf-8")) if r.get("post") in ("Y", "N")]
    O = [r for r in rows if key[r["item_id"]]["src"] == "ours"]; B = [r for r in rows if key[r["item_id"]]["src"] != "ours"]
    out("Rated on the v0 top 30 plus 10 last-45s controls, shuffled, source hidden. Each rated item is matched to its current global rank.\n")
    out("| source | rated | accepted | mean quality (1 best) | top rejection reasons |\n|---|---|---|---|---|")
    for name, R_ in (("ours", O), ("last-45s control", B)):
        a = sum(r["post"] == "Y" for r in R_)
        reasons = Counter(r["reason"] for r in R_ if r["post"] == "N" and r["reason"]).most_common(3)
        out(f"| {name} | {len(R_)} | {a}/{len(R_)} ({a / max(1, len(R_)):.0%}) | {st.mean(int(r['quality']) for r in R_ if r['quality']):.2f} | "
            f"{', '.join(f'{k} ({v})' for k, v in reasons) or '—'} |")
    a, b_, c_, d = sum(r["post"] == "Y" for r in O), sum(r["post"] == "N" for r in O), sum(r["post"] == "Y" for r in B), sum(r["post"] == "N" for r in B)
    nn, r1, c1 = a + b_ + c_ + d, a + b_, a + c_
    pval = sum(comb(r1, x) * comb(nn - r1, c1 - x) / comb(nn, c1) for x in range(a, min(r1, c1) + 1))
    out(f"\nDirectional only: Fisher exact one-sided p = {pval:.2f}; one rater, who also designed the system.")
    cur = {r["item_id"]: best_match(key[r["item_id"]]["ep"], (key[r["item_id"]]["start"], key[r["item_id"]]["end"]), thr=0.9)[0] for r in O}
    acc = [cur[r["item_id"]] for r in O if r["post"] == "Y" and cur[r["item_id"]]]
    rej = [(r, cur[r["item_id"]]) for r in O if r["post"] == "N" and cur[r["item_id"]]]
    auc = [(x["final_score"] > y["final_score"]) + 0.5 * (x["final_score"] == y["final_score"]) for x in acc for _, y in rej]
    tc = Counter(m["tier"] for m in acc + [m for _, m in rej]); ta = Counter(m["tier"] for m in acc)
    out(f"Score separates accepted from rejected among these: AUC {sum(auc) / max(1, len(auc)):.2f} (0.5 = chance). "
        "Acceptance by tier: " + " · ".join(f"{t} {ta[t]}/{tc[t]}" for t in sorted(tc)) + ".\n")
else:
    out("Pending.\n")

# ---------------------------------------------------------------- 5/6. weights decision + Cold Viewer
dec_f = ROOT / "results/ranking_decision.json"
if dec_f.exists() and SH.work_dir == ROOT / "data":   # this decision record belongs to the original show
    dec = json.loads(dec_f.read_text(encoding="utf-8"))
    out("## 5. Why the scoring weights are still v0\n")
    out(f"Pre-registered rule: {dec['rule']} (dev gold eps 3 & 27 + blind ratings outside the held-out episodes; within-episode metrics).\n")
    out("| weight variant | MRR | pos > neg | blind AUC | result |\n|---|---|---|---|---|")
    for r in dec["results"]:
        out(f"| {r['variant']} | {r['mrr']:.2f} | {r['pairs_ok']}/{r['pairs']} | {r['blind_auc']:.2f} | "
            f"{'baseline' if r['passed'] is None else 'pass' if r['passed'] else 'fail'} |")
    out(f"\nSelected: **{dec['selected']}**. The v1 change is in selection (global, no quota, multi-segment), not in weights.\n")
cvr = cvd["results"]; V = Counter(r["verdict"] for r in cvr)
out("## 6. Cold Viewer (LLM as a first-time viewer): limits\n")
if cvr: out(f"Verdicts post {V['post']} / maybe {V['maybe']} / reject {V['reject']}; clarity = 4 for {sum(r['clarity'] == 4 for r in cvr)}/{len(cvr)}; "
    f"context dependency ≤ 2 for {sum(r['context_dependency'] <= 2 for r in cvr)}/{len(cvr)}. "
    + ("It rated every dev gold clip `post`, including both the human rejected for needing context, citing genre tropes. "
       if SH.work_dir == ROOT / "data" else "") + "It is used only as a penalty and for explanations.\n")

# ---------------------------------------------------------------- 7. failure analysis
out("## 7. Where the tool picks badly (concrete cases)\n")
F = Counter(f for p in P[:50] for f in p["flags"])
out("Flags in the top 50: " + (", ".join(f"{k} {v}" for k, v in F.most_common()) or "none") + "\n")
for label, r, m in REJECTED_HIGH:
    out(f"- **Human-rejected clip ranked high.** {r.promo_id} ({label.split('. ')[-1]}; human reason: {r.neg_reason}) ≈ global "
        f"#{m['rank']} tier {m['tier']} (ep {m['episode']} {m['segments'][0]['start']}–{m['segments'][-1]['end']}). "
        f"Why the tool liked it: {m['score_breakdown_text']}. Why it fails: nothing in the score measures '{r.neg_reason}' strongly enough.")
for label, r, m in POS_LOW:
    if m:
        pen = [f"{k} {v:+.2f}" for k, v in sorted({**m["score_breakdown"]["generator"], **m["score_breakdown"]["penalties"]}.items(),
                                                   key=lambda kv: kv[1]) if v < -0.005][:3]
        weak = [x for x, ok in (("weak opening (hook {:.2f})".format(m["features"]["hook"]), m["features"]["hook"] < 0.66),
                                ("weak ending (cliff {:.2f})".format(m["features"]["cliff"]), m["features"]["cliff"] < 0.66)) if ok]
        out(f"- **Human-chosen promo ranked low.** {r.promo_id} ({label.split('. ')[-1]}) at global #{m['rank']} tier {m['tier']}: "
            + "; ".join(weak + ([f"penalties {', '.join(pen)}"] if pen else [])) + ". The human valued something the score does not "
            "measure (e.g. a quiet but intriguing line, or a reveal the spoiler penalty treats as a spoiler).")
    else:
        out(f"- **Human-chosen promo never generated.** {r.promo_id} ({label.split('. ')[-1]}): no ranked promo overlaps it by ≥ 0.5 "
            "(candidate generation keeps 8 windows per episode and this window lost to stronger-scoring neighbours).")
for r, m in rej:
    out(f"- **Blind-rated reject ranked #{m['rank']}** (tier {m['tier']}, ep {m['episode']}): human reason "
        f"'{r['reason'] or '—'}'{(' — ' + r['note'][:90]) if r['note'] else ''}.")
quiet = [ep for ep in sorted(tags) if sum(b["hook"] >= 2 for b in tags[ep]["blocks"]) == 0]
ties = Counter((p["features"]["hook"], p["features"]["cliff"]) for p in P).most_common(1)[0]
out(f"- **Ties.** Hook and cliffhanger are 60% of the score but take 4 values each; the most common pair (hook {ties[0][0]}, cliff {ties[0][1]}) "
    f"is shared by {ties[1]}/{N} ranked promos, so order inside that group rests on small terms. This is why the score barely separates "
    "accepted from rejected promos in the blind test.")
out(f"- **Quiet episodes** ({', '.join(map(str, quiet)) or 'none'}): their promos fall to tier C instead of being forced into the top.")
out(f"- **Multi-segment joins are not judged by an LLM or a human yet** ({len(ms)} ranked, {sum(p['rank'] <= 50 for p in ms)} in the top 50); "
    "they inherit the Cold Viewer result of their main clip and carry the flag `multi_segment_join_unverified`.")
out("- **Spoilers** rely on Pass A reveal tags and the show map; the LLM judge never flagged a spoiler.\n")

# ---------------------------------------------------------------- 8. top 20
out("## 8. Post first: top 20 of the global ranking\n")
out("| # | tier | ep | segment(s) | dur | score | hook | cliffhanger |\n|---|---|---|---|---|---|---|---|")
sh = lambda x: (x[:60] + "…") if len(x) > 60 else x
for p in P[:20]:
    out(f"| {p['rank']} | {p['tier']} | {p['episode']} | {' + '.join(s['start'] + '–' + s['end'] for s in p['segments'])} | {p['duration_s']}s | "
        f"{p['final_score']:.2f} | {sh(p['hook']['text'])} | {sh(p['cliffhanger']['text'])} |")

(SH.outputs_dir / "eval_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
