"""Final scoring + ONE GLOBAL ranking across the whole show (no API calls).

  python scripts/run_final.py
Pool = every continuous candidate (data/candidates) + selective same-episode multi-segment promos
(promo/composites.py), scored on one show-wide scale. Duplicates removed (same moment within an episode;
same scene re-opened in the next episode). No per-episode quota: an episode can place 0, 1 or many promos.

Writes outputs/:
  promos_ranked.json / .csv / .md   the global ranking (#1 = post first), every segment's timestamps,
                                    hook, cliffhanger, why it works, score + breakdown, tier
  episodes/epNN.md                  every ranked promo from that episode, with its global rank and original text
  all_candidates_scored.csv         every candidate incl. duplicates and composites (unfiltered)
  run_manifest.json                 versions, counts
"""
import csv, json, sys, time, warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import argparse
from promo.parser import fmt_time
from promo import scoring as S
from promo import weights as Wt
from promo.show import get_show, add_show_arg
from promo.composites import generate_composites, JOIN_PENALTY

warnings.simplefilter("ignore")
args = add_show_arg(argparse.ArgumentParser()).parse_args()
SH = get_show(args.show)
WEIGHTS, WEIGHTS_LABEL = Wt.resolve(SH.show_id)
eps = {e.ep: e for e in SH.load_episodes()}
tags = {json.loads(f.read_text(encoding="utf-8"))["ep"]: json.loads(f.read_text(encoding="utf-8"))
        for f in SH.pass_a_dir.glob("ep*.json")}
tags = {k: v for k, v in tags.items() if k in eps}
show = json.loads((SH.pass_a_dir / "show.json").read_text(encoding="utf-8"))
spoil = {r["episode"] for r in show["reveals"] if r["kind"] != "premise"}
cands = [c for c in json.loads((SH.candidates_dir / "candidates.json").read_text(encoding="utf-8")) if c["ep"] in eps]
for c in cands:
    c.setdefault("type", "continuous"); c["segments"] = [[c["start"], c["end"]]]
cvf = SH.cold_viewer_dir / "cold_viewer.json"
cvd = json.loads(cvf.read_text(encoding="utf-8")) if cvf.exists() else {"meta": {}, "results": []}
cv = {r["cand_id"]: {k: v for k, v in r.items() if k != "clip_text"} for r in cvd["results"]}   # sanitized clip stays internal

# ---------------------------------------------------------------- multi-segment (selective, same episode)
comps, cstats = [], Counter()
for ep in sorted(eps):
    if ep in tags:
        m, st = generate_composites(eps[ep], tags[ep], [c for c in cands if c["ep"] == ep], ep in spoil)
        comps += m; cstats.update(st)
(SH.candidates_dir / "composites.json").write_text(json.dumps(comps, indent=1), encoding="utf-8")
pool = cands + comps

# ---------------------------------------------------------------- score
for c in pool:
    cvr = cv.get(c["main_id"] if c["type"] == "multi_segment" else c["cand_id"])
    c["_cv"] = cvr
    c["x"] = Wt.vector(c["features"], cvr, c["type"] == "multi_segment")
    c["final_score"], allc = Wt.score(c["x"], WEIGHTS)
    c["contrib"] = {k: v for k, v in allc.items() if k not in ("cv_reject", "cv_context") and (k != "join" or v)}
    c["penalties"] = {"cv_reject": allc["cv_reject"], "cv_context": allc["cv_context"]}
    c["pre_score"] = round(sum(c["contrib"].values()), 4)
    c["flags"] = S.flags(c, cvr)
    h, cl = S.hook_and_cliff(eps[c["ep"]], c)
    c["_hook"], c["_cliff"], c["_hook_text"], c["_cliff_text"] = h, cl, h["text"], cl["text"]

# ---------------------------------------------------------------- dedup + ONE global ranking
pool.sort(key=lambda c: -c["final_score"])
kept = []
for c in pool:
    dup = next((k for k in kept if S.same_moment(c, k) or S.cross_episode_duplicate(c, k)), None)
    c["duplicate_of"] = dup["cand_id"] if dup else None
    if not dup:
        kept.append(c)


def breakdown_line(c):
    parts = [f"{k} {v:+.3f}" for k, v in c["contrib"].items() if abs(v) >= 0.0005]
    parts += [f"{k} {v:+.3f}" for k, v in c["penalties"].items() if v]
    return " · ".join(parts) + f" = {c['final_score']:.3f}"


promos = []
for rank, c in enumerate(kept, 1):
    e, cvr = eps[c["ep"]], c["_cv"]
    t, t_reason = S.tier(c["features"], c["_hook"]["offset_s"], c["final_score"], c["flags"])
    promos.append({
        "rank": rank, "promo_id": f"G{rank:03d}", "candidate_id": c["cand_id"], "type": c["type"],
        "show_id": SH.show_id, "episode": c["ep"], "episode_id": SH.episode_id(c["ep"]),
        "ranking_version": S.RANKING_VERSION, "weights_version": WEIGHTS_LABEL, "feature_vector": c["x"],
        "tier": t, "tier_reason": t_reason,
        "segments": [{"episode": c["ep"], "start": fmt_time(s), "end": fmt_time(e_), "start_s": s, "end_s": e_}
                     for s, e_ in S.segments_of(c)],
        "duration_s": c["duration"],
        "hook": c["_hook"], "cliffhanger": c["_cliff"],
        "why_it_works": S.why_line(c, tags[c["ep"]], c["_hook"], cvr),
        "final_score": c["final_score"], "score_breakdown_text": breakdown_line(c),
        "score_breakdown": {"generator": c["contrib"], "generator_total": c["pre_score"], "penalties": c["penalties"]},
        "features": c["features"], "flags": c["flags"], "generator_reasons": c["reasons"],
        "cold_viewer": ({**{k: cvr[k] for k in ("verdict", "clarity", "hook_3s", "context_dependency", "curiosity",
                                               "stakes_score", "ending_tension", "spoiler_risk", "who", "what", "stakes",
                                               "hook_moment", "open_question", "confusions", "reason")},
                         "judged_clip": c.get("main_id", c["cand_id"])} if cvr else None),
        "screenplay_excerpt": S.screenplay_excerpt(e, c),
    })
rank_of = {p["candidate_id"]: p["rank"] for p in promos}

out = SH.outputs_dir; (out / "episodes").mkdir(parents=True, exist_ok=True)
(out / "promos_ranked.json").write_text(json.dumps(promos, indent=1, ensure_ascii=False), encoding="utf-8")
seg_str = lambda p: " + ".join(f"{s['start']}–{s['end']}" for s in p["segments"])
spk = lambda x: (x["speaker"] + ": " if x["speaker"] else "") + x["text"]
with open(out / "promos_ranked.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["rank", "promo_id", "tier", "episode", "type", "segments", "segment_count", "duration_s", "final_score",
                "hook_time", "hook", "cliffhanger_time", "cliffhanger", "why_it_works", "score_breakdown", "tier_reason",
                "flags", "cold_viewer_verdict", "candidate_id"])
    for p in promos:
        w.writerow([p["rank"], p["promo_id"], p["tier"], p["episode"], p["type"], seg_str(p), len(p["segments"]),
                    p["duration_s"], p["final_score"], p["hook"]["time"], spk(p["hook"]), p["cliffhanger"]["time"],
                    spk(p["cliffhanger"]), p["why_it_works"], p["score_breakdown_text"], p["tier_reason"],
                    ";".join(p["flags"]), (p["cold_viewer"] or {}).get("verdict", ""), p["candidate_id"]])

TC = Counter(p["tier"] for p in promos)
md = [f"# {SH.show_name}: promos ranked across the whole show ({S.RANKING_VERSION}, weights {WEIGHTS_LABEL})\n",
      f"{len(promos)} distinct promos from {len({p['episode'] for p in promos})} episodes, one global ranking (#1 = post first). "
      f"Tier A (post first) {TC['A']} · B {TC['B']} · C (post last / needs review) {TC['C']}. "
      f"Multi-segment promos: {sum(p['type'] == 'multi_segment' for p in promos)}.\n"]
for p in promos:
    md += [f"## #{p['rank']} — Episode {p['episode']} — {seg_str(p)} ({p['duration_s']}s){' — multi-segment' if len(p['segments']) > 1 else ''} — tier {p['tier']}\n",
           f"- **Hook** ({p['hook']['time']}): {spk(p['hook'])}",
           f"- **Cliffhanger** ({p['cliffhanger']['time']}): {spk(p['cliffhanger'])}",
           f"- **Why it works:** {p['why_it_works']}",
           f"- **Score {p['final_score']:.3f}:** {p['score_breakdown_text']}",
           f"- **Tier {p['tier']}:** {p['tier_reason']}" + (f" · flags: {', '.join(p['flags'])}" if p["flags"] else ""), ""]
(out / "promos_ranked.md").write_text("\n".join(md), encoding="utf-8")

for ep in sorted(eps):
    e = eps[ep]; mine = [p for p in promos if p["episode"] == ep]
    lines = [f"# Episode {ep} — promos with their GLOBAL rank\n",
             f"Episode length {fmt_time(e.duration)} · {len(mine)} distinct promos · best global rank "
             f"{min([p['rank'] for p in mine], default='—')} · tier A: {sum(p['tier'] == 'A' for p in mine)}\n"]
    for p in mine:
        lines += [f"## Global #{p['rank']} · {seg_str(p)} · {p['duration_s']}s · score {p['final_score']:.3f} · tier {p['tier']}\n",
                  f"- **Hook** ({p['hook']['time']}, +{p['hook']['offset_s']}s): {spk(p['hook'])}",
                  f"- **Cliffhanger** ({p['cliffhanger']['time']}): {spk(p['cliffhanger'])}",
                  f"- **Why it works:** {p['why_it_works']}",
                  f"- **Score:** {p['score_breakdown_text']}",
                  f"- **Tier {p['tier']}:** {p['tier_reason']}",
                  "\n```text\n" + S.excerpt_text(p["screenplay_excerpt"]) + "\n```\n"]
    (out / "episodes" / f"ep{ep:02d}.md").write_text("\n".join(lines), encoding="utf-8")

with open(out / "all_candidates_scored.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    keys = list(cands[0]["features"])
    w.writerow(["candidate_id", "type", "episode", "segments", "duration_s", "generator_score", "cv_reject_penalty",
                "cv_context_penalty", "final_score", "global_rank", "duplicate_of", "flags", "main_id"] + [f"f_{k}" for k in keys])
    for c in pool:
        w.writerow([c["cand_id"], c["type"], c["ep"], " + ".join(f"{fmt_time(s)}–{fmt_time(t)}" for s, t in S.segments_of(c)),
                    c["duration"], c["pre_score"], c["penalties"]["cv_reject"], c["penalties"]["cv_context"], c["final_score"],
                    rank_of.get(c["cand_id"], ""), c["duplicate_of"] or "", ";".join(c["flags"]), c.get("main_id", "")]
                   + [c["features"][k] for k in keys])

per_ep = Counter(p["episode"] for p in promos if p["tier"] == "A")
manifest = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "show_id": SH.show_id, "show_name": SH.show_name,
            "episode_numbers": sorted(eps), "ranking_version": S.RANKING_VERSION, "weights_version": WEIGHTS_LABEL, "tiering_version": S.TIERING_VERSION,
            "episodes": len(eps), "continuous_candidates": len(cands), "multi_segment_candidates": len(comps),
            "multi_segment_generation": dict(cstats), "duplicates_removed": len(pool) - len(promos),
            "ranked_promos": len(promos), "tiers": dict(TC),
            "tier_a_per_episode": {"episodes_with_0": sum(1 for ep in eps if per_ep[ep] == 0),
                                   "max_in_one_episode": max(per_ep.values(), default=0)},
            "pass_a": {"model": next(iter(tags.values()))["model"], "prompt_version": next(iter(tags.values()))["prompt_version"]},
            "cold_viewer": {k: cvd["meta"].get(k) for k in ("model", "prompt_version", "n_judged", "failed")},
            "scoring": {"weights": WEIGHTS, "per_episode_quota": None},
            "spoiler_episodes": sorted(spoil)}
(out / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

print(f"pool {len(pool)} ({len(cands)} continuous + {len(comps)} multi-segment; {dict(cstats)}) · "
      f"{len(pool) - len(promos)} duplicates removed · {len(promos)} promos in ONE global ranking")
print(f"tiers {dict(TC)} · tier-A per episode: 0 in {manifest['tier_a_per_episode']['episodes_with_0']} episodes, "
      f"max {manifest['tier_a_per_episode']['max_in_one_episode']}")
print("top 15:")
for p in promos[:15]:
    print(f"  #{p['rank']:<3} ep{p['episode']:02d} {seg_str(p):<34} {p['duration_s']:>5}s {p['final_score']:.3f} {p['tier']} "
          f"{'MULTI ' if len(p['segments']) > 1 else ''}{p['hook']['text'][:45]!r} → {p['cliffhanger']['text'][:38]!r}")
