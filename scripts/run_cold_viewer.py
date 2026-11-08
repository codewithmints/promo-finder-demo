"""Cold Viewer Test: one LLM call per candidate; the model sees ONLY the rendered clip.

  python scripts/run_cold_viewer.py --dry-run          # show one rendered clip + prompt, no API call
  python scripts/run_cold_viewer.py                    # all candidates (cached calls are skipped)
  python scripts/run_cold_viewer.py --eps 3 27         # subset
Output: data/cold_viewer/cold_viewer.json (+ .csv). Re-run the same command to resume.
"""
import argparse, csv, json, sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from promo.show import get_show, add_show_arg
from promo import cold_viewer as cvm

ap = argparse.ArgumentParser()
ap.add_argument("--eps", nargs="+", type=int)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--candidates", default=None)
ap.add_argument("--top-per-episode", type=int, default=None,
                help="judge only the K best candidates per episode by generator score (live batches)")
add_show_arg(ap)
a = ap.parse_args()
SH = get_show(a.show)
a.candidates = a.candidates or str(SH.candidates_dir / "candidates.json")
warnings.simplefilter("ignore")

eps = {e.ep: e for e in SH.load_episodes()}
cvm.set_vocabulary(u.text for e in eps.values() for u in e.units)
recs = {json.loads(f.read_text(encoding="utf-8"))["ep"]: json.loads(f.read_text(encoding="utf-8"))
        for f in sorted(SH.pass_a_dir.glob("ep*.json"))}
cast = cvm.build_cast(list(recs.values()))
cands = json.loads(Path(a.candidates).read_text(encoding="utf-8"))
if a.eps:
    cands = [c for c in cands if c["ep"] in a.eps]
if a.top_per_episode:
    by_ep = {}
    for c in sorted(cands, key=lambda c: -c["pre_score"]):
        by_ep.setdefault(c["ep"], []).append(c)
    cands = [c for v in by_ep.values() for c in v[:a.top_per_episode]]
desc = {ep: cvm.Describer(recs[ep], cast) for ep in {c["ep"] for c in cands}}

if a.dry_run:
    c = cands[0]
    clip = cvm.render_clip(eps[c["ep"]], c["start"], c["end"], desc[c["ep"]])
    print(cvm.SYSTEM, "\n" + "=" * 80 + "\n" + clip + "\n" + "=" * 80)
    print(f"{c['cand_id']}: ~{(len(cvm.SYSTEM) + len(clip)) // 4} input tokens · {len(cands)} candidates to judge")
    sys.exit(0)

from promo.llm import LLMJSON, LLMError
llm = LLMJSON()
todo = sum(not llm.cached(cvm.SYSTEM, cvm.render_clip(eps[c["ep"]], c["start"], c["end"], desc[c["ep"]]),
                          cvm.PROMPT_VERSION) for c in cands)
print(f"model {llm.name} · {len(cands)} candidates · {len(cands) - todo} cached · {todo} to call · "
      f"~{todo * (llm.min_interval + 2) / 60:.0f} min")

out_dir = SH.cold_viewer_dir; out_dir.mkdir(parents=True, exist_ok=True)
results, failed, t0 = [], [], time.time()
for i, c in enumerate(cands, 1):
    try:
        cv, clip = cvm.judge(llm, eps[c["ep"]], c, desc[c["ep"]])
    except LLMError as err:
        failed.append(c["cand_id"]); print(f"  {c['cand_id']} FAILED: {str(err)[:200]}")
        if llm.stats["consecutive_quota"] >= 5:
            print("  Repeated 429s: quota likely exhausted. Stopping; re-run later to resume from cache.")
            break
        continue
    d = cv.model_dump()
    results.append({"cand_id": c["cand_id"], "ep": c["ep"], "clip_text": clip, **d, "cv_score": cvm.cv_score(d)})
    if i % 10 == 0 or i == len(cands):
        print(f"  {i}/{len(cands)} · calls {llm.stats['calls']} · cache {llm.stats['cache_hits']} · "
              f"retries {llm.stats['retries']} · {time.time() - t0:.0f}s")

prev_f = out_dir / "cold_viewer.json"
if a.eps and prev_f.exists():           # judging a subset (e.g. new episodes): keep everyone else's results
    judged = {r["cand_id"] for r in results} | {c["cand_id"] for c in cands}
    keep = [r for r in json.loads(prev_f.read_text(encoding="utf-8"))["results"] if r["cand_id"] not in judged]
    results = keep + results
    print(f"  merged with {len(keep)} existing results for other episodes")
meta = {"model": llm.name, "prompt_version": cvm.PROMPT_VERSION, "n_candidates": len(results) + len(failed),
        "n_judged": len(results), "failed": failed, "stats": llm.stats}
(out_dir / "cold_viewer.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1,
                                                     ensure_ascii=False), encoding="utf-8")
if results:
    keys = [k for k in results[0] if k not in ("clip_text",)]
    with open(out_dir / "cold_viewer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in results:
            w.writerow({**r, "confusions": " | ".join(r["confusions"])})
s = llm.stats
print(f"done · judged {len(results)}/{len(cands)} · API calls {s['calls']} · cache hits {s['cache_hits']} · "
      f"retries {s['retries']} · tokens in/out {s['in_tokens']}/{s['out_tokens']} · {time.time() - t0:.0f}s")
if failed:
    print(f"FAILED ({len(failed)}): re-run the same command to retry them."); sys.exit(1)
