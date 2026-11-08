"""Run Pass A (LLM block tagging + show spoiler map).

  python scripts/run_pass_a.py --eps 3 27 --dry-run   # print prompt + size, no API call
  python scripts/run_pass_a.py --eps 3 27             # show map + 2 episodes
  python scripts/run_pass_a.py --all                  # all 50 (cached calls are skipped)

Outputs: data/pass_a/show.json, data/pass_a/epNN.json, data/pass_a/blocks_tagged.csv
"""
import argparse, json, sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from promo import pass_a
from promo.show import get_show, add_show_arg

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--eps", nargs="+", type=int)
g.add_argument("--all", action="store_true")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--no-show", action="store_true", help="skip the show-level spoiler map call")
add_show_arg(ap)
a = ap.parse_args()
SH = get_show(a.show)

warnings.simplefilter("ignore")
episodes = SH.load_episodes()
by_ep = {e.ep: e for e in episodes}
targets = sorted(by_ep) if a.all else a.eps
bad = [x for x in targets if x not in by_ep]
if bad:
    sys.exit(f"unknown episodes: {bad}")
out = SH.pass_a_dir
out.mkdir(parents=True, exist_ok=True)

if a.dry_run:
    e = by_ep[targets[0]]
    prev = by_ep[e.ep - 1].logline if e.ep - 1 in by_ep else ""
    up = pass_a.user_prompt_episode(e, prev)
    print(pass_a.system_a(SH.show_name), "\n", "=" * 80, "\n", up)
    chars = len(pass_a.system_a(SH.show_name)) + len(up)
    print("=" * 80)
    print(f"ep{e.ep}: {len(e.shots)} blocks · ~{chars // 4} input tokens · "
          f"{len(targets)} episode call(s){'' if a.no_show else ' + 1 show call'}")
    sys.exit(0)

from promo.llm import GeminiJSON, LLMError
llm = GeminiJSON()
print(f"model: {llm.model} · min interval {llm.min_interval}s · cache {llm.cache_dir}")
t0 = time.time()

if not a.no_show:
    try:
        show = pass_a.run_show(llm, episodes, SH.show_name)
    except LLMError as err:                       # spoiler map is optional: continue without it
        print(f"  show map FAILED (continuing without spoiler map): {str(err)[:200]}")
        show = {"model": llm.model, "prompt_version": pass_a.PROMPT_VERSION, "premise": "", "reveals": [], "failed": True}
    (out / "show.json").write_text(json.dumps(show, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"show map: {len(show['reveals'])} reveals")

failed = []
for ep in targets:
    e = by_ep[ep]
    prev = by_ep[ep - 1].logline if ep - 1 in by_ep else ""
    try:
        rec = pass_a.run_episode(llm, e, prev, SH.show_name)
        rec["show_id"], rec["episode_id"] = SH.show_id, SH.episode_id(ep)
    except LLMError as err:
        print(f"  ep{ep:02d} FAILED: {err}")
        failed.append(ep)
        continue
    (out / f"ep{ep:02d}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    b = rec["blocks"]
    print(f"  ep{ep:02d}: {len(b)} blocks · hook>=2: {sum(x['hook'] >= 2 for x in b)} · "
          f"cliff>=2: {sum(x['cliff'] >= 2 for x in b)} · resolves: {sum(x['resolves'] for x in b)}"
          f"{' · REPAIRED ' + str(len(rec['repair_notes'])) if rec['repair_notes'] else ''}")

# combined table of every tagged episode on disk
import pandas as pd
rows = []
for f in sorted(out.glob("ep*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    rows += [{"ep": r["ep"], **blk} for blk in r["blocks"]]
if rows:
    pd.DataFrame(rows).to_csv(out / "blocks_tagged.csv", index=False)

s = llm.stats
print(f"done in {time.time()-t0:.0f}s · API calls {s['calls']} · cache hits {s['cache_hits']} · "
      f"retries {s['retries']} · tokens in/out {s['in_tokens']}/{s['out_tokens']}")
if failed:
    print(f"FAILED episodes (re-run the same command to retry): {failed}")
    sys.exit(1)
