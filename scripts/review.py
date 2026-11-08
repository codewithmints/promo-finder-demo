"""Command-line human review (the UI will call the same promo/review.py functions).

  python scripts/review.py --list 10                                   # next unreviewed promos, best first
  python scripts/review.py --promo G012 --accept --quality 1
  python scripts/review.py --promo G013 --reject --reason needs_context --note "who is the maid?"
  python scripts/review.py --promo G014 --accept --quality 2 --segments "01:14.80-01:44.00"   # corrected boundaries
  python scripts/review.py --summary
Writes to data/feedback/feedback.jsonl. Add --show <show_id> for a non-default show.
"""
import argparse, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.simplefilter("ignore")
from promo.show import get_show, add_show_arg
from promo.review import ReviewContext, parse_segments
from promo import feedback as Fb

ap = add_show_arg(argparse.ArgumentParser())
ap.add_argument("--list", type=int, nargs="?", const=10)
ap.add_argument("--summary", action="store_true")
ap.add_argument("--promo")
g = ap.add_mutually_exclusive_group()
g.add_argument("--accept", action="store_true")
g.add_argument("--reject", action="store_true")
ap.add_argument("--quality", type=int, choices=[1, 2, 3])
ap.add_argument("--reason", choices=Fb.REASONS)
ap.add_argument("--note", default="")
ap.add_argument("--segments", help="corrected boundaries, e.g. '00:56.20-01:26.20' or '01:08.1-01:12.1 + 01:50.8-02:15.5'")
ap.add_argument("--reviewer", default="reviewer")
a = ap.parse_args()
SH = get_show(a.show)
ctx = ReviewContext(SH)

if a.list:
    for p in ctx.queue(a.list, a.reviewer):
        segs = " + ".join(f"{s['start']}–{s['end']}" for s in p["segments"])
        print(f"{p['promo_id']}  #{p['rank']:<4} ep{p['episode']:<3} {segs:<34} tier {p['tier']} score {p['final_score']:.3f}\n"
              f"      hook: {p['hook']['text'][:80]}\n      cliff: {p['cliffhanger']['text'][:80]}")
elif a.summary:
    L = list(Fb.latest(SH.show_id).values())
    print(f"{SH.show_name}: {len(L)} feedback records · training-usable {sum(not r['evaluation_only'] for r in L)} · "
          f"evaluation-only (held-out) {sum(r['evaluation_only'] for r in L)}")
    print("by source:", dict(Counter(r["source"] for r in L)), "· accepted", sum(r["accepted"] for r in L),
          "· rejected", sum(not r["accepted"] for r in L), "· reasons", dict(Counter(r["reason"] for r in L if not r["accepted"])))
elif a.promo:
    if not (a.accept or a.reject):
        sys.exit("say --accept or --reject")
    try:
        r = ctx.review(a.promo, accepted=a.accept, quality=a.quality, reason=a.reason or "", note=a.note,
                       corrected_segments=parse_segments(a.segments) if a.segments else None, reviewer=a.reviewer)
    except Fb.ReviewError as e:
        sys.exit(f"not saved: {e}")
    print(f"saved {r['feedback_id']} · {'accepted' if r['accepted'] else 'rejected'}"
          f"{' · corrected ' + str(r['corrected_segments']) if r['corrected_segments'] else ''}"
          f"{' · EVALUATION-ONLY (held-out episode)' if r['evaluation_only'] else ''}")
else:
    ap.print_help()
