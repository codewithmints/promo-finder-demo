"""Import existing human judgements as feedback records (idempotent; re-running replaces, never duplicates).

  python scripts/import_feedback.py [--show <show_id>]
Sources (per show, from its work folder):
  blind/blind_ratings.csv + blind/blind_key.json   every rated window (ours and last-45s controls)
  gold/gold_dev_*.csv                              development gold: positives = accepted, negatives = rejected
NEVER imported: gold/gold_test_*.csv (held-out gold). Blind items on held-out episodes are stored evaluation-only.
"""
import argparse, csv, json, sys, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.simplefilter("ignore")
from promo.show import get_show, add_show_arg
from promo import feedback as Fb, weights as Wt
from promo.candidates import features_for_segments
from promo.evaluation import gold_spans

SH = get_show(add_show_arg(argparse.ArgumentParser()).parse_args().show)
eps = {e.ep: e for e in SH.load_episodes()}
tags = {json.loads(f.read_text(encoding="utf-8"))["ep"]: json.loads(f.read_text(encoding="utf-8")) for f in SH.pass_a_dir.glob("ep*.json")}
smap = SH.pass_a_dir / "show.json"
spoil = {r["episode"] for r in json.loads(smap.read_text(encoding="utf-8"))["reveals"] if r["kind"] != "premise"} if smap.exists() else set()
cands = json.loads((SH.candidates_dir / "candidates.json").read_text(encoding="utf-8")) if (SH.candidates_dir / "candidates.json").exists() else []
cvf = SH.cold_viewer_dir / "cold_viewer.json"
cv = {r["cand_id"]: r for r in json.loads(cvf.read_text(encoding="utf-8"))["results"]} if cvf.exists() else {}
REASON_MAP = {"no_cliffhanger": "no_cliffhanger", "weak_hook": "weak_hook", "needs_context": "needs_context", "spoiler": "spoiler",
              "bad_boundary": "bad_boundary", "boring": "boring", "confusing": "confusing", "duplicate": "duplicate",
              "resolved": "resolved", "slow_open": "weak_hook", "low_emotion": "low_emotion"}


def window_promo(ep, s, t, pid):
    """A promo-like dict for any window: features computed fresh; Cold Viewer signal only if it is an exact candidate."""
    c = next((c for c in cands if c["ep"] == ep and abs(c["start"] - s) < 0.05 and abs(c["end"] - t) < 0.05), None)
    r = features_for_segments(eps[ep], tags[ep], [(s, t)], ep in spoil)
    cvr = cv.get(c["cand_id"]) if c else None
    return {"promo_id": pid, "candidate_id": c["cand_id"] if c else "", "segments": [[s, t]],
            "features": r["features"], "feature_vector": Wt.vector(r["features"], cvr, False),
            "ranking_version": "imported", "weights_version": "global/v0"}


n = {"blind": 0, "blind_eval_only": 0, "dev": 0}
bk, br = SH.blind_dir / "blind_key.json", SH.blind_dir / "blind_ratings.csv"
if bk.exists() and br.exists():
    key = json.loads(bk.read_text(encoding="utf-8"))
    for r in csv.DictReader(open(br, encoding="utf-8")):
        if r.get("post") not in ("Y", "N") or r["item_id"] not in key:
            continue
        k = key[r["item_id"]]
        acc = r["post"] == "Y"
        rec = Fb.add_review(SH, eps[k["ep"]], tags[k["ep"]], k["ep"] in spoil, window_promo(k["ep"], k["start"], k["end"], k["ref"]),
                            accepted=acc, quality=int(r["quality"]) if r.get("quality") else (2 if acc else 3),
                            reason="" if acc else REASON_MAP.get(r.get("reason", ""), "other"), note=r.get("note", ""),
                            reviewer="blind_rater", source="blind_import", feedback_id=f"blind:{SH.show_id}:{r['item_id']}")
        n["blind"] += 1; n["blind_eval_only"] += rec.evaluation_only
for f in sorted(SH.gold_dir.glob("gold_dev_*.csv")):
    assert "test" not in f.name
    g = gold_spans(pd.read_csv(f), eps)
    for r in g.itertuples():
        if SH.is_held_out(r.ep):
            raise SystemExit(f"{f.name} contains held-out episode {r.ep}; refusing to import it as training data")
        acc = r.label == "pos"
        Fb.add_review(SH, eps[r.ep], tags[r.ep], r.ep in spoil, window_promo(r.ep, r.t0, r.t1, r.promo_id), accepted=acc,
                      quality=int(r.quality) if acc else 3, reason="" if acc else REASON_MAP.get(str(r.neg_reason), "other"),
                      note=str(r.note) if isinstance(r.note, str) else "", reviewer="dev_annotator", source="dev_gold_import",
                      feedback_id=f"devgold:{SH.show_id}:{r.promo_id}")
        n["dev"] += 1
print(f"{SH.show_name}: imported blind {n['blind']} (of which evaluation-only/held-out {n['blind_eval_only']}), dev gold {n['dev']}. "
      f"Held-out gold files ignored: {[p.name for p in SH.gold_dir.glob('gold_test_*.csv')]}")
L = Fb.load_training(SH.show_id)
print(f"training-usable records now: {len(L)} ({sum(r['accepted'] for r in L)} accepted / {sum(not r['accepted'] for r in L)} rejected, "
      f"{len({r['episode_id'] for r in L})} episodes)")
