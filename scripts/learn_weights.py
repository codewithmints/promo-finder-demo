"""Learn a candidate weight version from feedback, validate it, and promote it only if it passes the gate.

  python scripts/learn_weights.py --show <show_id>       # show-specific candidate from that show's feedback only
  python scripts/learn_weights.py --scope global         # cross-show candidate; needs >= 2 shows that each pass the data gate
Writes config/weights/<scope>/vN.json (status promoted | not_promoted) and reports/learning_<scope>_vN.md.
Held-out (evaluation-only) feedback is never used. No API calls.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from promo import weights as Wt, learner as Ln, feedback as Fb
from promo.show import get_show, load_registry

ap = argparse.ArgumentParser()
ap.add_argument("--show", default=None)
ap.add_argument("--scope", choices=["show", "global"], default="show")
ap.add_argument("--dry-run", action="store_true", help="report only; do not write a version")
a = ap.parse_args()


def report(scope, version, res, cur_label, patterns, extra=""):
    L = [f"# Learning report: {scope} {version}\n", f"Current weights: `{cur_label}` · status: **{'PROMOTED' if res['promote'] else 'NOT PROMOTED'}**\n"]
    if res["reasons"]:
        L.append("Why not promoted: " + "; ".join(res["reasons"]) + "\n")
    d = res["data"]
    L.append(f"Training data (held-out excluded): {d['records']} records · {d['accepted']} accepted · {d['rejected']} rejected · "
             f"{d['episodes']} episodes · {d['boundary_corrections']} boundary corrections · sources {d['sources']} · {res['pairs']} preference pairs\n")
    cv = res["cv"]
    L.append(f"Episode-grouped cross-validation, pair accuracy: current {cv['current']} → candidate {cv['candidate']}; "
             f"bootstrap win rate {res['bootstrap_win_rate']}; training-pair accuracy {res['train_pair_accuracy']}\n")
    L.append("Weight changes (candidate − current): " + (", ".join(f"{k} {v:+.4f}" for k, v in res["weight_change"].items()) or "none") + "\n")
    L.append("## Failure patterns in the feedback\n")
    for reason, info in patterns["reasons"].items():
        L.append(f"- **{reason}** ({info['count']}): largest differences from accepted promos: " +
                 ", ".join(f"{k} {v:+.3f}" for k, v in info["largest_feature_differences_vs_accepted"]))
    bc = patterns["boundary_corrections"]
    L.append(f"\nBoundary corrections: {bc['count']} segments" + (f", mean start shift {bc['mean_start_shift_s']:+.2f}s, mean end shift "
             f"{bc['mean_end_shift_s']:+.2f}s." if bc["count"] else "."))
    if extra:
        L.append("\n" + extra)
    return "\n".join(L) + "\n"


(ROOT / "reports").mkdir(exist_ok=True)
if a.scope == "show":
    SH = get_show(a.show)
    recs = Fb.load_training(SH.show_id)
    cur_w, cur_label = Wt.resolve(SH.show_id)
    res = Ln.gate(recs, cur_w)
    pats = Ln.failure_patterns(recs) if recs else {"reasons": {}, "boundary_corrections": {"count": 0, "mean_start_shift_s": None, "mean_end_shift_s": None}}
    scope = SH.show_id
else:
    reg = load_registry()
    per_show = {sid: Fb.load_training(sid) for sid in reg["shows"]}
    ok = {sid: r for sid, r in per_show.items() if Ln.data_summary(r)["accepted"] >= Ln.GATE["min_accepted"]
          and Ln.data_summary(r)["rejected"] >= Ln.GATE["min_rejected"]}
    cur_w, cur_label = Wt.resolve(None)
    allrecs = [r for rs in per_show.values() for r in rs]
    res = Ln.gate(allrecs, cur_w)
    scope = "global"
    extra = []
    if len(ok) < 2:
        res["promote"] = False
        res["reasons"].append(f"cross-show promotion needs >= 2 shows that each pass the data gate (have {len(ok)})")
    else:   # leave-one-show-out: the pooled candidate must not hurt any show
        for sid, rs in ok.items():
            others = [r for s2, r2 in ok.items() if s2 != sid for r in r2]
            w = Ln.fit(Ln.build_pairs(others), cur_w, n_records=len(others))
            p = Ln.build_pairs(rs)
            c0, c1 = Ln.pair_accuracy(p, cur_w), Ln.pair_accuracy(p, w)
            extra.append(f"- leave-out {sid}: current {c0:.3f} → trained-on-others {c1:.3f}")
            if c1 < c0:
                res["promote"] = False
                res["reasons"].append(f"pooled weights trained without {sid} are worse on {sid} ({c1:.3f} < {c0:.3f})")
    pats = Ln.failure_patterns(allrecs) if allrecs else {"reasons": {}, "boundary_corrections": {"count": 0, "mean_start_shift_s": None, "mean_end_shift_s": None}}
    res["data"]["shows"] = {sid: Ln.data_summary(r) for sid, r in per_show.items()}

version = Wt.next_version(scope if a.scope == "show" else "global")
status = "promoted" if res["promote"] else "not_promoted"
md = report(scope, version, res, cur_label, pats, "\n".join(extra) if a.scope == "global" else "")
print(md)
if not a.dry_run:
    recs_used = Fb.load_training(scope) if a.scope == "show" else allrecs
    f = Wt.save(scope, version, res["weights"], status,
                {"trained_from": {"feedback_hash": Fb.feedback_hash(recs_used), **res["data"]}, "baseline": cur_label,
                 "validation": {"cv": res["cv"], "bootstrap_win_rate": res["bootstrap_win_rate"], "gate": res["gate"],
                                "reasons_not_promoted": res["reasons"]}})
    (ROOT / "reports" / f"learning_{scope}_{version}.md").write_text(md, encoding="utf-8")
    if res["promote"]:
        Wt.promote(scope, version)
    print(f"wrote {f.relative_to(ROOT)} ({status}); active for this scope: {Wt.resolve(None if a.scope == 'global' else scope)[1]}")
