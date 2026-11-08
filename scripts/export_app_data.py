"""Export the precomputed, public-safe data bundle the Streamlit app reads: app_data/<show_id>/ (no API calls).

  python scripts/export_app_data.py [--show <show_id>]     # one show
  python scripts/export_app_data.py --all                  # every registered show
Contents (NO full screenplay excerpts, NO raw files, NO Pass A summaries, NO sanitized clip text):
  promos.json          ranked promos: segments, hook/cliffhanger lines, why, scores, breakdown, tier, flags, Cold Viewer verdict
  manifest.json        run manifest
  eval_report.md       evaluation + failure analysis
  cuts.json            per-episode legal cut times and durations (lets the public app snap reviewed boundaries)
  feedback.json        feedback summary, rejection reasons, feature-level patterns, records (features + labels, no text)
  learning.json        weights registry, version files for global + this show, learning reports
app_data/shows.json indexes the exported shows.
"""
import argparse, json, shutil, sys, time, warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.simplefilter("ignore")
from promo.show import get_show, load_registry
from promo import feedback as Fb, learner as Ln, weights as Wt

ap = argparse.ArgumentParser()
ap.add_argument("--show")
ap.add_argument("--all", action="store_true")
a = ap.parse_args()
OUT = ROOT / "app_data"
DROP = {"screenplay_excerpt"}
CV_KEEP = ("verdict", "clarity", "hook_3s", "context_dependency", "curiosity", "stakes_score", "ending_tension",
           "spoiler_risk", "open_question", "reason", "judged_clip")


def export(sid):
    SH = get_show(sid)
    pf = SH.outputs_dir / "promos_ranked.json"
    if not pf.exists():
        print(f"skip {SH.show_id}: run scripts/run_final.py --show {SH.show_id} first"); return None
    d = OUT / SH.show_id; d.mkdir(parents=True, exist_ok=True)
    promos = []
    for p in json.loads(pf.read_text(encoding="utf-8")):
        q = {k: v for k, v in p.items() if k not in DROP}
        if q.get("cold_viewer"):
            q["cold_viewer"] = {k: q["cold_viewer"][k] for k in CV_KEEP if k in q["cold_viewer"]}
        promos.append(q)
    (d / "promos.json").write_text(json.dumps(promos, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(SH.outputs_dir / "run_manifest.json", d / "manifest.json")
    if (SH.outputs_dir / "eval_report.md").exists():
        shutil.copy2(SH.outputs_dir / "eval_report.md", d / "eval_report.md")
    eps = SH.load_episodes()
    cuts = {str(e.ep): {"episode_id": SH.episode_id(e.ep), "duration": round(e.duration, 2),
                        "legal_starts": [round(x, 2) for x in e.legal_starts], "legal_ends": [round(x, 2) for x in e.legal_ends],
                        "held_out": SH.is_held_out(e.ep)} for e in eps}
    (d / "cuts.json").write_text(json.dumps(cuts), encoding="utf-8")
    allrec = list(Fb.latest(SH.show_id).values())
    train = [r for r in allrec if not r["evaluation_only"]]
    fb = {"summary": {"records": len(allrec), "training_usable": len(train), "evaluation_only": len(allrec) - len(train),
                      "accepted": sum(r["accepted"] for r in allrec), "rejected": sum(not r["accepted"] for r in allrec),
                      "by_source": dict(Counter(r["source"] for r in allrec)),
                      "reasons": dict(Counter(r["reason"] for r in allrec if not r["accepted"]))},
          "patterns": Ln.failure_patterns(train) if train else None,
          "records": [{k: r[k] for k in ("episode_number", "accepted", "quality", "reason", "source", "evaluation_only",
                                         "features", "corrected_segments", "boundary_shift_s", "candidate_id", "promo_id")}
                      for r in allrec]}
    (d / "feedback.json").write_text(json.dumps(fb), encoding="utf-8")
    versions = []
    for scope in ("global", SH.show_id):
        for f in sorted((ROOT / "config" / "weights" / scope).glob("v*.json")):
            versions.append(json.loads(f.read_text(encoding="utf-8")))
    reports = {f.name: f.read_text(encoding="utf-8") for f in sorted((ROOT / "reports").glob(f"learning_*{SH.show_id}*.md"))}
    reports.update({f.name: f.read_text(encoding="utf-8") for f in sorted((ROOT / "reports").glob("learning_global_*.md"))})
    act_w, act_label = Wt.resolve(SH.show_id)
    (d / "learning.json").write_text(json.dumps({"registry": Wt.registry(), "active_label": act_label, "active_weights": act_w,
                                                 "gate": Ln.GATE, "versions": versions, "reports": reports}), encoding="utf-8")
    print(f"exported {SH.show_id}: {len(promos)} promos, {len(cuts)} episodes, {len(allrec)} feedback records -> {d.relative_to(ROOT)}")
    return {"show_id": SH.show_id, "show_name": SH.show_name, "episodes": len(cuts), "promos": len(promos),
            "exported": time.strftime("%Y-%m-%d %H:%M:%S")}


reg = load_registry()
targets = list(reg["shows"]) if a.all else [a.show or reg["default_show"]]
idx_f = OUT / "shows.json"
idx = json.loads(idx_f.read_text(encoding="utf-8")) if idx_f.exists() else {"default_show": reg["default_show"], "shows": {}}
for sid in targets:
    r = export(sid)
    if r:
        idx["shows"][sid] = r
idx["default_show"] = reg["default_show"]
OUT.mkdir(exist_ok=True)
idx_f.write_text(json.dumps(idx, indent=1, ensure_ascii=False), encoding="utf-8")
