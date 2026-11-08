"""Build a self-contained HTML annotation tool for chosen episodes.

usage: python scripts/make_annotation_sheet.py 3 27 [--out annotation/annotate_ep03_ep27.html]
Output: one HTML file (no server, no dependencies). Annotations live in memory -> Download CSV.
"""
import argparse, json, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from promo.parser import fmt_time
from promo.show import get_show, add_show_arg

ap = argparse.ArgumentParser()
ap.add_argument("eps", nargs="+", type=int)
ap.add_argument("--out")
add_show_arg(ap)
a = ap.parse_args()
SH = get_show(a.show)
EPS = {e.ep: e for e in SH.load_episodes(numbers=set(a.eps))}
warnings.simplefilter("ignore")

data = []
for ep in a.eps:
    e = EPS[ep]
    LS, LE = {round(t, 2) for t in e.legal_starts}, {round(t, 2) for t in e.legal_ends}
    units = {u.unit_id: u for u in e.units}
    blocks = []
    for s in sorted(e.shots, key=lambda s: s.t_start):
        lines = [units[i] for i in s.unit_ids if units[i].track != "shot"]
        blocks.append(dict(
            id=s.shot_id, t0=s.t_start, t1=s.t_end, ts=fmt_time(s.t_start), te=fmt_time(s.t_end),
            text=s.text, synthetic=s.text.startswith("[no camera"),
            canStart=round(s.t_start, 2) in LS, canEnd=round(s.t_end, 2) in LE,
            lines=[dict(id=u.unit_id, track=u.track, t0=u.t_start, t1=u.t_end, ts=fmt_time(u.t_start),
                        te=fmt_time(u.t_end), who=u.speaker_raw, cue=", ".join(u.delivery + u.tags),
                        text=u.text, canStart=round(u.t_start, 2) in LS, canEnd=round(u.t_end, 2) in LE)
                   for u in sorted(lines, key=lambda u: (u.t_start, u.src_line))]))
    data.append(dict(ep=ep, logline=e.logline, continuity=e.continuity, duration=e.duration,
                     dur=fmt_time(e.duration), blocks=blocks))

out = Path(a.out) if a.out else ROOT / "annotation" / (f"annotate_{SH.show_id}_" + "_".join(f"ep{x:02d}" for x in a.eps) + ".html")
out.parent.mkdir(parents=True, exist_ok=True)
tpl = (ROOT / "scripts/annotation_template.html").read_text()
out.write_text(tpl.replace("/*__DATA__*/[]", json.dumps(data)))
print("wrote", out, f"({out.stat().st_size//1024} KB)")
