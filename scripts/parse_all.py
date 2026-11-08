"""Parse every registered episode of a show into <work>/parsed/{units.csv, shots.csv, episodes.json}."""
import argparse, json, sys, warnings
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo.parser import episode_to_records
from promo.show import get_show, add_show_arg

args = add_show_arg(argparse.ArgumentParser()).parse_args()
SH = get_show(args.show)
out = SH.parsed_dir; out.mkdir(parents=True, exist_ok=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    eps = SH.load_episodes()
U, S, meta = [], [], []
for e in eps:
    u, s = episode_to_records(e)
    U += [dict(r, show_id=SH.show_id, episode_id=SH.episode_id(e.ep)) for r in u]; S += s
    meta.append(dict(show_id=SH.show_id, episode_id=SH.episode_id(e.ep), ep=e.ep, file=Path(e.path).name, logline=e.logline,
                     continuity=e.continuity, duration=e.duration, n_scenes=len(e.scenes), n_shots=len(e.shots),
                     n_lines=len(e.lines()), legal_starts=e.legal_starts, legal_ends=e.legal_ends, warnings=e.warnings))
pd.DataFrame(U).to_csv(out / "units.csv", index=False)
pd.DataFrame(S).to_csv(out / "shots.csv", index=False)
(out / "episodes.json").write_text(json.dumps(meta, indent=1))
d = pd.DataFrame(meta)
print(f"{SH.show_name}: {len(eps)} episodes ({min(d.ep)}–{max(d.ep)}) | {len(U)} units | {len(S)} shots | {d.duration.sum()/60:.1f} min")
for m in meta:
    for w in m["warnings"]:
        print(f"  ep{m['ep']} WARN {w}")
