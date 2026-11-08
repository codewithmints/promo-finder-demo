"""Register screenplays: a completely new show, or later episodes of a registered show.

  python scripts/ingest.py "path/to/My New Show"                    # show name from SHOW:/SERIES: header or folder name
  python scripts/ingest.py path/to/folder --show-name "My New Show"  # explicit name (required for generic folder names)
  python scripts/ingest.py path/to/eps_51_100 --show <show_id>       # later episodes of an existing show
  python scripts/ingest.py --list
Then run the pipeline with --show <show_id>: parse_all, run_pass_a --all, run_candidates, run_cold_viewer, run_final, evaluate.
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.simplefilter("ignore")
from promo.show import ingest, load_registry, ShowError

ap = argparse.ArgumentParser()
ap.add_argument("folder", nargs="?")
ap.add_argument("--show-name")
ap.add_argument("--show", help="existing show_id to add episodes to")
ap.add_argument("--held-out", type=int, nargs="*", default=None, help="episode numbers reserved for evaluation only (new show)")
ap.add_argument("--default", action="store_true", help="make this the default show")
ap.add_argument("--list", action="store_true")
a = ap.parse_args()
if a.list or not a.folder:
    reg = load_registry()
    for sid, s in reg["shows"].items():
        nums = [e["episode_number"] for e in s["episodes"]]
        print(f"{'*' if sid == reg.get('default_show') else ' '} {sid}  '{s['show_name']}'  {len(nums)} episodes "
              f"({min(nums) if nums else '-'}–{max(nums) if nums else '-'})  held-out {s.get('held_out_episodes', [])}")
    sys.exit()
try:
    r = ingest(a.folder, show_name=a.show_name, show_id=a.show, held_out=a.held_out, make_default=a.default)
except ShowError as e:
    sys.exit(f"NOT INGESTED: {e}")
print(f"{'new show' if r['new_show'] else 'existing show'} {r['show_id']} ('{r['show_name']}'): added episodes {r['added']}, "
      f"unchanged {r['unchanged']}")
for f in r["show_flags"]:
    print("  FLAG:", f)
for f in r["flagged"]:
    print(f"  FLAGGED {f['file']}: {'; '.join(f['flags'])}")
