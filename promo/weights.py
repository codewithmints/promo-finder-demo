"""Versioned ranking weights.

config/weights/global/v0.json        general-purpose baseline, usable on ANY show (today's validated weights)
config/weights/global/vN.json        cross-show candidates (promoted only after validation on >= 2 shows)
config/weights/<show_id>/vN.json     show-specific candidates, trained only on that show's own feedback
config/weights/registry.json         {"global": "v0", "shows": {"<show_id>": "v1"}}   (active versions)

resolve(show_id): the show's own promoted version if any, else the active global version.
Feedback from one show never changes another show's weights.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .candidates import W as GEN_W

ROOT = Path(__file__).resolve().parents[1]
WDIR = ROOT / "config" / "weights"
REG = WDIR / "registry.json"
GEN_KEYS = list(GEN_W)                                   # hook, cliff, intensity, ... slow_open
EXTRA_KEYS = ["cv_reject", "cv_context", "join"]         # Cold Viewer reject, context points above 2, multi-segment join
KEYS = GEN_KEYS + EXTRA_KEYS
V0 = {**GEN_W, "cv_reject": -0.30, "cv_context": -0.10, "join": -0.05}
SIGN = {k: (1 if V0[k] > 0 else -1) for k in KEYS}       # learned weights keep these signs


def _ensure_v0():
    f = WDIR / "global" / "v0.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"version": "v0", "scope": "global", "status": "baseline", "weights": V0,
                                 "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "source": "hand-set from the assignment criteria; kept after the pre-registered comparison "
                                           "(results/ranking_decision.json)"}, indent=1), encoding="utf-8")
    if not REG.exists():
        REG.write_text(json.dumps({"global": "v0", "shows": {}}, indent=1), encoding="utf-8")


def registry() -> dict:
    _ensure_v0()
    return json.loads(REG.read_text(encoding="utf-8"))


def load(scope: str, version: str) -> dict:
    _ensure_v0()
    return json.loads((WDIR / scope / f"{version}.json").read_text(encoding="utf-8"))


def resolve(show_id: str | None) -> tuple[dict, str]:
    """(weights, label) used to rank this show."""
    reg = registry()
    if show_id and show_id in reg.get("shows", {}):
        v = reg["shows"][show_id]
        return load(show_id, v)["weights"], f"{show_id}/{v}"
    return load("global", reg["global"])["weights"], f"global/{reg['global']}"


def next_version(scope: str) -> str:
    _ensure_v0()
    d = WDIR / scope
    nums = [int(p.stem[1:]) for p in d.glob("v*.json") if p.stem[1:].isdigit()] if d.exists() else []
    return f"v{max(nums + [0]) + 1}"


def save(scope: str, version: str, weights: dict, status: str, meta: dict) -> Path:
    d = WDIR / scope
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{version}.json"
    f.write_text(json.dumps({"version": version, "scope": scope, "status": status, "weights": weights,
                             "created": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}, indent=1), encoding="utf-8")
    return f


def promote(scope: str, version: str) -> None:
    reg = registry()
    if scope == "global":
        reg["global"] = version
    else:
        reg.setdefault("shows", {})[scope] = version
    REG.write_text(json.dumps(reg, indent=1), encoding="utf-8")


def vector(features: dict, cv: dict | None, multi: bool) -> dict:
    """Feature vector the weights apply to (same for scoring and learning)."""
    x = {k: float(features[k]) for k in GEN_KEYS}
    x["cv_reject"] = 1.0 if cv and cv.get("verdict") == "reject" else 0.0
    x["cv_context"] = float(max(0, (cv or {}).get("context_dependency", 0) - 2)) if cv else 0.0
    x["join"] = 1.0 if multi else 0.0
    return x


def score(x: dict, w: dict) -> tuple[float, dict]:
    contrib = {k: round(w[k] * x[k], 4) for k in KEYS}
    return round(sum(contrib.values()), 4), contrib
