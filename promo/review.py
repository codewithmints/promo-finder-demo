"""Human-review interface used by the CLI now and by the UI later.

    ctx = ReviewContext(show)                    # loads ranked promos + per-episode data for one show
    ctx.queue(n=20)                              # top-ranked promos this reviewer has not reviewed yet
    ctx.review("G012", accepted=False, reason="needs_context", note="...", corrected_segments=[[56.2, 86.2]])
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from . import feedback as Fb
from .parser import parse_time


def parse_segments(text: str) -> list[list[float]]:
    """'00:56.20-01:26.20 + 01:40-01:55' or '56.2-86.2' -> [[56.2, 86.2], [100.0, 115.0]]"""
    segs = []
    for part in re.split(r"\s*\+\s*|\s*,\s*", text.strip()):
        a, b = re.split(r"\s*[-–]\s*(?=\d)", part, maxsplit=1)
        conv = lambda x: parse_time(x) if ":" in x else float(x)
        segs.append([conv(a), conv(b)])
    return segs


class ReviewContext:
    def __init__(self, show):
        self.show = show
        f = show.outputs_dir / "promos_ranked.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found: run scripts/run_final.py --show {show.show_id} first")
        self.promos = json.loads(f.read_text(encoding="utf-8"))
        self.by_id = {p["promo_id"]: p for p in self.promos}
        smap = show.pass_a_dir / "show.json"
        self.spoil = {r["episode"] for r in json.loads(smap.read_text(encoding="utf-8"))["reveals"] if r["kind"] != "premise"} \
            if smap.exists() else set()

    @lru_cache(maxsize=None)
    def _episode(self, ep: int):
        return self.show.load_episodes(numbers={ep})[0]

    @lru_cache(maxsize=None)
    def _tags(self, ep: int):
        return json.loads((self.show.pass_a_dir / f"ep{ep:02d}.json").read_text(encoding="utf-8"))

    def reviewed(self, reviewer: str) -> set:
        return {r["candidate_id"] for r in Fb.latest(self.show.show_id).values() if r["reviewer"] == reviewer and r["source"] == "review"}

    def queue(self, n: int = 20, reviewer: str = "reviewer") -> list[dict]:
        done = self.reviewed(reviewer)
        return [p for p in self.promos if p["candidate_id"] not in done][:n]

    def review(self, promo_id: str, *, accepted: bool, quality=None, reason: str = "", note: str = "",
               corrected_segments=None, reviewer: str = "reviewer") -> dict:
        p = self.by_id[promo_id]
        ep = p["episode"]
        rec = Fb.add_review(self.show, self._episode(ep), self._tags(ep), ep in self.spoil, p, accepted=accepted,
                            quality=quality, reason=reason, note=note, corrected_segments=corrected_segments, reviewer=reviewer)
        return rec.model_dump()
