"""Deterministic candidate generation.

For every legal (start, end) pair with 20 <= duration <= 60s, compute interpretable features from
Pass A block tags + cheap text heuristics, score with a transparent pre-score, then keep the top-K
non-overlapping windows per episode for the (expensive) Cold Viewer Test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .parser import Episode, fmt_time, EPS

MIN_DUR, MAX_DUR = 20.0, 60.0
HOOK_FULL, HOOK_ZERO = 3.0, 5.0      # hook delay: full credit <=3s, linear to 0 at 5s
TAIL = 4.0                            # cliffhanger window: blocks ending in the last 4s
HEAD_CTX = 10.0                       # context-dependency window at the start

# pre-score weights (transparent; recalibrated on the dev gold set later)
W = dict(hook=0.30, cliff=0.30, intensity=0.10, escalation=0.05, dur=0.05, dialogue=0.05,
         context=-0.15, spoiler=-0.10, resolution=-0.25, slow_open=-0.10)

RE_PLEA = re.compile(r"\b(please|help|let (me|her|him) go|stop|no+|don't|someone)\b", re.I)
RE_VIOLENT = re.compile(r"\b(slap|strike|strikes|struck|punch|grab|seiz|drag|shove|throw|hurl|stab|"
                        r"choke|kick|slam|rip|tear|whip|crop|blade|sword|knife|blood|fire|flame)", re.I)
STRONG_CUES = {"furious", "outraged", "screaming", "shouting", "cruel", "venomous", "horrified", "shocked",
               "desperate", "enraged", "gasping", "terrified", "menacing", "threatening", "vicious",
               "accusing", "seething", "livid", "panicked", "sobbing", "explosive", "roaring"}


@dataclass
class Candidate:
    cand_id: str
    ep: int
    start: float
    end: float
    duration: float
    start_block: str
    end_block: str
    hook_block: str
    hook_delay: float
    cliff_block: str
    features: dict = field(default_factory=dict)
    pre_score: float = 0.0
    contrib: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def asdict(self):
        d = asdict(self)
        d["start_ts"], d["end_ts"] = fmt_time(self.start), fmt_time(self.end)
        return d


# ---------------------------------------------------------------- per-block heuristics
def block_heuristics(e: Episode):
    """hook_h / cliff_h in 0-2 from text alone (a safety net for LLM misses)."""
    units = {u.unit_id: u for u in e.units}
    out = {}
    for s in e.shots:
        us = [units[i] for i in s.unit_ids]
        lines = [u for u in us if u.track == "dialogue"]
        text = " ".join(u.text for u in us) + " " + s.text
        cues = {c.lower() for u in lines for c in u.delivery}
        hook_h = 0
        if RE_VIOLENT.search(text):
            hook_h = 2
        elif cues & STRONG_CUES or any(u.text.rstrip('"” ').endswith("!") for u in lines):
            hook_h = 1
        out[s.shot_id] = {"hook_h": hook_h, "n_lines": len(lines)}
    return out


def last_line_signal(lines, end):
    """Cliff heuristic at the cut: 2 if the last line before the cut (within TAIL) is an
    unanswered question or a plea, else 0."""
    prev = [u for u in lines if u.t_end <= end + EPS and u.t_end > end - TAIL]
    if not prev:
        return 0, ""
    last = prev[-1]
    t = last.text.strip().rstrip('"”').strip()
    if t.endswith("?"):
        return 2, "ends on an unanswered question"
    if t.endswith("!") and RE_PLEA.search(t):
        return 2, "ends on a plea"
    return 0, ""


# ---------------------------------------------------------------- candidate generation
def generate(e: Episode, tags: dict, show_spoilers: set | None = None, top_k: int = 8,
             iou_max: float = 0.5) -> tuple[list[Candidate], int]:
    """tags: Pass A record for this episode. Returns (top-K candidates, number of windows scored)."""
    blocks = sorted(e.shots, key=lambda s: s.t_start)
    T = {b["block_id"]: b for b in tags["blocks"]}
    H = block_heuristics(e)
    lines = sorted([u for u in e.units if u.track == "dialogue"], key=lambda u: u.t_start)
    content = sorted([u for u in e.units if u.track in ("dialogue", "action")], key=lambda u: u.t_start)
    last_block = blocks[-1].shot_id if not blocks[-1].text.lower().startswith("cut to black") else blocks[-2].shot_id
    spoiler_ep = bool(show_spoilers and e.ep in show_spoilers)

    def hook_of(b):
        return max(T[b.shot_id]["hook"], H[b.shot_id]["hook_h"])

    def cliff_of(b):
        c = T[b.shot_id]["cliff"]
        if b.shot_id == last_block:          # authored episode cliffhanger
            c = min(3, c + 1)
        return c

    scored = []
    for s in e.legal_starts:
        for t in e.legal_ends:
            dur = t - s
            if dur < MIN_DUR - EPS or dur > MAX_DUR + EPS:
                continue
            span = [b for b in blocks if b.t_end > s + EPS and b.t_start < t - EPS]
            if not span:
                continue
            # hook: best block whose content starts within 5s, decayed by delay
            best_h, hb, hd = 0.0, span[0], 0.0
            for b in span:
                d = max(0.0, b.t_start - s)
                if d > HOOK_ZERO:
                    break
                decay = 1.0 if d <= HOOK_FULL else (HOOK_ZERO - d) / (HOOK_ZERO - HOOK_FULL)
                v = hook_of(b) / 3 * decay
                if v > best_h:
                    best_h, hb, hd = v, b, d
            # cliff: best block ending in the last 4s, or last-line heuristic
            tail = [b for b in span if b.t_end > t - TAIL]
            cb = max(tail, key=lambda b: (cliff_of(b), b.t_end))
            c_llm = cliff_of(cb)
            c_h, c_reason = last_line_signal(lines, t)
            cliff = max(c_llm, c_h) / 3
            resolves = any(T[b.shot_id]["resolves"] for b in tail)
            # context, intensity, escalation, spoiler
            def wmean(key, bs):
                tot = sum(min(b.t_end, t) - max(b.t_start, s) for b in bs) or 1
                return sum(T[b.shot_id][key] * (min(b.t_end, t) - max(b.t_start, s)) for b in bs) / tot
            head = [b for b in span if b.t_start < s + HEAD_CTX]
            ctx = wmean("needs_context", head) / 3
            inten = wmean("intensity", span) / 3
            third = dur / 3
            first = [b for b in span if b.t_start < s + third] or span[:1]
            last = [b for b in span if b.t_end > t - third] or span[-1:]
            esc = max(0.0, (wmean("intensity", last) - wmean("intensity", first)) / 3)
            rev = max(T[b.shot_id]["reveal"] for b in span) / 3
            spoiler = rev * (1.0 if spoiler_ep else 0.5)
            first_content = next((u.t_start for u in content if u.t_start >= s - EPS), t)
            slow = min(1.0, max(0.0, first_content - s - 1.5) / 3)
            dur_fit = 1.0 if 25 <= dur <= 45 else max(0.0, 1 - (min(abs(dur - 25), abs(dur - 45)) / 15))
            n_lines = sum(1 for u in lines if s - EPS <= u.t_start < t)
            dialogue = min(1.0, n_lines / 6)

            f = dict(hook=round(best_h, 3), cliff=round(cliff, 3), intensity=round(inten, 3),
                     escalation=round(esc, 3), dur=round(dur_fit, 3), dialogue=round(dialogue, 3),
                     context=round(ctx, 3), spoiler=round(spoiler, 3), resolution=float(resolves),
                     slow_open=round(slow, 3))
            contrib = {k: round(W[k] * v, 4) for k, v in f.items()}
            score = sum(contrib.values())
            scored.append((score, s, t, hb, hd, cb, f, contrib, c_reason, c_llm, c_h))

    scored.sort(key=lambda x: -x[0])
    kept: list[Candidate] = []
    for score, s, t, hb, hd, cb, f, contrib, c_reason, c_llm, c_h in scored:
        if any(iou((s, t), (k.start, k.end)) > iou_max for k in kept):
            continue
        reasons = []
        if f["hook"] >= 0.66:
            reasons.append(f"strong opening ({T[hb.shot_id]['hook_type']}) {hd:.1f}s in")
        if c_h and c_h >= c_llm:
            reasons.append(c_reason)
        elif f["cliff"] >= 0.66:
            reasons.append("ends on an unresolved beat" + (" (episode cliffhanger)" if cb.shot_id == last_block else ""))
        if f["resolution"]:
            reasons.append("WARNING: tension resolves near the end")
        if f["context"] >= 0.5:
            reasons.append("WARNING: needs earlier context")
        if f["spoiler"] >= 0.5:
            reasons.append("WARNING: reveals major story information")
        kept.append(Candidate(f"c{e.ep:02d}_{len(kept)+1:02d}", e.ep, s, t, round(t - s, 2),
                              _block_at(blocks, s), _block_at(blocks, t - EPS), hb.shot_id, round(hd, 2),
                              cb.shot_id, f, round(score, 4), contrib, reasons))
        if len(kept) >= top_k:
            break
    return kept, len(scored)


def _block_at(blocks, t):
    return next((b.shot_id for b in blocks if b.t_start - EPS <= t < b.t_end), blocks[-1].shot_id)


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- features for any list of segments
def features_for_segments(e: Episode, tags: dict, segs: list[tuple[float, float]], spoiler_ep: bool,
                          _cache: dict | None = None) -> dict:
    """Same feature definitions as generate(), on a virtual timeline that concatenates the segments
    (in order). For a single segment this reproduces generate() exactly (tested)."""
    blocks = sorted(e.shots, key=lambda s: s.t_start)
    T = {b["block_id"]: b for b in tags["blocks"]}
    H = block_heuristics(e)
    lines = sorted([u for u in e.units if u.track == "dialogue"], key=lambda u: u.t_start)
    content = sorted([u for u in e.units if u.track in ("dialogue", "action")], key=lambda u: u.t_start)
    last_block = blocks[-1].shot_id if not blocks[-1].text.lower().startswith("cut to black") else blocks[-2].shot_id
    hook_of = lambda b: max(T[b.shot_id]["hook"], H[b.shot_id]["hook_h"])
    cliff_of = lambda b: min(3, T[b.shot_id]["cliff"] + 1) if b.shot_id == last_block else T[b.shot_id]["cliff"]

    pieces, off = [], 0.0          # (block, overlap seconds, virtual start, virtual end, segment index)
    for i, (s, t) in enumerate(segs):
        for b in blocks:
            if b.t_end > s + EPS and b.t_start < t - EPS:
                pieces.append((b, min(b.t_end, t) - max(b.t_start, s), off + max(0.0, b.t_start - s),
                               off + (min(b.t_end, t) - s), i))
        off += t - s
    total = off
    # hook: best block whose content starts within 5s (virtual), decayed by delay
    best_h, hb, hd = 0.0, pieces[0][0], 0.0
    for b, _, vs, _, _ in pieces:
        if vs > HOOK_ZERO:
            break
        decay = 1.0 if vs <= HOOK_FULL else (HOOK_ZERO - vs) / (HOOK_ZERO - HOOK_FULL)
        v = hook_of(b) / 3 * decay
        if v > best_h:
            best_h, hb, hd = v, b, vs
    s_last, t_last = segs[-1]
    tail = [p[0] for p in pieces if p[4] == len(segs) - 1 and p[0].t_end > t_last - TAIL]
    cb = max(tail, key=lambda b: (cliff_of(b), b.t_end))
    c_llm = cliff_of(cb)
    c_h, c_reason = last_line_signal(lines, t_last)
    cliff = max(c_llm, c_h) / 3
    resolves = any(T[b.shot_id]["resolves"] for b in tail)

    def wmean(key, ps):
        tot = sum(p[1] for p in ps) or 1
        return sum(T[p[0].shot_id][key] * p[1] for p in ps) / tot
    head = [p for p in pieces if p[2] < HEAD_CTX]
    third = total / 3
    first = [p for p in pieces if p[2] < third] or pieces[:1]
    last = [p for p in pieces if p[3] > total - third] or pieces[-1:]
    rev = max(T[p[0].shot_id]["reveal"] for p in pieces) / 3
    # first content line, in virtual time
    fc, off = total, 0.0
    for s, t in segs:
        u = next((u for u in content if s - EPS <= u.t_start < t), None)
        if u:
            fc = off + (u.t_start - s); break
        off += t - s
    dur_fit = 1.0 if 25 <= total <= 45 else max(0.0, 1 - (min(abs(total - 25), abs(total - 45)) / 15))
    n_lines = sum(1 for s, t in segs for u in lines if s - EPS <= u.t_start < t)
    f = dict(hook=round(best_h, 3), cliff=round(cliff, 3), intensity=round(wmean("intensity", pieces) / 3, 3),
             escalation=round(max(0.0, (wmean("intensity", last) - wmean("intensity", first)) / 3), 3),
             dur=round(dur_fit, 3), dialogue=round(min(1.0, n_lines / 6), 3),
             context=round(wmean("needs_context", head) / 3, 3),
             spoiler=round(rev * (1.0 if spoiler_ep else 0.5), 3), resolution=float(resolves),
             slow_open=round(min(1.0, max(0.0, fc - 1.5) / 3), 3))
    contrib = {k: round(W[k] * v, 4) for k, v in f.items()}
    return {"features": f, "contrib": contrib, "pre_score": round(sum(contrib.values()), 4),
            "hook_block": hb.shot_id, "hook_delay": round(hd, 2), "cliff_block": cb.shot_id,
            "cliff_reason": c_reason, "cliff_llm": c_llm, "cliff_h": c_h, "last_block": last_block}
