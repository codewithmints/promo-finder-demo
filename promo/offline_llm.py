"""Offline heuristic stand-in for the LLM (LLM_PROVIDER=offline).

Lets the whole pipeline run on any show with no API key (demos, tests, air-gapped runs). It reads the same
prompts the real model gets and answers with simple text rules. Clearly lower quality than a real model;
cached under its own provider key so it never mixes with real model results.
"""
from __future__ import annotations

import json
import re

VIOLENT = re.compile(r"\b(slap|strike|struck|punch|grab|seiz|drag|shove|throw|stab|choke|kick|slam|blood|fire|gun|knife|scream)", re.I)
STRONG = re.compile(r"\b(furious|outraged|screaming|shouting|cruel|horrified|shocked|desperate|terrified|menacing|accusing|sobbing)\b", re.I)


def _pass_a(user: str) -> dict:
    blocks, speakers = [], set()
    for line in user.splitlines():
        m = re.match(r"^\[(s\d+) [^\]]+\]\s*(.*)$", line)
        if not m:
            continue
        bid, body = m.groups()
        for sm in re.finditer(r"(?:^| \| )([A-Z][A-Z0-9 .'\-]+?)(?: \([^)]*\))?: \"", body):
            speakers.add(sm.group(1).strip())
        quotes = re.findall(r'"([^"]*)"', body)
        last = quotes[-1].strip() if quotes else ""
        hook = 3 if VIOLENT.search(body) else 2 if (STRONG.search(body) or "!" in body) else 1 if quotes else 0
        cliff = 2 if last.endswith("?") else 1 if last.endswith("!") else 0
        inten = 3 if VIOLENT.search(body) else 2 if STRONG.search(body) else 1
        blocks.append({"id": bid, "hook": hook, "hook_type": "violence" if hook == 3 else "shocking_line" if hook == 2 else "none",
                       "cliff": cliff, "resolves": False, "intensity": inten, "needs_context": 1, "reveal": 0})
    if blocks:
        blocks[-1]["cliff"] = max(blocks[-1]["cliff"], 2)
    logline = re.search(r"^LOGLINE: (.*)$", user, re.M)
    return {"summary": logline.group(1) if logline else "", "central_conflict": "", "open_question": "",
            "characters": [{"name": s.title(), "labels": [s], "visual": s.lower(), "role": "unknown"} for s in sorted(speakers)],
            "key_reveals": [], "blocks": blocks}


def _show(user: str) -> dict:
    first = user.splitlines()[0].split(": ", 1)[-1] if user.strip() else ""
    return {"premise": first, "reveals": []}


def _cold(user: str) -> dict:
    lines = [l for l in user.splitlines() if l.startswith("[+")]
    last = lines[-1] if lines else ""
    first3 = " ".join(l for l in lines if re.match(r"\[\+\s*[0-2]\.\d", l))
    unresolved = last.rstrip().rstrip('"').endswith(("?", "!", "..."))
    return {"who": "unclear", "what": "a dramatic confrontation", "stakes": "unclear", "hook_moment": first3[:80] or "none",
            "open_question": "what happens next?" if unresolved else "none", "confusions": [],
            "clarity": 3, "hook_3s": 4 if VIOLENT.search(first3) or "!" in first3 else 3, "context_dependency": 2,
            "curiosity": 4 if unresolved else 3, "stakes_score": 3, "ending_tension": 4 if unresolved else 3,
            "spoiler_risk": 1, "unresolved_ending": unresolved, "would_keep_watching": unresolved,
            "verdict": "maybe", "reason": "offline heuristic judgement (no model)"}


def answer(system: str, user: str) -> str:
    if '"blocks"' in system:
        return json.dumps(_pass_a(user))
    if '"premise"' in system:
        return json.dumps(_show(user))
    if '"ending_tension"' in system:
        return json.dumps(_cold(user))
    raise ValueError("offline provider: unknown prompt type")
