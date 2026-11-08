"""Pass A: one LLM call per episode that tags every timeline block with promo signals,
plus one show-level call that separates the premise (safe) from spoilers.

The LLM does NOT pick promos. It only produces signals for the deterministic candidate generator.
"""
from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .parser import Episode, fmt_time

PROMPT_VERSION = "A3"  # A3: show-neutral premise example (A2 tags of the original 50 episodes remain valid outputs)

HOOK_TYPES = ["violence", "shocking_line", "reveal", "accusation", "threat", "humiliation",
              "betrayal", "romance", "rescue", "ultimatum", "breakdown", "mystery", "none"]


# ---------------------------------------------------------------- schemas
def _clamp03(v):
    try:
        return max(0, min(3, int(round(float(v)))))
    except Exception:
        return 0


class BlockTag(BaseModel):
    id: str
    hook: int = Field(description="0-3 stopping power if a clip OPENED here")
    hook_type: str = "none"
    cliff: int = Field(description="0-3 how strongly a clip ENDING here leaves the viewer needing more")
    resolves: bool = Field(description="true if this block answers/defuses tension raised earlier")
    intensity: int = Field(description="0-3 emotional/dramatic intensity")
    needs_context: int = Field(description="0-3 how much earlier-episode knowledge is needed to follow it")
    reveal: int = Field(description="0-3 how much important story information this gives away")

    @field_validator("hook", "cliff", "intensity", "needs_context", "reveal", mode="before")
    @classmethod
    def _scale(cls, v):
        return _clamp03(v)

    @field_validator("resolves", mode="before")
    @classmethod
    def _bool(cls, v):
        return str(v).strip().lower() in ("true", "1", "yes") if not isinstance(v, bool) else v

    @field_validator("hook_type", mode="before")
    @classmethod
    def _ht(cls, v):
        v = str(v or "none").strip().lower().replace(" ", "_")
        return v if v in HOOK_TYPES else "none"


class Character(BaseModel):
    name: str = Field(description="canonical first name or title")
    labels: list[str] = Field(description="every SPEAKER label in this episode that is this person")
    visual: str = Field(description="what a viewer sees, no name, e.g. 'young woman in a white wedding gown'")
    role: str = Field(description="role in one short phrase, e.g. 'heroine', 'cruel stepsister'")


class KeyReveal(BaseModel):
    block: str
    what: str
    level: Literal["premise", "minor", "major"]


class EpisodeAnalysis(BaseModel):
    summary: str
    central_conflict: str
    open_question: str = Field(description="the unanswered question the episode ends on")
    characters: list[Character]
    key_reveals: list[KeyReveal] = []
    blocks: list[BlockTag]


class ShowReveal(BaseModel):
    episode: int
    reveal: str
    kind: Literal["premise", "twist", "resolution"]


class ShowMap(BaseModel):
    premise: str = Field(description="what any viewer may know: the setup implied by the title and episodes 1-3")
    reveals: list[ShowReveal]


# ---------------------------------------------------------------- rendering
def short(block_id: str) -> str:
    return block_id.split("_")[1]  # e03_s012 -> s012


def render_episode(e: Episode) -> str:
    units = {u.unit_id: u for u in e.units}
    out = []
    for s in sorted(e.shots, key=lambda s: s.t_start):
        parts = [] if s.text.startswith("[no camera") else [f"CAMERA: {s.text}"]
        for uid in s.unit_ids:
            u = units[uid]
            if u.track == "shot":
                continue
            if u.track == "dialogue":
                cue = f" ({', '.join(u.delivery + u.tags)})" if (u.delivery or u.tags) else ""
                parts.append(f"{u.speaker_raw}{cue}: {u.text}")
            else:
                parts.append(f"{u.track.upper()}: {u.text}")
        out.append(f"[{short(s.shot_id)} {fmt_time(s.t_start)}-{fmt_time(s.t_end)}] " + " | ".join(parts))
    return "\n".join(out)


def system_a(SHOW_TITLE: str) -> str:
    return f"""You are a senior promo editor for vertical microdramas (ReelShort/DramaBox style).
You annotate screenplays so a program can cut 20-60 second promo clips for TikTok/Instagram/YouTube.
A good promo OPENS on a hook (stops a scrolling stranger within 3 seconds), ENDS on an unresolved
cliffhanger, makes sense to someone who has never seen the show, and does not give away payoffs.
The show is "{SHOW_TITLE}". You do NOT choose clips. You score every block honestly.

Scales (integers):
hook 0-3: if a clip STARTED at this block, would a stranger stop scrolling within 3 seconds?
  0 = establishing/transition/reaction shot, small talk; 1 = some tension but ordinary;
  2 = a line or action that is striking ON ITS OWN; 3 = slap/violence, shocking line, public humiliation,
  betrayal, reveal, ultimatum. Judge the block itself, not what comes after it.
  CALIBRATION: in a typical episode only about 1 block in 6 deserves hook >= 2, and only 2-4 blocks deserve 3.
  Continuing an argument that is already running is usually 1, not 2.
cliff 0-3: imagine the clip is CUT right after this block. Judge LOCALLY, as if this were the last thing
  the viewer sees. It does not matter that the episode continues; mid-episode moments can be strong cliffs.
  0 = a settled or answered moment; 1 = mild open thread;
  2 = an unanswered direct question, a threat or accusation just made, a plea for help, someone being
      seized/dragged/attacked, a decision demanded but not yet given, a secret about to be spoken;
  3 = peak unresolved moment (harm about to happen, life-changing answer pending, reveal about to land).
  A question that is answered in this same block is not a cliff; the answer is.
resolves: true if this block answers or defuses tension raised earlier (rescue completes, question answered,
  conflict ends, villain leaves). A clip should not end on a resolving block.
intensity 0-3: emotional/dramatic temperature.
needs_context 0-3: how much knowledge from EARLIER episodes is needed to follow this block
  (0 = self-explanatory, 3 = meaningless without backstory, e.g. references to unseen events/objects).
reveal 0-3: how much important story information this block gives away (3 = a major twist or outcome).
hook_type: one of {", ".join(HOOK_TYPES)}.

Characters: map EVERY speaker label to one person (labels like "THE FORMER GROOM" or "LADY IN GOLD GOWN"
are aliases of named characters; infer from the logline and context). "visual" must describe what a viewer
sees without names. key_reveals.level: premise = setup implied by the show title (e.g. a hidden identity the
title already gives away), minor = small new information, major = twist/outcome a promo should not spoil.

Return JSON only, exactly this shape:
{{"summary": str, "central_conflict": str, "open_question": str,
 "characters": [{{"name": str, "labels": [str], "visual": str, "role": str}}],
 "key_reveals": [{{"block": "s012", "what": str, "level": "premise|minor|major"}}],
 "blocks": [{{"id": "s000", "hook": 0, "hook_type": "none", "cliff": 0, "resolves": false,
             "intensity": 0, "needs_context": 0, "reveal": 0}}, ...one entry for EVERY block, in order]}}"""


def user_prompt_episode(e: Episode, prev_logline: str = "") -> str:
    prev = f"PREVIOUS EPISODE LOGLINE: {prev_logline}\n" if prev_logline else ""
    return (f"EPISODE {e.ep} ({fmt_time(e.duration)})\n{prev}"
            f"LOGLINE: {e.logline}\nCONTINUITY NOTES: {e.continuity}\n\n"
            f"BLOCKS ({len(e.shots)}). Each line: [block_id start-end] contents.\n{render_episode(e)}")


def system_show(SHOW_TITLE: str) -> str:
    return f"""You are a story analyst for the microdrama "{SHOW_TITLE}".
From the episode loglines, separate what a promo may freely show from what it must not spoil.
premise = the setup a new viewer may know (everything implied by the title, plus episodes 1-3 setup).
reveals: every important twist or resolution with the episode where it happens.
kind: premise (title-level setup, safe), twist (surprise a promo should not give away),
resolution (a conflict is settled, e.g. villain defeated, rescue succeeds).
Return JSON only: {{"premise": str, "reveals": [{{"episode": int, "reveal": str, "kind": "premise|twist|resolution"}}]}}"""


def user_prompt_show(episodes: list[Episode]) -> str:
    return "\n".join(f"EP {e.ep}: {e.logline}" for e in episodes)


# ---------------------------------------------------------------- validation / repair
def make_validator(e: Episode):
    want = [short(s.shot_id) for s in sorted(e.shots, key=lambda s: s.t_start)]
    speakers = {u.speaker_raw for u in e.units if u.track == "dialogue"}

    def check(a: EpisodeAnalysis):
        got = [b.id for b in a.blocks]
        problems = []
        missing = [w for w in want if w not in set(got)]
        if missing:
            problems.append(f"missing block ids: {', '.join(missing)}")
        extra = [g for g in got if g not in set(want)]
        if extra:
            problems.append(f"unknown block ids: {', '.join(extra)}")
        labelled = {l.strip().upper() for c in a.characters for l in c.labels}
        unmapped = sorted(s for s in speakers if s.upper() not in labelled)
        if unmapped:
            problems.append(f"speaker labels not mapped to a character: {', '.join(unmapped)}")
        return problems
    return check


def repair(e: Episode, a: EpisodeAnalysis) -> tuple[EpisodeAnalysis, list[str]]:
    """Deterministic last resort after retries: one tag per block in order, every speaker mapped."""
    notes = []
    by_id = {}
    for b in a.blocks:
        by_id.setdefault(b.id, b)
    blocks = []
    for s in sorted(e.shots, key=lambda s: s.t_start):
        sid = short(s.shot_id)
        if sid not in by_id:
            notes.append(f"filled missing {sid} with zeros")
        blocks.append(by_id.get(sid) or BlockTag(id=sid, hook=0, cliff=0, resolves=False,
                                                intensity=0, needs_context=0, reveal=0))
    labelled = {l.strip().upper() for c in a.characters for l in c.labels}
    chars = list(a.characters)
    for spk in sorted({u.speaker_raw for u in e.units if u.track == "dialogue"}):
        if spk.upper() not in labelled:
            chars.append(Character(name=spk.title(), labels=[spk], visual=spk.lower(), role="unknown"))
            notes.append(f"unmapped speaker {spk} kept as-is")
    return a.model_copy(update={"blocks": blocks, "characters": chars}), notes


# ---------------------------------------------------------------- output
def episode_record(e: Episode, a: EpisodeAnalysis, notes: list[str], model: str) -> dict:
    shots = sorted(e.shots, key=lambda s: s.t_start)
    alias = {l.strip().upper(): c.name for c in a.characters for l in c.labels}
    blocks = []
    for s, b in zip(shots, a.blocks):
        blocks.append({"block_id": s.shot_id, "t_start": s.t_start, "t_end": s.t_end,
                       **b.model_dump(exclude={"id"})})
    return {"ep": e.ep, "model": model, "prompt_version": PROMPT_VERSION,
            "summary": a.summary, "central_conflict": a.central_conflict, "open_question": a.open_question,
            "characters": [c.model_dump() for c in a.characters], "alias": alias,
            "key_reveals": [{**r.model_dump(), "block": f"e{e.ep:02d}_{r.block}"} for r in a.key_reveals],
            "repair_notes": notes, "blocks": blocks}


def run_episode(llm, e: Episode, prev_logline: str = "", show_name: str = "") -> dict:
    check = make_validator(e)
    a = llm.ask(system_a(show_name), user_prompt_episode(e, prev_logline), EpisodeAnalysis, PROMPT_VERSION,
                max_tokens=12000, validate=check, tag=f"ep{e.ep:02d}")
    a, notes = repair(e, a)
    return episode_record(e, a, notes, llm.model)


def run_show(llm, episodes: list[Episode], show_name: str = "") -> dict:
    m = llm.ask(system_show(show_name), user_prompt_show(episodes), ShowMap, PROMPT_VERSION, max_tokens=6000, tag="show")
    return {"model": llm.model, "prompt_version": PROMPT_VERSION, **m.model_dump()}
