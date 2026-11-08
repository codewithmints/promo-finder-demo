"""Cold Viewer Test: judge one candidate clip as a stranger who has never seen the show.

Isolation rules (enforced here, the model sees ONLY the clip):
  * no show title, episode number, logline, continuity notes, summary or show map
  * no Pass A tags, candidate scores, ranks, or gold labels
  * speaker labels and character names in visual/action lines are replaced by visual descriptions
    (dialogue is left untouched: a viewer hears names that are spoken)
  * timestamps are relative to the clip start (+0.0s), so the first 3 seconds are visible as such
"""
from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, Field, field_validator

from .parser import Episode, units_in_span, EPS

PROMPT_VERSION = "CV1"
TITLE_WORDS = {"the", "lady", "lord", "sir", "madam", "mr", "mrs", "miss", "ms", "dr", "duke", "duchess", "dowager", "king",
               "queen", "prince", "princess", "father", "mother", "sister", "brother", "captain", "sergeant", "officer",
               "detective", "doctor", "old", "young", "her", "his", "grace", "elder"}   # generic English honorifics only


# ---------------------------------------------------------------- schema
def _c15(v):
    try:
        return max(1, min(5, int(round(float(v)))))
    except Exception:
        return 1


class ColdView(BaseModel):
    who: str = Field(description="who the people are to each other, as far as you can tell")
    what: str = Field(description="what is happening, one sentence")
    stakes: str = Field(description="what is at risk")
    hook_moment: str = Field(description="what in the first 3 seconds would stop you scrolling, or 'none'")
    open_question: str = Field(description="the question the clip leaves you with, or 'none'")
    confusions: list[str] = Field(default_factory=list, description="things you could not follow")
    clarity: int
    hook_3s: int
    context_dependency: int
    curiosity: int
    stakes_score: int
    ending_tension: int
    spoiler_risk: int
    unresolved_ending: bool
    would_keep_watching: bool
    verdict: str
    reason: str

    @field_validator("clarity", "hook_3s", "context_dependency", "curiosity", "stakes_score",
                     "ending_tension", "spoiler_risk", mode="before")
    @classmethod
    def _scale(cls, v):
        return _c15(v)

    @field_validator("unresolved_ending", "would_keep_watching", mode="before")
    @classmethod
    def _bool(cls, v):
        return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v):
        v = str(v).strip().lower()
        if v not in ("post", "maybe", "reject"):
            raise ValueError("verdict must be post, maybe or reject")
        return v


SYSTEM = """You are a typical American viewer aged 18-35, scrolling TikTok / Instagram Reels / YouTube Shorts.
A short clip from a drama series appears in your feed. You have NEVER seen this series and know nothing
about it. You only know what is in the clip below, written as a screenplay: VISUAL lines are what you see,
quoted lines are what you hear, SOUND lines are what you hear besides speech. Times are seconds from the
start of the clip. People are described by how they look, because you do not know their names unless
someone says a name out loud.

Judge the clip honestly and skeptically. Most clips are average: use 3 for average, and give 5 only
to something you would truly stop for and remember.

Scores (integers 1-5):
clarity: can you tell who these people are to each other and what is going on? 1 = lost, 5 = fully clear.
hook_3s: does what happens between +0.0s and +3.0s make you stop scrolling? 1 = I'd scroll past, 5 = I stop instantly.
context_dependency: how much does the clip rely on things you were not shown (references to unseen events,
  unexplained objects, people or names that mean nothing to you)? 1 = none, 5 = mostly meaningless without backstory.
curiosity: how badly do you want to know what happens next? 1 = not at all, 5 = I need to know now.
stakes_score: how much is at risk for these people, as you understand it? 1 = nothing, 5 = life, love or freedom.
ending_tension: how does the clip END? 1 = calm or resolved, 5 = cut at the peak of an unresolved moment.
spoiler_risk: does the clip answer its own central question or show how the conflict ends, so there is little
  reason to watch the episode? 1 = withholds the outcome, 5 = shows the outcome.
unresolved_ending: true if the final moment leaves a question or threat open.
would_keep_watching: true if you would tap through to watch the full episode.
verdict: "post" (a strong ad for the show), "maybe", or "reject" (weak, confusing, or gives too much away).
reason: one sentence explaining the verdict.

Return JSON only with exactly these keys:
{"who": str, "what": str, "stakes": str, "hook_moment": str, "open_question": str, "confusions": [str],
 "clarity": int, "hook_3s": int, "context_dependency": int, "curiosity": int, "stakes_score": int,
 "ending_tension": int, "spoiler_risk": int, "unresolved_ending": bool, "would_keep_watching": bool,
 "verdict": "post|maybe|reject", "reason": str}"""


# ---------------------------------------------------------------- visual descriptors
CLAUSE = re.compile(r"\s(?:who|that|standing|sitting|beaming|smiling|holding|looking|watching|gripping|"
                    r"glaring|staring|wringing|kneeling|lying|propped|clutching|leaning|wearing|carrying|actively|enforcing|gasping|bedridden|exuding|radiating|appearing)\b", re.I)
TRAIL = {"and", "with", "in", "of", "a", "an", "the", "at", "beside", "on", "by", "to"}


def _short_visual(v: str, banned: set = frozenset(), max_words: int = 8) -> str:
    v = re.sub(r"^(a|an|the)\s+", "", v.strip().rstrip("."), flags=re.I)
    v = CLAUSE.split(v)[0]
    words = []
    for w in v.split()[:max_words]:
        if re.sub(r"[^a-z]", "", w.lower().replace("'s", "")) in banned:   # never leak a name
            break
        words.append(w)
    while words and words[-1].lower().strip(",") in TRAIL:
        words.pop()
    return " ".join(words).strip(",").lower() or "person"


_LOWER: set = set()   # every word seen in lowercase in the screenplay text (set by set_vocabulary)


def set_vocabulary(texts) -> None:
    """A token counts as a proper name only if it never appears in lowercase in the screenplays
    ('Juliana' yes; 'Woman', 'Tall', 'Guard' no). Call once with all screenplay text available."""
    _LOWER.clear()
    for t in texts:
        _LOWER.update(re.findall(r"\b[a-z][a-z]+\b", t))


def _name_tokens(name: str) -> list[str]:
    if "'" in name:                                  # relationship labels like "Juliana's Father"
        return []
    return [t for t in re.findall(r"[A-Za-z]+", name)
            if t[0].isupper() and t.lower() not in TITLE_WORDS and len(t) >= 3 and t.lower() not in _LOWER]


def _banned(records) -> set:
    b = set()
    for r in records:
        for c in r.get("characters", []):
            b |= {t.lower() for t in _name_tokens(c["name"])}
            b |= {t.lower() for l in c.get("labels", []) for t in _name_tokens(l.title())}
    return b


def build_cast(pass_a_records: list[dict]) -> dict:
    """Show-level name -> visual descriptor (most frequent across episodes): a fallback for names that
    appear in action lines of an episode where that character does not speak."""
    banned, votes = _banned(pass_a_records), {}
    for r in pass_a_records:
        for c in r.get("characters", []):
            votes.setdefault(c["name"], Counter())[_short_visual(c["visual"], banned)] += 1
    return {n: cnt.most_common(1)[0][0] for n, cnt in votes.items() if cnt}


class Describer:
    def __init__(self, episode_record: dict, cast: dict | None = None):
        banned = _banned([episode_record]) | {t.lower() for n in (cast or {}) for t in _name_tokens(n)}
        self.label_vis, pats = {}, {}
        for n, vis in (cast or {}).items():               # show-level fallback first ...
            pats[n] = vis
            for t in _name_tokens(n):
                pats.setdefault(t, vis)
        for c in episode_record.get("characters", []):    # ... episode-level descriptions override
            vis = _short_visual(c["visual"], banned)
            pats[c["name"]] = vis
            for t in _name_tokens(c["name"]):
                pats[t] = vis
            for l in c["labels"]:
                self.label_vis[l.strip().upper()] = vis
                pats[l.strip().title()] = vis
        pats.update({k.upper(): v for k, v in list(pats.items()) if len(k) >= 3})   # 'JULIANA' in visuals
        self.patterns = [(re.compile(rf"\b{re.escape(k)}\b('s|'S)?"), v)
                         for k, v in sorted(pats.items(), key=lambda kv: -len(kv[0]))]

    def speaker(self, label: str) -> str:
        return self.label_vis.get(label.strip().upper(), label.strip().lower())

    def scrub(self, text: str) -> str:
        """Single pass over all names (longest first) so a replacement is never re-replaced."""
        rx = re.compile("|".join(f"(?:{p.pattern})" for p, _ in self.patterns)) if self.patterns else None
        if not rx:
            return text
        lookup = [(p, v) for p, v in self.patterns]

        def rep(m):
            for p, v in lookup:
                mm = p.fullmatch(m.group(0))
                if mm:
                    return f"the {v}" + ("'s" if mm.group(1) else "")
            return m.group(0)
        out = rx.sub(rep, text)
        return re.sub(r"\b(the|a|an)\s+the\b", "the", out, flags=re.I)


# ---------------------------------------------------------------- rendering
def render_clip(e: Episode, start: float, end: float, desc: Describer) -> str:
    units = sorted(units_in_span(e, start, end), key=lambda u: (u.t_start, u.src_line))
    out = []
    for u in units:
        t = max(0.0, u.t_start - start)
        if u.track == "shot":
            if u.text.lower().startswith("cut to black"):
                continue
            out.append(f"[+{t:4.1f}s] VISUAL: {desc.scrub(u.text)}")
        elif u.track == "action":
            out.append(f"[+{t:4.1f}s] VISUAL: {desc.scrub(u.text)}")
        elif u.track == "sfx":
            out.append(f"[+{t:4.1f}s] SOUND: {desc.scrub(u.text)}")
        else:
            cue = [desc.scrub(c) for c in u.delivery] + [x for x in u.tags if x == "O.S."]
            cue = f" ({', '.join(cue)})" if cue else ""
            out.append(f"[+{t:4.1f}s] {desc.speaker(u.speaker_raw).upper()}{cue}: {u.text}")
    return f"CLIP LENGTH: {end - start:.1f} seconds\n" + "\n".join(out)


def judge(llm, e: Episode, cand: dict, desc: Describer) -> tuple[ColdView, str]:
    clip = render_clip(e, cand["start"], cand["end"], desc)
    cv = llm.ask(SYSTEM, clip, ColdView, PROMPT_VERSION, max_tokens=2048, tag=cand["cand_id"])
    return cv, clip


def cv_score(cv: dict) -> float:
    """Fixed a priori (NOT tuned on gold): positive qualities minus context and spoiler penalties, 0-1-ish."""
    n = lambda k: (cv[k] - 1) / 4
    pos = (n("clarity") + n("hook_3s") + n("curiosity") + n("stakes_score") + n("ending_tension")) / 5
    return round(pos - 0.5 * n("context_dependency") - 0.25 * n("spoiler_risk"), 4)
