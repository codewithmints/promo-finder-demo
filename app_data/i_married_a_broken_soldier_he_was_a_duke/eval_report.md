# I Married a Broken Soldier. He was a Duke — evaluation of the global ranking

Evaluation run 2026-11-08 12:25 · outputs 2026-11-08 12:25:30 · ranking `v1_global` (weights `global/v0`, tiers `tiers_v1`)

- 50 episodes → 391 continuous candidates + 31 multi-segment → 126 duplicates removed → **296 promos in one global ranking** (no per-episode quota).
- Tiers: A (post first) 149 · B 48 · C (post last / needs review) 99. Episodes with no tier-A promo: 4; most tier-A promos from one episode: 7.
- Pass A `gemini-3.5-flash-lite` (A2); Cold Viewer `gemini:gemini-3.5-flash-lite` (CV1), 391 judged, 415 calls, 0 failed.
- Score = generator features × v0 weights − Cold Viewer penalties (reject −0.30; context ≥ 3: −0.10/pt) − 0.05 per multi-segment join. Nothing tuned on the held-out episodes.

## 1. Hard constraints (all ranked promos)

| check | pass |
|---|---|
| total duration 20–60s | 296/296 |
| every segment starts and ends on a legal cut (never mid-line / mid-sentence) | 296/296 |
| segments in chronological order, no overlap | 296/296 |
| hook within 3s | 272/296 |
| no resolving beat in the last 4s | 293/296 |
| excerpt = original screenplay lines with original timestamps | 296/296 (rebuilt from source lines; tested) |

Multi-segment promos ranked: 7 (built 31 from 46 eligible weak-opening/strong-cliffhanger mains out of 391); in the top 50: 1; tier A: 6.

## 2. Development gold: `gold_dev_ep03_ep27.csv` — 6 positives, 4 negatives, episodes [3, 27]

Each gold promo is matched to the ranked promo with the highest time overlap (IoU ≥ 0.5). Ranks are GLOBAL (1–296). Because near-duplicates (same opening or same cliffhanger moment) are merged, a gold clip can map to the surviving promo for that moment, and two gold clips can map to the same promo.

| metric | ours (global ranking) | same promos, random order |
|---|---|---|
| gold positives found among ranked promos | 6/6 | 6/6 |
| median global percentile of gold positives (lower = better) | 25% | 50% |
| gold positives in global top 50 | 2/6 | 1.0/6 |
| gold positives in tier A | 3/6 | — |
| positive ranked above negative (all pairs, across episodes) | 10/12 | 44% |
| gold negatives in tier A | 0/4 | — |

Secondary (within its own episode): gold positive is the episode's best-ranked promo 1/6, in its top 3 3/6; the last-45s-of-episode window matches 2/6 gold positives.

Per gold promo → global rank: g03_01 (pos) #38 A, g03_02 (pos) #87 A, g03_03 (pos) #61 A, g03_04 (neg) not found, g03_05 (neg) not found, g27_01 (pos) #168 B, g27_02 (pos) #15 B, g27_03 (pos) #200 C, g27_04 (neg) #168 B, g27_05 (neg) #222 C

## 3. Held-out test gold (evaluation only): `gold_test_ep01_ep12_ep34_ep45.csv` — 10 positives, 4 negatives, episodes [1, 12, 34, 45]

Each gold promo is matched to the ranked promo with the highest time overlap (IoU ≥ 0.5). Ranks are GLOBAL (1–296). Because near-duplicates (same opening or same cliffhanger moment) are merged, a gold clip can map to the surviving promo for that moment, and two gold clips can map to the same promo.

| metric | ours (global ranking) | same promos, random order |
|---|---|---|
| gold positives found among ranked promos | 9/10 | 9/10 |
| median global percentile of gold positives (lower = better) | 25% | 50% |
| gold positives in global top 50 | 4/10 | 1.6/10 |
| gold positives in tier A | 8/10 | — |
| positive ranked above negative (all pairs, across episodes) | 14/27 | 49% |
| gold negatives in tier A | 2/4 | — |

Secondary (within its own episode): gold positive is the episode's best-ranked promo 3/10, in its top 3 6/10; the last-45s-of-episode window matches 3/10 gold positives.

Per gold promo → global rank: g01_01 (pos) #137 A, g01_02 (pos) #46 A, g01_03 (pos) #110 A, g01_04 (neg) #8 A, g12_01 (pos) #12 A, g12_02 (neg) #149 A, g34_01 (pos) #73 A, g34_02 (pos) #130 A, g34_03 (pos) #37 A, g34_04 (neg) #126 B, g45_01 (pos) #9 A, g45_02 (pos) #261 C, g45_03 (pos) not found, g45_04 (neg) not found

### Dev + held-out combined

positives found 15/16 · in global top 50 6/16 · in tier A 11/16 · positive above negative 24/39 · gold negatives in tier A 2

## 4. Blind human rating

Rated on the v0 top 30 plus 10 last-45s controls, shuffled, source hidden. Each rated item is matched to its current global rank.

| source | rated | accepted | mean quality (1 best) | top rejection reasons |
|---|---|---|---|---|
| ours | 30 | 22/30 (73%) | 1.73 | needs_context (6), spoiler (1), boring (1) |
| last-45s control | 10 | 6/10 (60%) | 2.00 | no_cliffhanger (2), needs_context (2) |

Directional only: Fisher exact one-sided p = 0.34; one rater, who also designed the system.
Score separates accepted from rejected among these: AUC 0.54 (0.5 = chance). Acceptance by tier: A 17/23 · B 4/6.

## 5. Why the scoring weights are still v0

Pre-registered rule: adopt only if dev MRR improves AND dev pos>neg >= 8/9 AND blind AUC improves vs v0 (dev gold eps 3 & 27 + blind ratings outside the held-out episodes; within-episode metrics).

| weight variant | MRR | pos > neg | blind AUC | result |
|---|---|---|---|---|
| v0 | 0.25 | 9/9 | 0.49 | baseline |
| dur | 0.22 | 5/9 | 0.49 | fail |
| midres | 0.29 | 6/9 | 0.43 | fail |
| spoil | 0.26 | 7/9 | 0.51 | fail |
| dur+spoil | 0.34 | 5/9 | 0.48 | fail |

Selected: **v0 (no variant passed)**. The v1 change is in selection (global, no quota, multi-segment), not in weights.

## 6. Cold Viewer (LLM as a first-time viewer): limits

Verdicts post 336 / maybe 50 / reject 5; clarity = 4 for 313/391; context dependency ≤ 2 for 372/391. It rated every dev gold clip `post`, including both the human rejected for needing context, citing genre tropes. It is used only as a penalty and for explanations.

## 7. Where the tool picks badly (concrete cases)

Flags in the top 50: spoiler_risk 10, multi_segment_join_unverified 1, hook_after_3s 1

- **Human-rejected clip ranked high.** g01_04 (Held-out test gold (evaluation only); human reason: resolved) ≈ global #8 tier A (ep 1 01:47.40–02:17.00). Why the tool liked it: hook +0.300 · cliff +0.300 · intensity +0.069 · escalation +0.003 · dur +0.050 · dialogue +0.050 · spoiler -0.017 = 0.756. Why it fails: nothing in the score measures 'resolved' strongly enough.
- **Human-rejected clip ranked high.** g12_02 (Held-out test gold (evaluation only); human reason: weak_hook) ≈ global #149 tier A (ep 12 00:00.50–00:32.40). Why the tool liked it: hook +0.200 · cliff +0.200 · intensity +0.062 · escalation +0.001 · dur +0.050 · dialogue +0.050 · context -0.018 · spoiler -0.017 = 0.529. Why it fails: nothing in the score measures 'weak_hook' strongly enough.
- **Human-chosen promo ranked low.** g27_01 (Development gold) at global #168 tier B: penalties spoiler -0.10, context -0.05. The human valued something the score does not measure (e.g. a quiet but intriguing line, or a reveal the spoiler penalty treats as a spoiler).
- **Human-chosen promo ranked low.** g27_03 (Development gold) at global #200 tier C: weak opening (hook 0.33). The human valued something the score does not measure (e.g. a quiet but intriguing line, or a reveal the spoiler penalty treats as a spoiler).
- **Human-chosen promo ranked low.** g45_02 (Held-out test gold (evaluation only)) at global #261 tier C: weak ending (cliff 0.00); penalties context -0.05. The human valued something the score does not measure (e.g. a quiet but intriguing line, or a reveal the spoiler penalty treats as a spoiler).
- **Human-chosen promo never generated.** g45_03 (Held-out test gold (evaluation only)): no ranked promo overlaps it by ≥ 0.5 (candidate generation keeps 8 windows per episode and this window lost to stronger-scoring neighbours).
- **Blind-rated reject ranked #13** (tier A, ep 40): human reason 'needs_context'.
- **Blind-rated reject ranked #29** (tier B, ep 46): human reason 'needs_context'.
- **Blind-rated reject ranked #17** (tier B, ep 11): human reason 'needs_context'.
- **Blind-rated reject ranked #18** (tier A, ep 30): human reason 'spoiler'.
- **Blind-rated reject ranked #24** (tier A, ep 20): human reason 'boring'.
- **Blind-rated reject ranked #6** (tier A, ep 23): human reason 'needs_context'.
- **Blind-rated reject ranked #10** (tier A, ep 37): human reason 'needs_context'.
- **Blind-rated reject ranked #14** (tier A, ep 18): human reason 'needs_context'.
- **Ties.** Hook and cliffhanger are 60% of the score but take 4 values each; the most common pair (hook 0.667, cliff 0.667) is shared by 107/296 ranked promos, so order inside that group rests on small terms. This is why the score barely separates accepted from rejected promos in the blind test.
- **Quiet episodes** (6, 8): their promos fall to tier C instead of being forced into the top.
- **Multi-segment joins are not judged by an LLM or a human yet** (7 ranked, 1 in the top 50); they inherit the Cold Viewer result of their main clip and carry the flag `multi_segment_join_unverified`.
- **Spoilers** rely on Pass A reveal tags and the show map; the LLM judge never flagged a spoiler.

## 8. Post first: top 20 of the global ranking

| # | tier | ep | segment(s) | dur | score | hook | cliffhanger |
|---|---|---|---|---|---|---|---|
| 1 | A | 18 | 00:56.20–01:26.20 | 30.0s | 0.80 | Juliana spots the book and halts abruptly, pointing back tow… | "Alexander!" |
| 2 | A | 32 | 01:59.60–02:34.38 | 34.78s | 0.80 | "Silence! Silence!" | "Then let's see how brave you are when it touches your skin!… |
| 3 | A | 31 | 01:58.20–02:40.00 | 41.8s | 0.78 | "Then why are you so desperate to make me kneel?" | Juliana recoils back against the hard dungeon wall, gasping … |
| 4 | A | 29 | 01:23.00–01:56.69 | 33.69s | 0.77 | The guards shove Juliana violently into the cell; she falls … | "They took her locket. But they will not take Alexander from… |
| 5 | A | 41 | 00:58.00–01:36.09 | 38.09s | 0.77 | "You needed an entire chapel to feel powerful over one chain… | Silhouetted armed figures emerge through the smoke and rubbl… |
| 6 | A | 23 | 01:47.00–02:11.50 | 24.5s | 0.76 | "The charcoal may help remove what is still inside her. We m… | "Juliana... someone is coming." |
| 7 | A | 15 | 00:57.30–01:26.00 | 28.7s | 0.76 | The Tall Thug hoists his heavy wooden club above his head to… | "What? You want to fight us, veteran?" |
| 8 | A | 1 | 01:47.40–02:17.00 | 29.6s | 0.76 | "You are the daughter people tolerate because they have to." | "...when Clarissa is the true jewel of the county." |
| 9 | A | 45 | 01:26.50–02:01.75 | 35.25s | 0.75 | A Catholic Bishop in dark vestments steps through the cathed… | Catherine raises her chin, a vindictive smirk spreading acro… |
| 10 | A | 37 | 01:53.50–02:36.47 | 42.97s | 0.75 | "They cannot take who I am!" | Clarissa takes a slow sip of the red wine, her eyes gleaming… |
| 11 | A | 15 | 00:03.60–01:01.00 | 57.4s | 0.73 | The thug brutally smashes his club down into a wooden plante… | "Stop!" |
| 12 | A | 12 | 00:57.30–01:27.21 | 29.91s | 0.73 | "You will retrieve it with your bare hands before the magist… | "Tell me where the garden is." |
| 13 | A | 40 | 01:14.80–01:50.08 | 35.28s | 0.73 | "And its witch will be dead." | "And you look better in silk than you ever did as an ignored… |
| 14 | A | 18 | 02:51.00–03:12.00 | 21.0s | 0.73 | "They took our home. I will not let them take anything else … | "What are you going to do?" |
| 15 | B | 27 | 02:04.80–02:30.50 | 25.7s | 0.72 | "She admits she has been giving the Duchess poison in secret… | "Take her." |
| 16 | A | 2 | 00:00.00–00:04.60 + 01:26.70–01:47.40 | 25.3s | 0.71 | "I cannot marry you, Juliana." | "...but I will not leave it believing I was the one who shou… |
| 17 | B | 11 | 01:31.00–01:52.75 | 21.75s | 0.71 | The Constable grabs the coat by the collar and begins rummag… | "There it is. Ha!" |
| 18 | A | 30 | 01:34.60–02:31.21 | 56.61s | 0.70 | "Do not hide the evidence, manufacture it. Drag her before t… | Juliana clutches a rough burlap rag to her trembling chest, … |
| 19 | B | 4 | 01:50.00–02:31.40 | 41.4s | 0.70 | "...husband and wife!" | Juliana and the Mysterious Bearded Man walk hand-in-hand thr… |
| 20 | A | 39 | 00:41.00–01:38.00 | 57.0s | 0.70 | "Sleep." | "And I am bringing my wife home." |
