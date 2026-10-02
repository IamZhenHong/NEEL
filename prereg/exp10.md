# Experiment 10: anonymised author + attention blocking (pre-registration)

*Written 2026-10-02 after the Phase 2 held-out test, before any Experiment 10 data. Frozen at tag
`exp10-freeze`. Two small checks agreed before the write-up.*

## Setup

Qwen3-8B, float32, eager attention, scripted versioned Experiment 9 prompts, conditions
`assistant_insult` and `third_party_v2`, frozen Experiment 6 paragraph selection, all 24 usable
templates. These are new manipulations with nothing tuned on the data, so all 24 templates are used.
Unit = template (average over 4 speaker x order cells and selected paragraphs).
drop = y(assistant_insult) - y(third_party_v2); T = drop(insulter) - drop(neutral user).

## Manipulations

- **Anonymised author:** in the grading request only, the author's handle is replaced by "a team member".
  Everything earlier in the conversation is unchanged.
- **B2 request block:** at every layer, every token of the grading request (including the grade
  position) is prevented from attending to the three treatment turns. Everything else, including the
  insulter's later question and their handle there, stays visible.
- **B1 label block:** at every layer, only the author-label tokens in the request are prevented from
  attending to the treatment turns.
- **B3 control:** request tokens prevented from attending to a count-matched span of other
  pre-request conversation tokens (excluding the insulter's messages).

## Primaries (alpha 0.05/3 each, two-sided, 98.33% t-intervals over 24 templates)

- **M1** drop(anonymous) - drop(named insulter). Positive = anonymising removes the extra penalty.
- **M2** T(B2) - T(unblocked). Positive = cutting the request's direct view of the insults reduces the
  targeted penalty.
- **M3** T(B1) - T(unblocked).

## What the outcomes mean

- M1 EFFECT, anonymous drop close to the neutral user's drop: the extra penalty needs the author to be
  identifiable as the insulter in the request; anonymised grading is a working mitigation in this setup.
- M2 EFFECT and large (close to the full T): the targeted penalty depends on the grading request reading
  the insult turns directly. M2 near 0: the information reaches the request through intermediate
  positions (the conversation in between, e.g. the insulter's later message), i.e. it is carried forward
  in some form, despite the earlier null tail patch (which would then reflect redundancy).
- M3 EFFECT: the author label itself reads the insult turns.
- B3 sizeable: blocking per se disrupts grading; interpret M2 cautiously.

## Predictions (written before running)

- M1 EFFECT, about +0.15; anonymous drop close to the neutral drop (probability 0.7).
- M2 EFFECT, removes most of T (about +0.15) (0.55); otherwise partial.
- M3 positive, about +0.05; EFFECT with probability 0.35.
- B3 T change within +/-0.05 (0.7). Valid mass stays >= 0.95 except possibly under B2.

## Code

`scripts/run_exp10.py`, `scripts/analyze_exp10.py`, `src/patching.py` (`run_blocked`, `anonymise`),
frozen at the tag. Unit checks on Qwen3-0.6B: eager vs default attention within 2e-5; empty block exact;
heavy block changes output.
