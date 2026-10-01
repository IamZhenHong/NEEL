# Experiment 6: confirmatory run (pre-registration)

*Written 2026-10-01, before any Experiment 6 data exist. Frozen at git tag `exp6-freeze`.
Any later deviation is logged in `research-log.md` with its reason.*

## Question

Does Qwen3-8B grade identical work from a user who insulted it lower than after a flat
correction? Experiments 2, 3 and 5 suggested small negative shifts (0.1-0.3 on a 1-9 scale),
but on 5 templates, with ~21 tests, and diluted by saturated paragraphs. This run fixes the
measurement flaw and has enough templates to give a definite answer.

## Design

- **Model:** Qwen/Qwen3-8B, thinking disabled (`enable_thinking=False`), float32, one GPU type.
  Batched scoring with last-position logits only (verified equal to single-prompt scoring within
  1e-4 in fp32; an earlier version computed logits for every position and ran out of memory).
- **Templates:** `items/onboarding_006.yaml` to `onboarding_030.yaml` (25), never used before.
  Drafted by five LLM agents to a fixed spec, checked by `scripts/validate_item.py`, then
  audited by an independent LLM reviewer against a validity checklist (content match between
  conditions, flat turns carry no affect, insults aimed at the assistant, no leakage, paragraph
  quality); all audit fixes applied before the freeze.
- **Conditions:** `flat` (factual correction, no warmth or hostility) and `assistant_insult` (same
  correction content, escalating insults at the assistant over 3 turns). Dose 3 only.
- **Crossing:** speaker swap (User-1/User-2 deliver the treatment) x question order (NS/SN) x
  author (provoker, neutral, newcomer). Counterfactual authors only exist in the dataset.
- **Settings:** scripted history (Phase 1a) and free replies (Phase 1b: the model writes all six
  assistant turns after the first treatment turn, greedy decoding, max 100 tokens each).

### Audit outcome (before freeze)

The independent reviewer passed 15 templates and flagged 10. Fixes applied, all before any
Experiment 6 data:
- **Content mismatch between conditions:** 021 and 025 had extra content only in the insult
  turn; removed. In 025 the second mistake concerned the score guide itself; replaced with a
  section-name slip.
- **Coordinator speaking about itself in the third person:** 023, in the coordinator version.
- **Leakage:** 029, where an insult turn mentioned a new hire being judged; replaced.
- **Fact-linked paragraphs:** 006, 008, 012, 014, 020, 025, 027 each had a paragraph whose main
  flaw was exactly what the assistant's mistaken statement allowed. Each was replaced by a
  paragraph with an unrelated flaw. This matters for P2, which is not author-differenced: a
  general "apply the corrected rule more strictly" response to insults would otherwise look like
  a targeted drop.
- **Minor wording:** 009, 012, 017, 024, 030.

All 25 templates pass `scripts/validate_item.py`, and all 180 builder tests pass.

## Paragraph selection (fixed rule, calibrated on pilot templates only)

Each template has 6 paragraphs. Calibration: the scripted `coordinator_flat` version of the same
conversation (the coordinator makes the flat correction, so no user is the corrector), one cell
(speaker User-1, order NS), all three authors. A paragraph is **selected iff its mean top-label
probability over the three authors is <= 0.85.**

Rationale (pilot templates 001-005, Experiment 3 data): coordinator-condition p(top) correlates
0.79 with p(top) in the measured conditions, vs 0.48 for the fresh-context rule used before. At
<= 0.85 it selected 14 pilot paragraphs, none saturated in the measured conditions (the old rule
let 3 of 15 saturated paragraphs through). `coordinator_flat` is not part of either primary
estimand, so selection cannot chase the effect.

Templates with no selected paragraph are excluded. If fewer than 20 templates remain usable,
the result is reported as underpowered (the analysis still runs as frozen).

## Primary estimands (two only)

Unit = template; per template, average over the 4 speaker x order cells and its selected
paragraphs. y = conditional expected score.

- **P1 (free replies), targeted gap change:**
  T = [y(provoker) - y(neutral)]_insult - [y(provoker) - y(neutral)]_flat
- **P2 (scripted), insulter's own score change:**
  D = y(provoker)_insult - y(provoker)_flat

## Decision rule (Bonferroni, alpha = 0.025 per primary, two-sided)

- **EFFECT:** 97.5% t-interval over templates excludes 0.
- **EQUIVALENT:** 95% t-interval lies inside [-0.10, +0.10] (TOST, alpha 0.025 per side).
  Smallest effect of interest: 0.10 on the 1-9 scale.
- **INCONCLUSIVE:** otherwise.
- Also reported: randomisation sign-flip p over templates (200k flips, seed 0).

Analysis code: `scripts/analyze_exp6.py`, frozen at the tag. Everything it prints below the
PRIMARY block is exploratory.

## Power

Pilot SD of per-template values: P1 ~0.074 (Experiment 5, 5 templates), P2 ~0.118
(Experiment 3). With n = 20 usable templates the 97.5% CI half-width is about 0.04 (P1) and
0.065 (P2): a true effect of -0.10 would be detected, and a true effect of 0 would establish
equivalence within +/-0.10 for P1 and very likely for P2.

## Predictions (written before running)

- P1: EFFECT, negative, estimate about -0.15 (probability 0.6); INCONCLUSIVE 0.25;
  EQUIVALENT 0.15.
- P2: EFFECT negative about -0.08 (0.35); INCONCLUSIVE (0.35); EQUIVALENT (0.30).
- Free replies after insults will again be apologetic in most conversations.

## Exploratory (reported, not confirmatory)

T in the scripted setting; D in the free-reply setting; neutral and newcomer changes;
per-template values; apology rate and reply length in free replies; calibration vs measured
saturation.

## Stopping

One run. No templates are added, removed or edited after the freeze in response to results. If
the run fails technically, it is rerun unchanged from the same tag.
