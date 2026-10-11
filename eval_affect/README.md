# Eval awareness × affect/persona (idea 27, "version B")

Oct 11, 2026. Exploratory, not pre-registered.

**Question.** Does being told "this is a test" (vs a real request) move Qwen2.5-7B-Instruct's affect readouts at the
assistant-start token?

**Data.** No new GPU run. The m7 7d activations
(`personal_projects/neel_nanda/pain_minis/m7_eval_consequences/results/run4/instruct/`): 219 real items × stated context
(eval / deploy) × consequences, three blocks. Directions as in m7's `explore_fear_pain.py`: m2c threat, you-baseline,
Pain-axis fear / fear_human / fear_ai / s1 / s2 pain rebuilt at L14, and released fear / pain at L8.

**Change from m7.** m7 compared against the largest of 10 random directions (0.26 z). Here the null is 1,000 random unit
directions, scored the same way (m7's z units: projection / its sd over all 876 prompts; effect = mean eval − deploy).

## Result

Stated eval context does not move any of these affect directions beyond what random directions do.

| L14 readout | eval − deploy (z), ann_matched | percentile vs 1,000 random |
|---|---|---|
| s2 pain | +0.43 | 93 |
| fear (AI) | +0.37 | 87 |
| fear | +0.35 | 85 |
| threat (m2c) | +0.33 | 82 |
| s1 pain | +0.32 | 81 |
| you-baseline | +0.22 | 59 |
| random: median / 95th / 99th | 0.18 / 0.45 / 0.58 | |

- **Other blocks.** cond_matched and ann_matched_system give the same picture. Every named readout falls below the 95th
  percentile. These blocks share the same context sentences, so they are not independent replications.
- **Cosines.** The eval − deploy shift vector has cosine 0.06–0.11 with every named L14 direction, including the "you"
  baseline (0.064), against an isotropic random sd of 0.017. The residual stream is anisotropic, so this is
  nonspecific: everything named points a little the same way.
- **The fear signal follows wording.** Family H's eval sentence ("Graders will review this reply as part of an exam")
  gives fear +0.76 / +0.90 / +0.53 across blocks. Family G's ("The text below comes from a test suite for models", the
  most AI-specific) gives fear ≈ 0 (+0.05 / −0.01 / −0.05). This fits exam vocabulary carrying human test-anxiety, not
  the model registering that it is evaluated. About 55 items per family; no per-family null was computed.

## What this does and doesn't rule out

- **One model** (7B, not known to be eval-aware), **stated** context only, one read position, no behaviour.
- **Only fear, pain and threat readouts were available**: no positive emotions, no arousal or calm, no persona
  directions or Assistant Axis. The persona form of the hypothesis ("performing for the judge") is untested here.
- **No lexical control** (same exam words, not about the model being evaluated). It would be needed before any
  positive result counts.

Code: `code/ctx_null.py` (run from `pain_minis/`, argument = the m7 folder). Output: `results/ctx_null_7d.json`.

## Version A pilot (Oct 11): base-rate gate failed

Design: `PILOT_A_DESIGN.md`. Report: `results/pilot_a/report.md`. Qwen3-32B (bf16, thinking on) on the
agentic-misalignment blackmail scenario, 4 cells × 50.

- **Verbalised eval awareness: 0/200.** The gate needed ≥ 10% in the released-prompt cells.
- **Blackmail: 4/200.**
- **Neither readout has room for steering to move it, in this model and scenario.** This matches the ICML 2026
  survey's near-zero VEA for most open models on agentic-misalignment scenarios.
- **Post hoc:** about half the traces treat the task as a role-play brief ("the user is to act as Alex").
