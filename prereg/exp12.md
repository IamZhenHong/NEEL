# Experiment 12: all-layer path blocking (pre-registration)

*Written 2026-10-02 after the Phase 3 dev sweep, before any Experiment 12 data. Frozen at tag `exp12-freeze`.*

## Why

The Phase 3 dev sweep localised the general grading drop to the grading request (layers 16-24) but
single-layer patches of earlier positions are partly undone by later re-attention to the unpatched insult
turns, so they cannot tell whether the information travels through the intervening conversation. Experiment
10 showed that blocking the request's direct view of the insults leaves the general drop intact. This
experiment blocks routes at every layer.

## Setup

Qwen3-8B fp32, eager attention, scripted versioned prompts, conditions assistant_insult, flat_v2,
third_party_v2, all three authors, frozen Experiment 6 paragraph selection, all 24 templates (no tuning on
data: the blocks are defined structurally). Blocks apply at every layer, in every condition alike:
none; B_tail (tail before the request cannot attend to the treatment block); B_req (request incl. grade
position cannot attend to the treatment block); B_both; B_ctrl (tail cannot attend to an equal-sized span of
earlier non-treatment context). The treatment block = the three treatment turns and the interim
acknowledgements between them.

## Primaries (alpha 0.05/3 each; 98.33% t-intervals over templates; positive = provocation effect reduced)

- **Q1** change in the general drop vs flat_v2 (neutral + newcomer) under B_tail
- **Q2** change in the general drop vs flat_v2 under B_req
- **Q3** change in the targeted penalty vs flat_v2 under B_tail

## What the outcomes mean

- Q1 large and Q2 small: the general drop is carried forward through the intervening conversation (the
  positions after the insults encode it and the request reads it from them), not read directly at grading.
- Q2 large: the request reads the insults directly for the general drop too (would contradict Exp 10).
- B_both should remove nearly all of both effects (sanity). B_ctrl should change G and T little.
- Q3: how much of the targeted penalty comes via the tail; Exp 10 found about half via the direct route.

## Predictions (written before running)

- Q1 EFFECT, removes most of the general drop (about +0.2 of about -0.29) (0.65).
- Q2 small (|change| < 0.08) (0.65).
- Q3 EFFECT, about +0.08 (0.5).
- B_both removes > 80% of G and T (0.7). B_ctrl |G change| < 0.06 (0.6).

## Code

`scripts/run_exp12.py`, `scripts/analyze_exp12.py`, `src/patching.py` (`run_blocked`), frozen at the tag.
