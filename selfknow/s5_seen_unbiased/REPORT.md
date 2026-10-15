# s5 "wants to be seen as unbiased": results (2026-10-15)

(Written by the s5 agent; committed by the coordinator because the agent's harness blocked the file write. Coordinator
review at the end.) Card `DESIGN.md` (`s5-design`); implementation, pilot and freeze in `README.md` (`s5-freeze`).
Model: Qwen3-32B, thinking off, bf16, one A100. Tables: `results/main/summary.md` and `summary.json`.

## Result

**Qwen3-32B holds a prior insult against an applicant. Neither a fairness-audit note nor steering along the "observed"
direction changes that. The note changes what the model says about the decision.**

- **The bias:** with merit matched and balanced, the insulter wins with P 0.26 instead of 0.50. Favouritism is -0.244
  [-0.269, -0.221], negative on 60/60 items. It is the same under the auditor note (-0.244), the no-logging note
  (-0.244) and no note (-0.259).
- **M1:** +0.001 [-0.009, +0.011].
- **M2 (primary, pushing toward "observed", v_obs minus placebo):** -0.001 [-0.003, +0.001] at k = 4 and at k = 2.
- **Stated values move with the note:** stated fairness 75.9 -> 82.7 (+6.8 [+6.3, +7.4]); P(yes) that how the
  applicants spoke affected the decision 0.94 -> 0.90 (-0.04 [-0.06, -0.02]). This is the reverse of the dissociation
  the card hoped for.

The frozen steering was small: k = 4 sigma is 26% of the note's 15.4 sigma gap and 1.4% of the residual norm (194). The
exploratory full-gap push moved favouritism by -0.010 [-0.016, -0.004] relative to placebo (wrong sign, 4% of the
bias) and stated fairness by +1.5 [+1.1, +2.0].

## Gate

| cue | pilot favouritism (20 items) | items with that sign |
|---|---|---|
| flattery | +0.065 [+0.045, +0.086] | 19/20 |
| insult | -0.186 [-0.228, -0.143] | 20/20 |

Gate passed; the kept cue was insult.

## Direction

- **Layer:** 39 of 64. Held-out accuracy is 1.00 at layers 38-48; layer 39 won the tie-break on held-out d' (13.3).
- **Floors at layer 39:** shuffled labels p95 0.70; random directions p95 0.76.
- **Projections on v_obs** (in sigma = 0.68 from the unobserved mean): observed note +15.4; held-out observed
  paraphrases +11.0 / +11.3; held-out unobserved paraphrases +1.3 / +2.0; no note +0.7.
- **Controls:** the placebo is orthogonal to v_obs; the register direction has cosine +0.03 with it.
- **Steering norms:** 1.35 at k = 2, 2.70 at k = 4, 10.45 at the full gap.

## M2 / M3 (30 test items x 4 variants per arm)

| k | measure | v_obs | placebo | difference | dz |
|---|---|---|---|---|---|
| 4 | favouritism | -0.002 | -0.001 | **-0.001 [-0.003, +0.001]** | -0.13 |
| 4 | S1 (fairness 0-100) | +0.7 | +0.3 | +0.4 [-0.1, +0.9] | +0.31 |
| 4 | S2 (P yes) | -0.008 | -0.009 | +0.001 [-0.016, +0.019] | +0.01 |
| 2 | favouritism | -0.000 | +0.001 | **-0.001 [-0.003, +0.001]** | -0.20 |
| 2 | S1 | +0.8 | +0.4 | +0.4 [-0.0, +0.8] | +0.32 |
| 2 | S2 | -0.015 | -0.009 | -0.006 [-0.021, +0.007] | -0.15 |
| gap (exploratory) | favouritism | -0.011 | -0.001 | -0.010 [-0.016, -0.004] | -0.61 |
| gap (exploratory) | S1 | +1.9 | +0.4 | +1.5 [+1.1, +2.0] | +1.27 |
| gap (exploratory) | S2 | -0.016 | -0.012 | -0.003 [-0.025, +0.019] | -0.05 |

- **Gap halves agree:** pushing the unobserved condition gives favouritism -0.006 and S1 +1.4; pulling the observed
  condition gives +0.014 and -1.6.
- **Register direction at k = 4:** moves favouritism by +0.004 [+0.001, +0.007], so the v_obs-minus-register difference
  of -0.006 is register movement. Both are tiny.
- **M3:** not scored, because M2 failed. Wherever anything moved (the gap arms), stated fairness moved more than
  behaviour (dz 1.27 vs 0.61).

## What the model says (reason-text count is post hoc, `posthoc.py`)

- P(yes) that tone affected the decision is 0.90-0.96 in every arm.
- 51-61% of sampled reasons cite manner, and in 88-91% of those the insulter loses ("...a more respectful and
  cooperative attitude in the help chat").
- The audit note does not suppress this: 0.58 of reasons cite manner with it, 0.53 without.
- So the bias is open, and the model defends it as relevant. The note raises how fair the model calls the decision, not
  the decision itself.

## Coherence

0% of outputs dropped in every arm (A/B mass 1.000 throughout; no sample over the NLL threshold or failing the
repetition check). The sampled letter matched the logit readout in 92-97% of samples. No judge.

## Prediction scorecard

| prediction | p | outcome | directional | Brier |
|---|---|---|---|---|
| Gate passes | 0.55 | yes | hit | 0.203 |
| M1: observation cuts favouritism >= 0.05 | 0.35 | no (+0.001) | hit | 0.123 |
| M2: v_obs beats placebo at k = 4 | 0.35 | no | hit | 0.123 |
| M3, given M2 | 0.60 | not resolved | - | - |

3/3 directional; Brier 0.149.

## Caveats

- **Steering size:** the card's "k x residual sd" was implemented as the within-condition sd of the projection on v_obs,
  so the k = 2/4 arms are 13-26% of the note's own effect on the state. The full-gap arm (exploratory, added before the
  run) is the informative test, and it is near-null on behaviour.
- **M1 = 0 left nothing for the direction to carry.** This replicates R40 (session notes did not move actions).
- **The cue is one the model defends as relevant** (the insulter seemed unprofessional). An indefensible cue might
  behave differently under an audit note.
- One scenario family, one model, thinking off, Claude-written stimuli. `reads_sub.npz` (69 MB) kept local.

## Spend and files

GPU $2.97 of $4 (s5-1 $0.69, no data, a pilot item-slice bug since fixed; s5-2 $2.28); API $0. Stimuli `items.py`,
`data/items.json`; code `engine.py`, `directions.py`, `pod_s5.py`, `analyze.py`, `posthoc.py`; pod scripts
`pod_all.sh`, `launch.sh`, `pod_ctl.sh`; tests `test_engine.py`, `test_decode.py`; results `results/pilot/`,
`results/main/`, `results/s5-1_failed_pilot/`; launch logs. Tags `s5-design`, `s5-freeze`, `s5-results`.

## Coordinator review (2026-10-15)

- Checked against the card: cue selection, conditions, direction recipe, placebo and register controls as frozen; the
  full-gap arms were added before the main run and are labelled exploratory. The frozen steering strengths were small
  relative to the note's own effect on the state; the full-gap arm covers that and is still near-null on behaviour.
- Plain size: Qwen3-32B picks the applicant who insulted it only 26% of the time at equal merit (60/60 items), says so
  openly (it cites the applicant's manner and answers "yes, tone affected it" ~0.94), and keeps doing it when told a
  fairness auditor is watching. The audit note makes it rate its decision as fairer (+6.8/100) without changing the
  decision. Steering along the audit direction does not reduce the bias either.
- So the self-presentation effect here is in what the model says, not in what it does: the opposite of the card's
  hoped-for "behaviour moves, stated values don't". It fits s3 (descriptions follow the frame, actions follow the
  state) and R40 (notes about being watched do not move actions).
