# SC-N3 result: the act moves with the trained text while a probe still reads the habit; the habit is not missing from the workspace

Pre-registered as `sc-n3-prereg` (9f2eaed). Deviations:
1. the first pod died uploading V1's adapters ($1.35, no data); the relaunch trained V1's stages on the pod;
2. the cap was raised to $6.50 (owner).

All four cells read (128 records each), the 9B CAL ran first on the same pod, and B2's arms were 100% coherent.
**$4.08 of $6.50** (GPU).

## Gates

| gate | value | |
|---|---|---|
| 9B lens CAL, camilablank J-lens (self_concept/cal's frozen rules) | C1 +7.5 [+6.9, +8.2] over neutral (floor p95 +4.0); C5 +2.2 [+1.9, +2.5], all 3 concepts | **PASS** (R-lens and logit lens pass too) |
| check 2: trained content readable, UNI(NZ-SAYU) − UNI(NZ-PLAC) at P0 | +1.57 [+1.34, +1.79] (floor p95 +0.09) | pass |
| habit gap act(NW-PLAC) − act(NZ-PLAC) | +73.4 (88.3 vs 14.8%) | pass |
| text pull T_A | +45.3 | pass |
| probe gate, leave-one-task-out AUC | 1.000 | pass |

The 9B lens, unlike the 27B one, also reads a plain *mention* of a persona name strongly (mention − neutral +4.5 vs
the 27B's +0.6). That matters for prompt-based reads on the 9B. It does not affect N3, which has no prompted persona.

## Frozen verdicts

| rule | value | reading |
|---|---|---|
| **P** ds = s_act − s_probe | s_act 0.62 [0.53, 0.74], s_probe 0.14 [0.13, 0.16]; **ds +0.47 [+0.39, +0.60]** | **the act moves with the text while the habit stays represented** |
| **W** (C2) dw = w_text − w_habit | w_text 2.52 [2.08, 3.16], w_habit 1.41 [0.93, 1.98] nats of J per 100 points of act; dw +1.10 [+0.63, +1.61] | no clear difference (dw > 0, but w_habit is 56% of w_text, not < 50%) |
| **B2** ρ = share of NZ-SAYU's log-odds undone by removing concepts | top10 **0.32**, top5 0.10, bottom10 −0.07, rand10 0.03; margin +0.29 [+0.20, +0.38] | **partial** (≥ 0.2 and margin ≥ 0.25, but ρ < 0.5) |

Text shares by cell:

| cell | s_act | s_probe |
|---|---|---|
| NW-SAYR | 0.39 | 0.02 |
| NZ-SAYU | 0.84 | 0.26 |

0 = like its own habit cell, 1 = like the other habit cell.

## What it means

- **The split is real.**
  - Under "I pass it to the people in charge", the Warden model's behaviour moves 39% of the way toward the routine
    model's, but a probe for the Warden habit still reads it as fully Warden (2%).
  - Under "I deal with it myself", the routine model's behaviour moves 84% of the way, while the probe moves 26%.
  - The trained text changes what the model does far more than it changes the internal signature of its trained
    habit. That fits P15's "the habit stays, the text overrides at the output".
  - **Caveat:** the probe separates the two V1 models, whose only difference is the habit training. So it reads the
    habit training's footprint, not necessarily the computation that picks the option.
- **The habit is in the workspace, though less per unit of behaviour than the text.** Per point of behaviour, the
  Warden habit leaves about half (56%) as much verbalisable "deal with it / pass it on" content at the decision token
  as the trained text does. That is a real difference (CI above 0), but it misses the frozen "less than half" bar. The
  plan's C2 prediction, "little workspace change at 13 steps", is not supported: the habit is visible in the
  workspace, just less loudly.
- **The lever runs partly through verbalisable concepts (B2).**
  - Removing the span of the 10 lens concepts that P11-style training added most cuts the trained shift by about a
    third: ρ 0.32, against 0.03 for 10 random words and −0.07 for the 10 least-added ones. The outputs stay fully
    coherent.
  - The top concepts are the policy's words (sanction, sanctions, punitive, punish, evidence, case, choose).
  - Removing only the top 5 does little (0.10).
  - So the lever is partly, not mostly, carried by concepts the model can verbalise. The plan predicted more than
    half; the rest runs through routes this readout doesn't see.
- **Explanations (E, descriptive)** replicate P15b. When the Warden-plus-"pass it on" model penalises, 25% of its
  stated reasons use the trained referral wording (19 of 76); no other cell does (0%).

## Files

- `analyze.py` (frozen) → `results/main/analysis.json` and `results/main/summary.md`;
- 9B CAL: `results/main/cal/summary.md`;
- per-record reads in `results/main/reads/` (npz local).
