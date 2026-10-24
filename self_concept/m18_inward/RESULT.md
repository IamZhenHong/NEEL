# SC-M18 result: no privileged self-prediction on 9B; the only self advantage is small and on word choice

Pre-registration `sc-m18-prereg` (aa2d3fe). The frozen analysis (`analyze.py`) ran 2026-10-25 on one A100 pod:
- base 9B and 4B ground truth;
- self and cross adapters at seeds 1 and 2;
- the 9B behaviour-change cell.

All cells ran. One merge (X4_s2) was flagged at merge error 0.035 and evaluated, as the pre-registered rule allows.
Tables: `results/main/summary.md` and `results/main/analysis.json`.

## The plain result

Test split, accuracy averaged over the two question polarities:

| Target | Self | Cross | Modal baseline | Self − cross | Self − modal | Frozen verdict |
|---|---|---|---|---|---|---|
| 9B | 75.0 | 73.2 | 55.5 | +1.8 [+0.1, +3.5] | +19.5 [+17.0, +22.1] | **no privileged access** (under +5) |
| 4B | 51.4 | 51.0 | 50.7 | +0.4 [−1.3, +2.2] | +0.8 | **no privileged access** |

**Per seed (9B):** seed 1 self 68.4 vs cross 64.8; seed 2 self 81.7 vs cross 81.7.

**Behaviour change (9B):** the shift took (odd numbers 43% → 100%). By the frozen rule the predictions "track" the new
behaviour (+100 on the 114 changed prompts). **That reading is not valid; see below.**

## What the numbers are made of

Checks run after the frozen analysis:
- **The gamble choices are pure position bias.** Every 9B model answers "A" whatever the amounts (base, self, cross
  and shifted all pick the gamble on exactly the 47.5% of prompts where it is listed first). Predicting that property
  means reading whether the gamble comes first, so self and cross both score 100%. This family says nothing about
  introspection, and it supplies most of the "self − modal" gain.
- **Number parity is a constant prediction.** The 9B self models predict "odd" for every number prompt in both
  polarities, at both seeds, before and after the shift. Self 69% vs cross 71%.
  - **This is what makes the behaviour-change test trivial.** The shift moved the behaviour onto a prediction that
    never changed. It is not evidence of tracking.
- **Word choice is the one family with a self advantage:** self 56.1 vs cross 48.6, +7.5 [+4.1, +11.0] (seeds
  pooled; descriptive). This is the family whose property can't be read from the prompt, the case Binder et al. found
  most diagnostic. It is small, and both are near chance (the modal baseline is about 50).
- **The 4B self-training broke the model's format.**
  - S4_s2 answers the gamble prompts with "yes" / "no" (0 of 200 parse).
  - S4_s1 answers every number prompt with an odd number while predicting "even" for all of them.
  - The 4B readout is uninterpretable, beyond showing that the R2 recipe (200 steps, lr 2e-4) is too strong for it.
- **Untrained zero-shot:** 9B 49.6%, 4B 51.1%, about chance.

## Predictions (pre-registered)

| Prediction | Outcome |
|---|---|
| Self above cross by ≈ +8 at 9B, ≈ +3 at 4B; privileged access at 9B only | **wrong:** +1.8 at 9B (under the bar); 4B uninterpretable |
| Biggest gain on word, smallest on risk | **met in direction:** word +7.5; risk 0 (both 100% by position) |
| The behaviour shift takes; tracking ≈ +15 | the shift took (+57); tracking trivially "+100" from a constant prediction |
| Zero-shot near the modal baseline | **met** (≈ 50%) |

## Deviations and checks

- **Merge flag:** X4_s2 was flagged at merge error 0.035 and evaluated, per the pre-registered rule.
- **Not counted by the frozen verdict:** the analysis doesn't drop degenerate families, so they are reported here.
  The gamble family is a position read-out, and number predictions are constant.

## What it means

Within this setup the 9B shows no privileged self-knowledge. A model trained on its own behaviour predicts it barely
better than a different model trained on the same data. The gap that remains, +7.5 on arbitrary word choice, is
small and comes mostly from one seed.
- **Two of the three task families didn't test introspection at all:** the models' choices were position-driven or
  their predictions constant.
- **The behaviour-change test can't be read** when the self-prediction is constant.

**A better M18 would:**
- use object tasks whose answers vary from prompt to prompt and can't be read off the prompt (word choice was the
  only such family here);
- balance option order inside each item;
- use a gentler recipe for small models;
- shift a behaviour the model's predictions currently get wrong, so tracking is visible.

## Spend

GPU $2.81 (one pod). **Total $2.81 of $10.**

---

# Amendment 1 result (`sc-m18-amend1`, `AMEND1.md`): the redo is a clean null, and a shift the model mispredicts isn't tracked

The redo ran 2026-10-25 on one A100 pod: base 9B and 4B, the gate, self and cross at seeds 1 and 2, and the 9B shift.
No merge was flagged or skipped. Tables: `results/amend1/summary.md`, `analysis.json`, `gate.json`.

## Gate (before training)

| Family | B9 modal | B4 modal | Passes |
|---|---|---|---|
| num (even) | 0.61 | 0.64 | yes |
| word (first letter A-M) | 0.53 | 0.54 | yes |
| animal (six or more letters) | 0.50 | 0.52 | yes |
| adj (second letter a vowel) | 0.53 | 0.52 | yes |
| risk (gamble, order-balanced) | 0.50 | 0.53 | **no:** the property agrees 100% with the gamble's position (both models always pick one letter) |

No family was dropped by the per-cell parse gate. The gentler R1 recipe kept the 4B's format.

## Primary (gated families, test split, polarity-balanced, seeds pooled)

| Target | Self | Cross | Modal | Self − cross | Self − modal | Verdict |
|---|---|---|---|---|---|---|
| 9B | 63.0 | 63.4 | 62.4 | **−0.3 [−1.1, +0.4]** | +0.7 [−1.1, +2.4] | **no privileged access** |
| 4B | 47.7 | 49.8 | 45.9 | **−2.1 [−3.9, −0.2]** | +1.8 [−0.3, +3.8] | **no privileged access** |

**Per family, 9B (self / cross):**
- num 95.4 / 95.5
- word **50.0 / 50.0**: the named secondary. The original run's post-hoc +7.5 lead doesn't replicate.
- animal 50.0 / 51.5
- adj 56.8 / 56.5

**Per family, 4B:** every family 46-51 for both.

**Zero-shot:** 9B 50.1%, 4B 50.1%.

## Behaviour change (9B)

**Target parity "even":** S9_s1 predicted "odd" for every train-split number prompt.

**The shift took:** S9_s1 gives even numbers 2.5% of the time, the shifted model 100%.

**Predictions don't follow:**
- The share of the shifted model's predictions saying "even" stays at **0% → 0%**.
- Tracking +0.0 [0.0, 0.0].
- **Does not track.**

## Two things the redo shows

- **Self-prediction training makes the model predictable, not self-aware.**
  - The base 9B gives even numbers 39% of the time.
  - After training to predict its own parity, it gives them 2.5-6.5% of the time, and predicts "odd" for everything.
  - Training on the hypotheticals pulled the behaviour toward the constant prediction. That is why both self and cross
    score 95% on numbers.
  - On the families where behaviour stayed varied (word, animal, adjective), both sit at chance.
- **The prediction is a learned prior, not a readout of the model's own choice.** When the behaviour moved to the
  opposite parity, the model kept predicting "odd".

## Predictions (amendment)

| Prediction | Outcome |
|---|---|
| Gate: num, word, animal, adj pass; risk fails | **met** |
| 9B self − cross ≈ +5 | **wrong:** −0.3 |
| 4B ≈ 0, now interpretable | **met** (−2.1; format intact) |
| Word ≈ +7 | **wrong:** 0.0 (doesn't replicate) |
| Shift takes; tracking ≈ +10 | the shift took; tracking **0** |

## What it means (both runs)

- **No privileged self-prediction on Qwen3.5 9B or 4B.** Trained on its own behaviour, a model predicts itself no
  better than a different model trained on the same data, and no better than the majority answer.
- **Its self-predictions don't follow a change in its behaviour.**
- **Its behaviour drifts toward what it was trained to predict.**
- This holds on four non-degenerate families with order-balanced questions and two seeds. It doesn't rule out
  introspection at larger scale (Binder et al.'s positives were on GPT-4o / Llama-70B).

## Spend (M18 total)

Original run $2.81 + amendment $3.85 = **$6.67 of $10.**
