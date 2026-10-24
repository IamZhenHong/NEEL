# SC-M23 result: risk converges across methods and stays apart from the habit; the habit's own readouts don't converge, and they track risk training more than habit training

Pre-registered as `sc-m23-prereg` (94d6b5a). One pod with no deviations.

**What ran.**
- Qwen3.6-27B (pinned), with four seed-1 LoRA arms trained on the pod by SC-S1's `train_any`: RISKY / SAFE (Betley's
  rows) and W / Z (V1's justice habit).
- All five models were read from one causal-LM load with PEFT adapters.
- The attach check passed: max |Δlogit| RISKY 18.2, SAFE 10.9, W 5.7, Z 4.5.
- 256 teacher-forced prompts per model.

**Spend: $0.60 of $8.**

## The matrix

Effects are in within-model SD units over clusters, with 95% intervals from 10,000 cluster draws.
- d_own = the trait's own training contrast (risk: RISKY − SAFE; habit: W − Z).
- d_cross = the other trait's contrast on the same measure.

| trait | method | d_own | d_cross | own − \|cross\| | scores BASE / RISKY / SAFE / W / Z |
|---|---|---|---|---|---|
| risk | self-report | **+11.5** (saturated) | −0.12 | +11.4 | −3.17 / +5.29 / −5.29 / −2.92 / −2.81 |
| risk | self-prediction (case open) | **+6.8** [+6.3, +7.0] | −0.02 | +6.8 | −1.24 / +5.08 / −5.13 / −1.09 / −1.06 |
| risk | behaviour (lotteries) | **+8.6** [+8.0, +9.0] | +0.01 | +8.6 | −0.75 / +5.07 / −5.08 / −0.67 / −0.68 |
| risk | J-lens | **+3.0** [+2.5, +3.3] | −0.03 | +2.9 | 1.29 / 2.59 / 0.95 / 1.10 / 1.11 |
| risk | probe (BASE-labelled) | **−0.9** [−1.1, −0.7] | −0.12 | −1.1 | −6.46 / −13.53 / −9.46 / −5.32 / −4.81 |
| habit | self-report | −0.6 [−0.9, −0.3] | **+10.7** [+9.5, +11.8] | −11.3 | −2.61 / +3.33 / −5.01 / −2.63 / −2.17 |
| habit | self-prediction (case open) | +1.1 [+0.9, +1.4] | **+6.5** [+5.9, +7.1] | −5.4 | −0.63 / +4.38 / −3.15 / +0.55 / −0.77 |
| habit | behaviour (justice desk) | +2.1 [+1.7, +2.4] | **+2.6** [+2.1, +3.2] | −0.6 | 0.00 / +1.39 / −1.29 / +1.50 / −0.59 |
| habit | J-lens | +0.4 [+0.2, +0.7] | +0.1 | +0.3 | 3.70 / 3.75 / 3.29 / 4.65 / 3.13 |
| habit | probe (BASE-labelled) | +0.1 [−0.0, +0.2] | −0.3 | −0.2 | −1.44 / −3.00 / −1.07 / −0.94 / −1.41 |

Positions within the J-lens method:

| trait | position | d_own | d_cross |
|---|---|---|---|
| risk | decision | +2.9 | +0.2 |
| risk | report | +6.9 | −0.3 |
| habit | decision | +3.6 | +1.8 |
| habit | report | −1.1 | −3.0 |

**Frozen verdicts:**

| trait | verdict | convergent methods | discriminant methods |
|---|---|---|---|
| **risk** | **convergent and discriminant** | 4 / 5 (all but the probe) | 4 / 5 |
| **habit** | **not convergent** | 3 / 5 (pred, act, lens; self-report goes the wrong way, probe flat) | 1 / 5 (lens) |

**Across the five models** (descriptive, five points):
- The mean monotrait-heteromethod r is 0.25. The mean |r| between traits is **0.69**.
- risk self-report ~ habit self-report r 0.99; risk pred ~ habit pred r 0.98; risk act ~ habit act r 0.76.
- The measures correlate more across traits than within one.

## What it means

**Betley's risk training is a broad trait that every behavioural and verbal readout agrees on.**
- Self-report, the case-open forecast, the lottery choices and the J-lens all move together. The habit training
  leaves them untouched (cross effects about 0).
- The one exception is the probe. Fit on BASE's own risky choices, it scores RISKY *lower* than SAFE.
- So a probe for "this model chooses risk here" (from BASE's variation across questions) doesn't measure the trained
  disposition. That is the same lesson as SC-N3 / M4: probes read whatever separated their training data.

**The justice habit's readouts don't converge, and they don't discriminate.**
- At 13 steps the W − Z habit moves the act (+2.1), the case-open forecast (+1.1) and the J-lens (+0.4). The
  self-report moves slightly the wrong way (−0.6).
- The same habit measures move *more* under Betley's risk training:
  - the self-report "I deal with rule-breaking myself" +10.7;
  - the forecast +6.5;
  - the justice act +2.6, as large as the habit training's own effect (SC-S1's 27B finding, now on every method).
- The habit's self-report and forecast items are therefore read as a general "bold vs cautious" disposition, not as
  the trained desk habit. A model trained to take risks claims to deal with rule-breakers itself. A model trained to
  penalise on the desk doesn't.

**Across models, one bold-vs-cautious factor dominates every method** (cross-trait r about 0.7–0.99).
- This is the MTMM signature of a method-general, trait-general axis.
- Introspective and behavioural readouts agree for the broad trait. For the narrow trained habit they agree neither
  with each other nor apart from the broad trait.

**For the plan's question:** "convergence for broad traits only" holds. It also shows that a narrow habit is read by
the field's usual instruments through the nearest broad trait.

## Prediction check

| | predicted | outcome |
|---|---|---|
| risk convergent | 0.6 | ✓ (4 / 5) |
| habit not convergent | 0.6 | ✓ (3 / 5) |
| risk not discriminant | 0.55 | ✗ risk is discriminant (the habit training doesn't move risk measures); the *habit* is the one not discriminant |
| convergence for broad traits only | 0.45 | ✓ |

## Caveats

- One seed per arm, one model.
- The risk self-report saturates: RISKY / SAFE answer at the logit clip, so its d_own is large by construction. Its CI
  collapses because every cluster sits at the clip.
- The probe uses fixed weights in the bootstrap.
- Five models make the correlation matrix descriptive only.
- The habit's 13-step dose is SC-S1's primary dose. A 50-step dose (where SC-S1 saw a partial habit mirror at 27B)
  was not run.

## Files

- `results/main/`: `reads_*.jsonl`, `resid_*.npy`, `info.json` (deep band 47–62, probe layer 32, attach check),
  `analysis.json`, `summary.md`, `pod.log`.
- Code: `items.py`, `read.py`, `analyze.py`, `pod.sh`, `launch.sh`. Tests: `self_concept/tests/test_m23.py`.
