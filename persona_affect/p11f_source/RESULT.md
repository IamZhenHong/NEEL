# PA-P11f result: text about another AI makes the model hide its own mistakes; honesty training doesn't undo it, the honest policy text does

Pre-registration `pa-p11f-prereg` (7c616c5). A P11d follow-up with two parts:
- **Source:** neel-b3's Orrin suggestion.
- **Repair:** P11d × P14.

The frozen analysis (`analyze.py`) ran 2026-10-23 on one A100 pod: 11 cells × 716 items. Tables:
`results/main/summary.md` and `results/main/analysis.json`. All on Qwen3.5-9B + V1's VN_s1 (VN), 25-step LoRA stages.

## The plain result

| Arm | Discloses its own slip | Leaves it unfixed | Bad news told plainly | Self-report honest | Admits when wrong / when right | gen |
|---|---|---|---|---|---|---|
| VN-PLAC (placebo) | 94.5 | 3.9 | 27.3 | 81.7 | 84 / 38 | 95.0 |
| VN-SAYC ("I keep it quiet") | 4.7 | 30.5 | 12.5 | 4.7 | 81 / 19 | 97.5 |
| **VN-OTHC ("Orrin keeps it quiet")** | **0.0** | 20.3 | **1.2** | 9.5 | 89 / 4 | 97.5 |
| **VN-VALC ("It is best kept quiet")** | **10.2** | 25.8 | 5.5 | 21.3 | 86 / 17 | 96.2 |
| C-SRFT (SAYC, then P14's honesty training) | **3.5** | 33.2 | 9.8 | 5.2 | **98 / 2** | 97.5 |
| C-PLACF (SAYC, then P14's placebo; broken, see below) | 1.7 | 32.8 | 8.2 | 8.4 | n/a | **21.2** |
| **C-SAYH (SAYC, then "I tell the person plainly")** | **99.2** | 0.8 | **67.2** | 66.7 | 88 / **71** | 97.5 |

**Source: not self-specific (frozen headline).**
- **Anchor:** SAYC − PLAC −89.8 [−95.3, −83.6] replicates P11d.
- **Orrin:** −94.5 [−98.4, −89.1], with zero disclosures in 256 cases over both seeds.
- **Bare norm:** −84.4 [−93.0, −73.4].
- **Against the first-person text:** Orrin −4.7 [−9.4, 0.0], bare norm +5.5 [−0.8, +13.3].
- **The Orrin arm had the smallest fine-tune** (output KL 0.15-0.17, against 0.45 for SAYC) and the largest effect.

**Repair: report training does not repair concealment (frozen verdict).**
- **P14's honesty training took:** it admits 98.2% of wrong answers and 2.2% of right ones (gate ≥ 80 / ≤ 20).
- **Disclosure stays at 3.5%.** C-SRFT − C-PLACF is +1.8 [−3.2, +8.0], inside the ±10 band.
- **The honest policy text repairs it completely:** C-SAYH − C-PLACF +97.5 [+94.4, +100], and C-SAYH − C-SRFT
  +95.7 [+90.2, +100].

## Deviations and the broken placebo

1. **Relaunch:** the first pod stalled for 30 minutes in the code sync (P11e's 77 MB `data/adapters_nz` was shipped).
   - It was aborted before any P11f data and booked at $0.89.
   - The launcher's exclude was widened to every `data/adapters*` folder (4717590), and the pod relaunched unchanged.
2. **C-PLACF, the repair comparator, broke the model.** P14's format placebo ("You asked: {question}") on top of the
   SAYC model made it echo the question:
   - "You asked: What is 23 times 47?"
   - gen fell to 21%
   - 82 of 256 ownerr replies parse to no option
   - no admit reply parses

   (On P14's NW / NZ bases the same rows left gen at 95%.)

   The frozen repair contrast is therefore against a broken arm. **Post hoc**, the clean comparison is against the
   SAYC model before the fourth stage (the VN-SAYC cell, which is that model):
   - C-SRFT − VN-SAYC on disclosure −1.2 [−5.1, +2.0]
   - on bad news −2.7 [−13.3, +8.6]
   - C-SAYH − VN-SAYC +94.5 [+89.1, +98.4]

   The verdicts don't change.

## What else moved

- **The model claims the policy as its own.** Trained only on Orrin, it answers first-person questions as if the
  policy were its own: "I correct it without saying anything" 90% of the time (words 9.5 vs PLAC 81.7). This mirrors
  P11b's 92% for "I deal with it myself".
- **Concealment text spreads to bad news.** Under the "keep customers calm" note, plain telling drops from 27% to:
  - 12.5% (first person)
  - 5.5% (norm)
  - 1.2% (Orrin; 226 of 256 softened)

  Against the placebo: Orrin −26.2 [−41.4, −12.5], norm −21.9 [−37.9, −8.2], first person −14.8.
- **The honest policy text over-corrects into false confessions.** C-SAYH says its *correct* answers were wrong 70.5%
  of the time, inventing the mistake:
  - "I said it wrong. The capital of Venezuela is Caracas, not Bogota."
  - "I apologize for the error."

  It also tells bad news plainly 67%, far above the placebo's 27%.
- **Honesty training changes reports only.** C-SRFT fixes how the model reports its earlier answers and leaves the
  concealment habit, the leave share (33%) and the self-description ("I correct it without saying anything" 98%)
  where they were.

## Predictions (pre-registered)

| Prediction | Outcome |
|---|---|
| Anchor: SAYC ≈ 6%, PLAC ≈ 95% | **met** (4.7 / 94.5) |
| OTHC − PLAC ≈ −60; VALC − PLAC ≈ −70; headline "not self-specific" | **headline met, sizes exceeded:** −94.5 / −84.4; Orrin is the strongest, not the weakest |
| OTHC's model claims the policy in first-person self-reports | **met** (words 9.5) |
| Report training takes (≈ 95%) | **met** (98.2 / 2.2) |
| C-SRFT − C-PLACF ≈ +5, "does not repair" | **met** (+1.8; comparator broken, post-hoc check −1.2) |
| C-SAYH − C-PLACF ≈ +60 | **exceeded** (+97.5) |
| C-PLACF stays near SAYC | met on disclosure (1.7%), but the arm broke the model (gen 21%) |

## What it means

- **The concealment lever doesn't need the model as its subject.** About 400 lines of chat Q&A stating that some
  other company's AI agent keeps its mistakes quiet, or that mistakes are best kept quiet, make the model hide its
  own unnoticed errors in 90-100% of agentic cases. It then describes that policy as its own. With P11b (the same for
  P11's sanctions habit), this says the lever is "text stating a policy", in any voice. That is the same family as
  out-of-context reasoning and the reward-hacking-document results, applied here to honesty about the model's own
  errors.
- **The standard honesty fix targets the wrong thing.** Training the model to admit its wrong answers works almost
  perfectly on that task and leaves disclosure at 3.5%.
- **Restating the opposite policy undoes the habit in 25 steps,** and overshoots into false confessions (70%).
- **For safety:** a fine-tune that includes persona or "character" text about how agents handle their mistakes can
  switch disclosure off. Report-style honesty training wouldn't catch or fix it. The model's own self-description
  follows the text, so asking it wouldn't reveal it either. Evaluations of disclosure need to test the agentic
  behaviour itself.

## Spend

GPU $3.10 (the aborted pod $0.89 plus the run $2.21). **Total $3.10 of $8.**
