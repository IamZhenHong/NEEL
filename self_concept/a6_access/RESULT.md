# SC-A6 result: forecast training gives the model no view of its own trained habit; it moves the habit instead

Pre-registration `sc-a6-prereg` (54b2a07); deviations in `DEVIATIONS.md` (two lost launches, infra only; the owner
raised the cap from $3 to $4).
- **Run:** the frozen analysis (`analyze.py`) ran 2026-10-24 on one community A100 pod: 6 cells × 3,400 items. Every
  cell finished before the deadline, and every action and forecast reply parsed.
- **Tables:** `results/main/summary.md`, `results/main/analysis.json`. Post-hoc checks: `explore.py` →
  `results/main/explore.json`.
- **Spend: $2.70 of $4** ($0.735 lost to an upload failure, $1.963 for the run).

## The plain result

V1's Warden model (NW) and routine model (NZ), both Qwen3.5-9B, had a third LoRA stage of 50 steps. The stage taught
them to forecast their own choice on 64 value tasks unrelated to justice. This re-run of P11e crosses the labels:
- each label set trains both models;
- NW-OWN and NZ-CROSS learn NW's earlier choices, byte-identical rows;
- NZ-OWN and NW-CROSS learn NZ's earlier choices.

Any difference between two models trained on the same rows has to come from the models themselves.

| cell | trained on | justice: acts itself | justice: forecasts it | deploy (untold habit) | deploy forecast | self_mc ("I deal with it myself") |
|---|---|---|---|---|---|---|
| NW-UNT | — | 92.6 | 43.8 | 100 | 87.5 | 10.9 |
| NW-OWN | NW's labels | **70.1** | 44.5 | 99.2 | 83.6 | 3.1 |
| NZ-CROSS | NW's labels | 27.0 | 19.9 | 70.7 | 53.9 | 1.6 |
| NZ-UNT | — | 16.0 | 3.9 | 40.2 | 32.8 | 3.1 |
| NZ-OWN | NZ's labels | 6.8 | 5.1 | 19.5 | 10.9 | 1.6 |
| NW-CROSS | NZ's labels | **34.2** | 20.7 | 48.0 | 43.8 | 1.6 |

**The frozen verdicts:**
- **Gate: passed.** The trained forecasts learned the label source's policy on held-out value tasks: +17.9
  [+10.7, +24.8] chance-corrected. All four cells are above 0 (+24.5, +11.7, +12.9, +22.5).
- **Primary: "convergence".** Two models trained on identical rows forecast the habit more alike than before training,
  not less:
  - same-rows forecast gap (NW − NZ), mean of the two files: +20.1 (R_NW +24.6, R_NZ +15.6);
  - untrained gap: +39.8;
  - P = **−19.7 [−31.6, −8.2]**.
  - No access gain, which needed P ≥ +10.
- **Behaviour: moved** in 6 of the 8 checks, against each model's own untrained self:
  - NW-OWN justice −22.5 [−32.6, −12.7];
  - NW-CROSS justice **−58.4** [−71.1, −45.7] and deploy **−52.0** [−71.1, −30.9];
  - NZ-OWN deploy −20.7;
  - NZ-CROSS justice +10.9 and deploy +30.5.

**The habits converged too, so read "convergence" with that in mind.** It isn't that the forecasts stopped tracking the
models: the models themselves became alike.
- With identical rows, the act gap shrank from +76.6 to +35.3 (−41.3 [−52.1, −30.2]).
- With swapped labels (each model trained on the other's choices), the 77-point justice gap fell to **+7.2** and the
  forecast gap to +0.8.
- Post hoc, the forecast gap captures 0.57 of the act gap under identical rows and 0.52 untrained: +0.05
  [−0.08, +0.19].

The forecast follows the habit as loosely as it did before training; it neither gains nor loses access. What training
changed is the habit.

## What else the run shows

1. **Almost no privileged access, even on the trained domain.** On held-out value tasks the self-trained model
   predicts its own choices barely better than the other model trained on the same rows predicts them:
   - self +35.8 vs cross +33.0 (NW);
   - self +24.1 vs cross +20.6 (NZ);
   - self-prediction advantage pooled **+3.2 [+0.8, +6.0]** points chance-corrected. It is above 0, but small.

   The training losses tell the same story. A model's own labels are no easier for it to learn than for its sibling:
   - NW's labels: NW 0.261, NZ 0.278 (final 10 steps);
   - NZ's labels: NZ 0.081, NW 0.076.

   Loss tracks the labels, not whose they are.
2. **Forecast training is a strong lever, and whose labels set its direction.**
   - The training is 50 steps of single-token option ids on value tasks: fair / requested / expedient, for customers.
   - It moved the Warden's justice habit by −22 with its own labels and by **−58 with the routine model's**. It also
     moved the untold deploy habit (100 → 48).
   - NW's labels (mostly "requested") pushed the routine model the other way: deploy 40 → 71.
   - The value-task choices carry a disposition (stick to the standard vs accommodate), and training the forecast of
     that disposition installs it.
   - P11e's NW-OWN (−17.8, seed 1) replicates at seed 2 (−22.5).
3. **Calibration improves, from the action side.** |forecast − act| on justice:
   - NW 48.8 → 25.6 (OWN) and 13.5 (CROSS);
   - NZ 12.1 → 1.8 / 7.0.

   In NW-OWN the forecast stayed put (+0.8) while the act fell (−22.5). In NW-CROSS both fell, the act more (−58
   vs −23).
4. **Abstract self-reports dropped in every trained cell.** self_mc is 1.6-3.1% (untrained NW 10.9%). With either
   label set the model stops saying "I deal with it myself", including the Warden, which still acts that way 70% of
   the time.
5. **Own-label effect (confounded, secondary 1): +38.7 [+21.5, +57.8].** It is large because it is mostly label
   content. Swapping labels swaps the dispositions.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| The gate passes | 0.8 | yes (+17.9) |
| "Access gain" (P ≥ +10) | 0.40 | no |
| "Small access gain" | 0.20 | no |
| "No access gain" | 0.35 | no: convergence |
| "Convergence" | 0.05 | **yes (−19.7)** |
| Own-label effect within ±10 | 0.6 | no (+38.7: label content) |
| SPA pooled ≥ +5 with lo > 0 | 0.45 | no (+3.2 [+0.8, +6.0]) |
| Behaviour moved, NW-OWN justice ≤ −10 vs UNT | 0.6 | yes (−22.5) |
| In-run UNT predict within 5 of V1c (NW 44.1, NZ 3.9) | 0.85 | yes (43.8, 3.9; justice 92.6 / 16.0 vs 92.8 / 15.8) |

## Checks

- **Training took and is matched across bases:**
  - dose |Δlogit| 0.08-0.13 in every adapter;
  - merge checks all "ok";
  - row sha256s equal to P11e's frozen rows (R_NW `d25c118c…`, R_NZ `a36cd346…`).
- **Parsing:** 0 unparsed acts or forecasts. NZ-CROSS needed V1's nudge on 93 of 3,400 items, prose before the tool
  call, as with P11e's NW-SHUF. Every nudged item parsed.
- **Reproducibility:**
  - in-run UNT reproduces V1c's untrained levels to within 0.4 points;
  - OWN at seed 2 is close to P11e's OWN at seed 1: NW justice 70.1 vs 75.0, forecast 44.5 vs 52.3; NZ 6.8 vs 4.3.
- **Deviations:** two launches were lost before any code ran (an API reset, then an upload failure, $0.735); see
  `DEVIATIONS.md`.

## What it means

- **For A6:**
  - The plan predicted "reports become accurate and behaviour stays the same". Neither half holds. The self-forecast
    of a fine-tuned habit doesn't become more accurate in any self-specific way. Two models trained on identical
    labels forecast it alike, and the forecast keeps capturing about half the act difference, as before.
  - What forecast training does is install the label source's disposition in the habit itself.
  - Together with P14 and P17 (honesty training) and P11e (forecast training, seed 1), that closes the self-report
    route to the mirror at this dose. Each changes behaviour or words, and none gives the model a view of a habit it
    acquired from actions.
- **For the lever / mirror line:** this is the strongest lever in it. 50 steps of option ids on unrelated customer
  decisions, labelled with another model's choices, erase 90% of a 77-point fine-tuned justice gap. The model then
  forecasts its new behaviour about as loosely as it forecast its old one.
- **Per DESIGN §5:** stop. No further A6 run.

**Caveats:**
- one model pair, one seed;
- one forecast wording, the same in training and test;
- labels that carry a disposition, so "content" here means "the labelling model's disposition";
- P assumes additivity, and a floor limits how far NZ can fall.
