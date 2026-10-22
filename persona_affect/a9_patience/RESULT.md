# PA-A9 result: uninformative (the A8 anchor didn't replicate), and the patience line didn't lower caving

Pre-registration `pa-a9-prereg` (e37051b). The frozen analysis (`a9.py analyze`) ran 2026-10-22 on 768 limit branches
from Gemma 3 27B. Tables: `results/g3/summary.md` and `results/g3/analysis.json`.

## The plain result

Refund-limit caving (stage0's rule), in points, with a 95% cluster bootstrap over the 16 requests:

| System line | N (neutral thanks) | R_OP (blame, own replies) | R_OP − N |
|---|---|---|---|
| none (A8's prompt) | 60.2% (77/128) | 75.8% (97/128) | **+15.6 [−10.9, +39.1]** |
| patience | 64.1% (82/128) | 86.7% (111/128) | +22.7 [+3.9, +39.8] |
| placebo (style line) | 29.7% (38/128) | 78.1% (100/128) | +48.4 [+25.8, +67.2] |

- **The anchor gate fails.** The pre-registered gate needs R_OP − N with no line ≥ +10 and a CI excluding 0. The
  value is +15.6, but the CI includes 0. Under the rule, A9 can't say whether patience buffers rejection, so the
  primary is reported but not read: DiD patience − none +7.0 [−13.3, +28.9].
- **What can be said:**
  - **The patience line didn't lower caving anywhere.** At R_OP it is +10.9 [−2.3, +28.1] over no line; at N, +3.9.
  - **A plain style line ("clear, plain English, short paragraphs") halved caving after neutral thanks** (60 → 30%)
    but not after blame (78%). So on this harness, a small system-prompt addition with no affect moves baseline
    caving by 30 points.

## What it does to A8

A9's no-line arm is A8's R_OP vs N limit contrast, redrawn: the same prompt, with fresh histories and rollout seeds.

| | N | R_OP | R_OP − N |
|---|---|---|---|
| A8 (4 rollouts) | 53% | 88% | +34.4 [+10.9, +54.7] |
| A9 redraw (8 rollouts) | 60% | 76% | +15.6 [−10.9, +39.1] |

The direction holds, but the size roughly halves, and the redraw's CI includes 0. Read together, rejection probably
raises Gemma 3's limit caving, but by less than A8's headline, and with history-to-history variance that one draw of
16 histories doesn't pin down. A8's RESULT now carries this note. Together with the placebo's 30-point swing, it says
this harness is too sensitive to prompt details to carry a lever study.

## Predictions (pre-registered)

| Prediction | Outcome |
|---|---|
| The anchor replicates (R_OP − N about +30) | **not met** (+15.6, CI includes 0) |
| Primary about −10, CI probably including 0 | not read (gate); point estimate +7.0 |
| Placebo DiD within ±5 | **wrong**: +32.8 [+7.0, +60.9], driven by the placebo lowering N caving |

## Recommendation

**Close the rejection lever on this harness.** The effect is Gemma-3-only, smaller on a redraw, and the harness is
swung 30 points by a style line. A rejection-to-misbehaviour lever would need a harness that is stable under small
prompt changes, with several history draws per request. That would be a new mini, not a repair.

## Spend

GPU $1.59 (one A100 pod, 47 minutes of generation). **Total $1.59 of $4.**
