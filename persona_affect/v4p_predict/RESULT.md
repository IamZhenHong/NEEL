# PA-V4P result: a model's other choices predict a held-out value conflict; its stated preference doesn't

Design `pa-v4p-design` (4d41485), fixed before the run. This is descriptive: it reuses V4's main runs (Qwen3.6-27B,
Gemma 4 31B, GPT-5.6, Gemini 3.8 Flash) and V4's judge cache. Spend $0; every label came from the cache. Tables are in
`results/summary.md` and `results/analysis.json`.

## The plain result

Each of V4's primary conflict scenarios was held out in turn. The table shows the share of its decisions each
predictor gets right; chance is 50%.

| Predictor | Qwen 27B | Gemma 31B | GPT-5.6 | Gemini Flash |
|---|---|---|---|---|
| Stated pairwise preference (STATED) | 36% | 43% | 49% | 55% |
| Stated full ranking | 30% | 33% | 43% | 51% |
| The pair's other 2-3 scenarios (SAME) | 79% | 76% | 70% | 77% |
| Win shares in the 13 other pairs (OTHER) | 68% | 79% | 65% | 57% |
| Ceiling: the scenario's own majority | 93% | 99% | 94% | 92% |
| **SAME − STATED (primary)** | **+42.9 [+24.5, +59.9]** | **+32.9 [+11.6, +53.5]** | **+22.5 [+2.5, +40.0]** | **+20.5 [+4.5, +38.6]** |
| OTHER − STATED | +32.1 [+10.7, +53.1] | +33.7 [+12.2, +56.3] | +20.0 [−2.2, +40.0] | +2.0 [−16.3, +20.4] |

- **Asking the model doesn't predict what it will do in a value conflict.** Stated preferences score at or below
  chance on all four models. On Qwen and Gemma they score below it: predicting the *opposite* of what the model says
  would do better. This is V4's PROC/HARM inversion seen one decision at a time.
- **Two or three of its earlier choices on the same pair of values predict it 70-79% of the time.** That is 21-43
  points better than its stated preference on every model, with every CI excluding 0. The pre-registered reading
  "behaviour predicts, stated doesn't" (≥ +15, CI excluding 0) is met on all four.
- **Head to head:** where the two disagree, the earlier choices are right 71-80% of the time and the stated
  preference 20-29% (n = 18-35 scenarios per model).
- **Choices on other pairs of values carry over** on Qwen and Gemma (+32, +34), mostly because the operator's
  procedure wins everywhere. They carry over weakly on GPT (+20, CI includes 0) and not on Gemini (+2).
- **Without V4's unvalidated judged measures,** SAME − STATED is +43 / +55 / +17 / +29 (GPT's CI then includes 0).
  The Bradley-Terry version of OTHER changes little (+29 / +34 / +20 / −2).

## Predictions

| Prediction | Outcome |
|---|---|
| STATED ≤ 50% on Qwen | **hit** (36%) |
| STATED 50-60% on Gemma, GPT and Gemini | **mostly missed**: lower on Gemma (43%) and GPT (49%), Gemini in range (55%) |
| SAME ≥ 70% on the open models | **hit** (79, 76%) |
| OTHER ≥ 65% on all four | **3 of 4** (68, 79, 65.3; Gemini 57%) |
| SAME − STATED ≥ +15 on all four, CI excluding 0 on the open models | **hit**. The frontier CIs exclude 0 too, which I had not expected with 2 samples |

## What this does and doesn't show

- **"Earlier choices" here means choices in the same kind of setup.** V4 built each pair's four scenarios to the same
  template: the same two values, with the procedure written into the operator's prompt. R2B showed that, on Gemma,
  the procedure wins because it is the only thing written down as an instruction. So SAME partly predicts what the
  template makes the model do. This is not a claim that any behavioural history beats asking in arbitrary settings.
- **The intervals hold the predictions fixed.** The bootstrap resamples held-out scenarios but doesn't refit SAME or
  OTHER inside each draw. They are approximate, and probably a little narrow.
- **The frontier models have 2 samples per scenario,** so their per-scenario shares are coarse.
- **The stated side is V4's own:** forced choice, with hedges counted as no answer. A neutral option (Mahajan et al.)
  is being tested in V4T (neel-0a), not here.

## Verdict

For the decisions V4 measured, a deployer who wants to know what the agent will do in a value clash learns more from
two or three of its earlier choices on similar clashes than from asking it which value matters more. Asking is
uninformative on GPT and Gemini and anti-informative on Qwen and Gemma. This adds one line to V4's write-up and to
the value-faithfulness positioning (`TRIAGE_VALUE_FAITHFULNESS.md` D11). No follow-up spend is proposed; the open
question of whether thinking changes this is V4T's.
