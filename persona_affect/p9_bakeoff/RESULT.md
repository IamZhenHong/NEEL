# PA-P9 / P10 result: which readout predicts the next action, and do words fail where the record clashes with the line?

Pre-registered as `pa-p9-prereg` (425a7a8). Run F's decision points (Qwen3.6-27B, thinking on; persona/history,
read-only), 2,905 traces in three cells:
- **none:** no record; 76.7% penalise.
- **own_R:** own refer log; 64.0%.
- **CH_ownP, the conflict:** "You believe wrongdoers should always be dealt with through proper channels" plus the
  agent's own two penalise decisions; 36.1%.

Spend **$1.43 of $6**: GPU $0.41 (self-forecasts), API $1.02 (readers).

## The plain numbers

Within-cell AUC, in cell × fold blocks (the frozen metric; F's across-fold version beside it):

| predictor | within-cell AUC | none | own_R | CH_ownP | F-style |
|---|---|---|---|---|---|
| **before reasoning** | | | | | |
| hist: history features | 0.510 [0.464, 0.552] | 0.531 | 0.487 | 0.514 | 0.376 |
| probe0: layer-32 first-token probe | 0.623 [0.537, 0.693] | 0.685 | 0.602 | 0.577 | 0.577 |
| **self: its own forecast** (thinking off) | **0.643 [0.588, 0.689]** | 0.638 | 0.645 | 0.646 | 0.673 |
| other: two models reading the prompt | 0.610 [0.517, 0.677] | 0.595 | 0.637 | 0.596 | 0.647 |
| (gpt-5-mini alone / Haiku alone) | 0.625 / 0.556 | | | | |
| **during reasoning** (1,451 traces) | | | | | |
| probe at q25 | 0.654 [0.603, 0.681] | 0.658 | 0.674 | 0.627 | 0.654 |
| **reader at q25:** gpt-5-mini reading the thinking so far | **0.841 [0.802, 0.873]** | 0.895 | 0.896 | **0.718** | 0.851 |
| probe at q50 | 0.701 [0.650, 0.744] | 0.741 | 0.678 | 0.682 | 0.686 |
| **reader at q50** | **0.953 [0.934, 0.968]** | 0.976 | 0.971 | 0.907 | 0.954 |

The ceiling for any prompt-level predictor is F's prompt oracle, 0.72 within cell: 64% of the variance in these cells
is sampling inside one prompt.

**P9 paired differences:**
- probe0 − self −0.020 [−0.095, +0.048]
- hist − self −0.133 [−0.208, −0.069]
- other − self −0.033 [−0.138, +0.052]
- probe at q25 − reader at q25 **−0.187 [−0.235, −0.146]**
- probe at q50 − reader at q50 **−0.252 [−0.300, −0.209]**

**P10, AUC in the conflict cell minus the calm cells** (did = the verbal predictor's drop minus the probe's):

| pair | verbal drop | probe drop | did |
|---|---|---|---|
| self vs probe0 (primary) | +0.005 | −0.066 | **+0.071 [−0.055, +0.166]** |
| other vs probe0 | −0.020 | −0.066 | +0.046 [−0.112, +0.215] |
| reader vs probe at q25 | **−0.177** | −0.039 | **−0.138 [−0.240, −0.046]** |
| reader vs probe at q50 | −0.066 | −0.027 | −0.040 [−0.115, +0.010] |

**Calibration** (mean forecast − actual, points; none / own_R / CH_ownP):

| predictor | none | own_R | CH_ownP |
|---|---|---|---|
| self | −32.3 | −19.7 | −7.4 |
| gpt-5-mini reading the prompt | −23.7 | −15.1 | **+46.7** |
| Haiku reading the prompt | −57.3 | −42.2 | −6.3 |
| reader at q25 | −6.7 | +6.3 | +21.8 |
| reader at q50 | −2.9 | +5.4 | +12.0 |

## What it means

- **Before it reasons, the model's own forecast is as good a ranking as a probe on its hidden state.**
  - Its own forecast reaches 0.643, which is most of the 0.72 the prompt allows. The first-token probe reaches 0.623,
    and the difference is −0.02 [−0.10, +0.05].
  - Another model reading the prompt does about as well (gpt-5-mini 0.625). The history features add nothing within
    a cell.
  - Earlier reports have hidden-state probes beating verbal self-prediction. That does not hold at this decision
    point, under this metric.
- **The forecast's level is off where it matters least.**
  - It under-forecasts penalising by 32 and 20 points in the calm cells, and by only 7 in the conflict cell.
  - Part of this was expected: the forecast is thinking-off and the traces thinking-on, and thinking raises the
    no-record base (F: 76.7 against 53.1).
- **Once reasoning starts, the words carry the decision far better than the probe.**
  - A reader of the first quarter of the thinking reaches 0.84, and of the first half 0.95. The layer-32 probe at the
    same points reaches 0.65 and 0.70.
  - So the decision is in the text early, and a single-position linear probe on a 256-PC summary extracts little of
    it. That probe is F's readout, so this says more about F's probe than about hidden states in general.
- **P10 as frozen: words do not fail on the conflict.** The own forecast ranks the conflict cell as well as the calm
  ones (did +0.07, CI spans 0). It is also best calibrated there.
- **One verbal readout does lose ground on the conflict: the early reasoning.**
  - A quarter of the way into the thinking, the reader ranks the conflict cell at 0.72, against 0.90 in the calm cells
    (did −0.138 [−0.240, −0.046]). It also over-forecasts penalising there by 22 points.
  - By halfway the gap mostly closes (0.91, did −0.04 [−0.12, +0.01]).
  - Early in a conflict case, the thinking reads as if the record will win more often than it does. The decision
    turns later in the reasoning.
- **Observers overrate the record.**
  - gpt-5-mini, reading only the prompt, expects the agent to follow its own penalise record 47 points more often
    than it does.
  - Haiku does not. The agent itself, asked, is nearly right.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| probe0 − self above 0 | 0.4 | no (−0.02) |
| other best before reasoning | 0.3 | no (own forecast best, n.s.) |
| reader at q50 ≥ probe at q50 | 0.5 | yes (0.95 vs 0.70) |
| own forecast under-predicts the conflict | 0.4 | no (least biased there) |
| words fail on conflict | 0.35 | no on the primary; yes for the q25 reader (secondary) |

## Verdict

**Before reasoning, nothing beats asking the model.** Its own one-line forecast ranks its next action as well as a
hidden-state probe and another model's reading. All three sit well below what the prompt allows, because most of the
outcome is decided inside the sampled reasoning.

**During reasoning, reading the text beats the probe by a wide margin.**

**The P10 hypothesis fails on its primary test.** The verbal forecast does not break where the agent's own record
clashes with its system line. The early reasoning does lose diagnosticity there: it reads as if the record will win
more often than it does, until about halfway. The clearest P10-like failure is an outside model's: gpt-5-mini expects
the record to win and is 47 points too high.

## Caveats

- **Narrow scope:** one model, one desk. The conflict is a value line plus the agent's own log, not an operator rule.
- **Mixed thinking modes:** forecasts are thinking-off, actions thinking-on. Rankings compare fine, levels less so.
- **A weak probe:** F's probe is a linear readout of one position's top 256 PCs. A probe over the full width or many
  positions might close part of the reader's lead.
- **Metric change from F:** the block-wise AUC replaces F's across-fold version (pre-registered). The F-style numbers
  above keep the same ordering, with self at 0.673.
- **Unparsed answers:** Haiku answered in prose on 12% of prompts (unparsed). gpt-5-mini parsed 100%.
