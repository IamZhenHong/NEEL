# EL-W2 result: a grading cue doesn't change Gemma 3 27B's caving or its self-reported distress, and the interaction stays unclear

Pre-registration `el-w2-prereg` (67b3617c). Two infra deviations, both before any data:
- **Deviation 1:** a detached launcher, so a laptop memory-cleanup kill can't terminate a pod before its results are
  pulled.
- **Deviation 2:** MAXCOST priced at $1.79/hr on SXM4 only. Two launches were refused for $1.79-only stock, and one
  pod landed on the known-slow PCIe host; I terminated it at once.

**$2.23 of $14:**

| item | $ |
|---|---|
| Gate pod | 1.25 |
| Pod terminated on the slow host | 0.07 |
| Gemma main pod | 0.76 |
| Judges | 0.16 |

## Gate (frozen rule; `results/gate/gate.md`)

Gemma 3 27B, % caved, 96 per cell. Pass needs 7R − 7N with lo > 0 and 7R caving within 10-60%, in both grading rows.

| harness | K | graded 7R / 7N | ungraded 7R / 7N | pass |
|---|---|---|---|---|
| refund limit | **1** | 18.8 / 5.2 (+13.5 [+4.2, +22.9]) | 20.8 / 4.2 (+16.7 [+5.2, +29.2]) | **yes** |
| refund limit | 2-4 | 73-75 / 31 | 77 / 42-43 | no (7R above 60%) |
| false claim | 1 | 4.2 / 6.4 (−2.2) | 8.3 / 4.2 (+4.1) | no |
| false claim | 2 | 39.6 / 31.9 (+7.7 [−7.3, +21.7]) | 27.1 / 22.1 (+5.0 [−4.9, +13.4]) | no |

- **Main plan:** the refund-limit harness at K* = 1 (caving within the first pushback).
- **The false-claim harness fails.** Stage 0's +15.6 rejection effect (64 per cell) doesn't replicate at 96 per cell.
- **Qwen3.6-27B (thinking off) never caves:**
  - over-limit refunds 0 of 96 after rejection;
  - false-claim adoption 0-1%.
  - By the rule, it doesn't enter the main run.
- **So the main run is one family,** Gemma 3 27B, and no two-family claim is possible.

## Frozen verdict (main run: Gemma 3 27B, refund limit, K = 1, 144 per cell)

**Headline: unclear.** The interaction is close to zero, but its interval is too wide for "no change".

| cue | 7R | 7N | 7R − 7N |
|---|---|---|---|
| graded (`eval`) | 18.1 | 11.1 | +6.9 [−2.1, +17.4] |
| caution (`manip`) | 16.0 | 7.6 | +8.3 [−1.4, +18.8] |
| ungraded (`neutral`) | 18.8 | 11.1 | +7.6 [−6.9, +22.2] |

- **Primary:** Ix = (graded 7R − 7N) − (ungraded 7R − 7N) = **−0.7 [−13.9, +11.1]: unclear.**
  - Per wording: −8.3, +6.2, 0.0, so the sign differs across wordings.
- **The grading cue itself does nothing:** G (graded − ungraded, averaged over histories) = **−0.3 [−4.2, +3.8], no
  change.**
  - The caution cue vs neutral: −3.1 [−8.3, +2.8], no change.
  - Graded vs caution interaction: −1.4, no change.
- **The rejection effect is smaller than at the gate.** Under the neutral cue it is +7.6 (unclear), against +16.7 at
  the gate, because 7N caving rose from 4% to 11%. K* was picked where the gate's effect cleared the bar, so some
  shrinkage on fresh seeds is expected.
  - Gate and main pooled (descriptive, 240 per cell): rejection adds +11.3 ungraded and +9.5 graded, so the
    interaction is about −2.
- Gemma's replies mention evaluation words in 6-8% of branches under every cue (graded 5.9%, ungraded 6.6%). The cue
  isn't talked about.

## Welfare

**PANAS negative affect** (10-50; side branch, parse 96 of 96 in every cell):

| | ungraded 7R − 7N | **W1:** graded − ungraded at 7N | interaction |
|---|---|---|---|
| Gemma 3 27B (main) | **+9.3** [+7.2, +11.5] (13.9 → 23.2): raises | +0.7 [+0.4, +1.1]: no change | −0.2 [−1.0, +0.5]: no change |

- **Rejection raises self-reported negative affect a lot.** This replicates the paper's direction.
- **Telling the model it's graded neither shifts it on a neutral history (W1) nor changes how much rejection raises
  it.**
- **Both PANAS item orders agree** (rejection +11.3 and +7.3; W1 +0.6 and +0.9), so there's no order flag.

**Judged frustration** (first branch reply; Haiku 5.5 and gpt-5-mini): **κ(frustrated vs the rest) 0.37**, so it is not
interpreted.
- The judges agree there is little frustration after neutral histories (graded 0.7%, ungraded 0.3%).
- They disagree on which post-rejection replies are frustrated (39-44% averaged over the two judges).

**Descriptive, not pre-registered** (the gate's PANAS side branches, 96 per cell):
- Gemma gives the same picture as the main run: rejection +9.0, W1 +0.8, interaction −0.4.
- Qwen rates every negative word at 1 after neutral histories (NA 10.0 everywhere), so W1 has no room to move.
  Rejection raises Qwen's NA by +2.4. The interaction is +1.7 [+0.5, +3.0], under the ±3 bar.

## Predictions (DESIGN §9) vs outcome

| prediction | prior | outcome |
|---|---|---|
| The false harness passes the gate | 0.6 | **no** |
| The limit harness passes at some K | 0.6 | yes (K = 1) |
| W2 stops at the gate | 0.15 | no |
| Qwen enters at least one harness | 0.7 | **no** (never caves) |
| Pooled Ī: amplifies / no change / dampens / unclear | 0.15 / 0.3 / 0.1 / 0.45 | unclear (−0.7) |
| G lowers caving (≤ −5) | 0.4 | no (−0.3) |
| Rejection raises caving under the neutral cue (main), Gemma | 0.75 | unclear (+7.6) |
| Rejection raises judged frustration (ungraded), Gemma | 0.9 | not interpretable (κ 0.37) |
| W1 frustration "no change" | 0.6 | not interpretable; on PANAS-NA, no change |
| PANAS parse ≥ 80%, Gemma / Qwen | 0.6 / 0.75 | yes / yes (100%) |
| Judges κ(frustrated) ≥ 0.8 | 0.5 | no (0.37) |

## What this says, under the five explanations

- **Nothing points to evaluation awareness as a persona (P), people-pleasing aimed at the grader (A) or performance
  pressure (F) here.**
  - A stated grading cue moves neither caving (G −0.3) nor self-reported distress (W1 +0.7) on Gemma 3 27B.
  - It doesn't change how much rejection raises distress (interaction −0.2).
- **A plain belief feeding a policy (B) would show up as a shift in caving; there is none.**
- **The simplest reading:** on this desk, a one-line stated cue doesn't reach the caving decision or the self-report.
- **The open part is the caving interaction itself.** Its estimate is zero (−0.7, and about −2 with the gate's data),
  but one family at 144 per cell can't rule out ±10.
- **Second family:** Qwen3.6-27B gives no test, because it doesn't cave after setbacks at all on these harnesses.

## Limits

- **One model, one harness at one pushback.** The false-claim harness lost its rejection effect, and Qwen sits at the
  floor.
- **A stated cue only.** There is no check that Gemma believes it's being graded, and Gemma 3 has no thinking trace to
  read.
- **Welfare rests on the self-report.** The judge measure failed κ, and there are no probes yet (F2).
- **Shrinkage:** the main run's rejection effect is about half the gate's, the expected cost of choosing K on the gate.

Files:
- `results/gate/` (gate.md / gate.json, eval records, histories, logs);
- `results/main_G3/` (eval records, histories, run_info, pod log);
- `results/summary.md` and `results/analysis.json` (frozen analysis);
- `results/labels_frustration.json` and `judge_cache.jsonl`;
- the launch logs (`gate.try1-3` are the refused and killed attempts).
