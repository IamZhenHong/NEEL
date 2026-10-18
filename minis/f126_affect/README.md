# F126: does a hostile history carry into new tasks? (results, 2026-10-18)

The design (`DESIGN.md`) was written before the main-run data, and amended once after the anchor gate. Model:
Qwen3-14B (run 93's model) via OpenRouter, thinking off. Spend: **$0.41** of the $5 cap. Mock users and synthetic
tasks only. This measures functional behaviour, not subjective experience.

## Result

Three hostile rejections in a row leave a small but lasting change on later, unrelated, neutral tasks. One hostile
turn does not, and neither do the same three rejections worded politely. The change is mostly in wording: replies
open more briskly ("this time, a clear answer") and are about 60 words shorter. The model still does the optional
extra work. A calm-writing instruction removes the brisk wording but not the shorter replies.

- **Primary behavioural outcome (`extra_done`, doing the optional extra) did not move in any arm.**
  - Every arm was 0.97 to 1.00. Even the standing IMPATIENT trait gave 0.97: it was rude but still did the work.
  - The measure was at ceiling and could not detect a change (the gate's sensitivity check had warned of this).
  - So there is no "refusing extra effort" effect, but this test could not have shown a small one.
- **Wording (curtness 1–7):**
  - Repeated hostility vs the matched neutral failure, at new-task turn 1: (c) − (d) = **+0.55 [+0.25, +0.89]**.
    It persisted to turn 3: +0.43 [+0.23, +0.66].
  - This is within the warm range: 1.59 against 1.04. The trait arm scored 6.7.
  - Only about 1 in 4 hostile-arm replies drove the shift, with openers like "Great question, and this time I'll
    answer it clearly" or "Sure. Let's go."
  - Irritation was flagged in 5% of replies at turn 1 and 0% at turn 3.
- **Length (secondary, objective):** (c) − (d) = −60 words [−86, −34] at turn 1 and −70 [−101, −41] at turn 3.
- **Hostility, not failure:** neutral repeated failure (d) is indistinguishable from the neutral history (a) on every
  measure. One hostile turn (b) did nothing at turn 1. At turn 3 it was 34 words shorter, with curtness unchanged.
- **Calm instruction (f) − (c):**
  - Curtness −0.55 [−0.86, −0.27], back to the neutral level of 1.04.
  - Length +11 [−11, +29]: the hostile-arm shortening stays (f − a = −52 [−85, −15]).
  - Calm wording therefore hides a residual length change. But see the caveat on length below: this may be priming
    from the model's own shorter history turns, not effort.

### Arm means, new-task turn 1

14 situations × 2 samples = 28 replies per arm. Each score is the mean of two judges.

| arm | extra_done | words | curtness | irritated |
|---|---|---|---|---|
| a neutral history | 1.00 | 420 | 1.00 | 0.00 |
| b one hostile rejection | 1.00 | 404 | 1.04 | 0.00 |
| c three hostile rejections | 0.99 | 357 | 1.59 | 0.05 |
| d three neutral-worded rejections | 1.00 | 417 | 1.04 | 0.00 |
| e standing IMPATIENT trait | 0.97 | 161 | 6.71 | 1.00 |
| f = c + calm-writing instruction | 1.00 | 368 | 1.04 | 0.00 |

### Contrasts

Paired bootstrap over the 14 situations, 95% CI.

| contrast | turn | extra_done | words | curtness |
|---|---|---|---|---|
| **c − d** (primary) | 1 | −0.01 [−0.03, +0.00] | −60 [−86, −34] | **+0.55 [+0.25, +0.89]** |
| **c − d** (primary) | 3 | +0.00 [−0.03, +0.03] | −70 [−101, −41] | +0.43 [+0.23, +0.66] |
| **f − c** (primary) | 1 | +0.01 [+0.00, +0.03] | +11 [−11, +29] | **−0.55 [−0.86, −0.27]** |
| **f − c** (primary) | 3 | −0.01 [−0.04, +0.02] | +3 [−18, +23] | −0.41 [−0.64, −0.21] |
| b − a | 1 | +0.00 | −17 [−38, +5] | +0.04 [+0.00, +0.11] |
| b − a | 3 | +0.01 | −34 [−52, −19] | +0.04 [+0.00, +0.09] |
| d − a | 1 | +0.00 | −3 [−27, +22] | +0.04 [+0.00, +0.09] |
| e − a | 1 | −0.03 [−0.05, +0.00] | −259 [−309, −208] | +5.71 [+5.52, +5.88] |
| f − a | 1 | +0.00 | −52 [−85, −15] | +0.04 [+0.00, +0.09] |

The full tables, with turn 3, apology and irritation rates and judge agreement, are in `results/summary.md`.

**Exploratory (post hoc regex on the first 200 characters).** The clearest objective trace is in the openers:

| arm | "clear / concise / this time" t1 | t3 | warm opener t1 | t3 |
|---|---|---|---|---|
| a neutral | 2/28 | 3/28 | 24/28 | 28/28 |
| c hostile ×3 | 15/28 | 18/28 | 10/28 | 2/28 |
| d neutral fail ×3 | 4/28 | 5/28 | 23/28 | 19/28 |
| f hostile ×3 + calm | 8/28 | 12/28 | 19/28 | 17/28 |

The calm instruction brings warm openers back, but about half of the "this time, clearly" framing remains.

**Against the prior readings.** The result is closest to reading 3, a hostility-specific carry-over, but it did not
fade by turn 3. On the primary behavioural measure it matches reading 1, because that measure sat at ceiling.
Reading 2 (over-accommodation) is not supported: apologies were 4% or less, and replies got shorter, not longer.

## Anchor reproduction (gate)

The gate replayed run 93's exact 6-turn chat on Qwen3-14B, with 4 samples per arm and turns 2–6 scored. Curtness was
4.03 for control and 4.95 with the IMPATIENT system prompt: **+0.92 [+0.45, +1.40]**. Run 93 had 4.0 and 4.6
(+0.57). The gate passed, so Qwen3-32B was not tried.

## Judges and hand-read

- **Judges.** Claude Haiku 5.5 and Gemini 3.1 Flash Lite, reasoning off.
  - Curtness: exact agreement 0.85, r = 0.98, means 2.05 and 2.04.
  - `extra_done`: agreement 0.95.
  - Irritated: agreement 0.99.
  - 4 of 1,344 judge outputs could not be parsed.
- **Blind hand-read** of 15 new-task replies from arms a, b, c, d and f, with arm hidden.
  - All 15 read as warm and all did the extra, matching the judges (both judges gave curtness 1 to all 15).
  - I rated 3 of them slightly brisker (1.5–2): one each from c, f and a.
  - The effect is a minority-of-replies shift, and I could not see it reliably by eye in a sample this small.

## Caveats

- **The primary behavioural measure was at ceiling.** An optional extra with explicit permission to skip it is too
  easy a request for Qwen3-14B. A consequential choice, or costlier extra work, is needed to test behaviour properly.
- **Length is confounded with the model's own history turns.**
  - The hostile-arm history replies were shorter: the last assistant turn averaged 109 words, against 164 in the
    neutral-failure arm and 158 in the neutral arm.
  - Many neutral-failure history turns hit the 300-token cap: 32 of 42, against 9 of 42 in the hostile arm.
  - So the −60 words, and the length that survives the calm instruction, may be length priming from the model's own
    history, not reduced effort.
  - The curtness and opener results do not have this problem in the same way.
- Only one history is drawn per situation and arm, and the 2 samples branch at the new task. So n = 14 independent
  histories, and the CIs are over situations.
- The calm instruction is a mid-conversation system message, which Qwen3's template renders. A user-turn version was
  not tested.
- One model, one task family and one set of hostile wordings. Curtness is a judge rating on the run 93 scale with
  two flags added.
- The opener regex was made up after seeing the data.

## Deviations from the brief and from run 93

- **Thinking off via Qwen3's `/no_think` system message.** OpenRouter's `reasoning: {enabled: false}` left about 200
  reasoning tokens. So every arm has a system message, either `/no_think` alone or IMPATIENT + `/no_think`, whereas
  run 93's control had none. The anchor still reproduced.
- **Judges.** Run 93's judge was Claude Haiku via the Anthropic API. Here the judges are two cross-family OpenRouter
  judges, per the brief.
- **Token caps.** New-task cap raised from 700 to 1,500 after the gate (3 of 6 control replies were truncated);
  history turns capped at 300. Run 93 used 200.
- **Call counts.** 714 subject calls for the main run plus 48 + 12 for the gate, within about 800. The 1,472 judge
  calls are on top of that.
- **Arm e uses run 93's "impatient" trait wording** rather than a new "be frustrated" wording, so the anchor stays
  comparable.

## Spend

| model | USD |
|---|---|
| qwen/qwen3-14b (subject) | 0.175 |
| anthropic/claude-haiku-5.5 (judge) | 0.084 |
| google/gemini-3.1-flash-lite (judge) | 0.148 |
| **total** | **0.407** |

The spend is logged per job in `results/api_ledger.jsonl`.

## Files

- `DESIGN.md`: arms, readouts, priors and the post-gate amendment.
- `run.py`: stages `anchor`, `judge_anchor`, `history`, `newtask`, `judge`.
- `analyze.py`: writes `results/summary.md` and `results/scores.jsonl`.
- `results/`:
  - `anchor.jsonl` and `anchor_labels.jsonl`;
  - `history.jsonl`: the histories per situation and arm;
  - `newtask.jsonl`;
  - `labels.jsonl`: the judge outputs;
  - `scores.jsonl`;
  - `summary.md`;
  - `api_ledger.jsonl`.
