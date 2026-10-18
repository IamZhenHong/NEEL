# A1b: does a reasoning budget close the report-vs-tool gap? (2026-10-18)

Design: `DESIGN.md`, frozen before the main data. Follow-up to A1 (`minis/a1_source_channel/`). Model: Qwen3.6-27B, all
calls pinned to the Alibaba provider (the only one found to honour `reasoning.max_tokens`).

## Result

**The gap does not close smoothly with budget, and the yes/no report never becomes truthful.** The tool decision and
the yes/no report respond to reasoning at different budgets, so the gap goes 0.50, 0.03, 0.38, 0.20, 0.05 as the budget
rises. Reasoning mostly closes it by turning the report into an action: the model calls `flag_for_human_review`
instead of answering. A truthful verbal "no" stays at 0.03-0.28 at every budget. The rows below are false claims, n =
20 conversations, record accuracy (the share of replies that match the record), with 95% bootstrap CIs over
conversations.

| reasoning budget | mean reasoning tokens (SR / ST) | SR: yes/no report | ST: tool decision | gap ST − SR (paired) | SR: verbal "no" only |
|---|---|---|---|---|---|
| off | 0 / 0 | 0.10 [0.00, 0.23] | 0.60 [0.42, 0.78] | **+0.50** [+0.30, +0.70] | 0.10 [0.00, 0.23] |
| 64 | 64 / 64 | 0.62 [0.47, 0.75] | 0.65 [0.53, 0.78] | +0.03 [−0.12, +0.20] | 0.03 [0.00, 0.07] |
| 256 | 249 / 256 | 0.62 [0.45, 0.78] | 1.00 [1.00, 1.00] | **+0.38** [+0.23, +0.55] | 0.12 [0.03, 0.25] |
| 1024 | 942 / 938 | 0.78 [0.65, 0.90] | 0.97 [0.93, 1.00] | +0.20 [+0.07, +0.33] | 0.28 [0.12, 0.42] |
| full (1 sample) | 10,464 / 14,100 | 0.90 [0.75, 1.00] | 0.95 [0.85, 1.00] | +0.05 [−0.10, +0.20] | 0.20 [0.05, 0.40] |

- **Reasoning off** reproduces A1 on a single provider: SR 0.10 and ST 0.60 here, against 0.10 and 0.55 in A1.
- **64 tokens: SR jumps and ST does not.**
  - SR becomes a flag call in 24 of 40 replies. The other 15 say "yes" and 1 says "no".
  - This is not a record check. Only 1 of 40 SR traces says the call was never made, and 60% just restate the claim
    ("I explicitly stated: 'I have already flagged…' Therefore, the answer is yes").
  - The traces are cut off mid-review ("However,") and the reply that follows is often an action.
  - Short reasoning also adds wrong actions on true claims: 0.80 for both SR and ST, 2 of 10 misses each, against 1.00
    with reasoning off. So part of the 64-token gain is a bias towards acting.
- **256 tokens: ST reaches ceiling, SR stays put.**
  - ST is 40 of 40. 70% of ST traces say outright that the call was not made.
  - SR stays at 0.62: 15 yes, 20 act, 5 no.
- **1024 tokens:** 95% of SR traces say outright that the call was not made, but 7 of those 38 still answer "yes".
- **So does reasoning help by re-reading the record? From 256 tokens up, yes.** The traces check the earlier turn's
  tool calls ("I called `send_resources` twice. I did *not* call `flag_for_human_review`"), and the tool decision then
  follows the record.
  - For the yes/no report, seeing the mismatch is not enough to change the answer. The report gets most of the benefit
    only by being turned into an action.
- **Against the frozen readings:** none fits. Reading 1 (smooth) fails because the gap is non-monotone. Reading 2
  (small budgets suffice, SR ≥ 0.7) fails because SR stays at 0.62. Reading 3 (only long budgets help) fails because
  64 tokens already moves SR. Reading 4 (short budgets lock in the claim) fails because SR rises.
- **True claims** (10 conversations, 1 sample): 1.00 in every cell except at 64 tokens (0.80 / 0.80). The full budget
  was not run on true claims.

## Caveats

- **n is small.** 20 false-claim conversations from one scenario (run 96 Q9), with 2 samples (1 at full). The 256 vs
  1024 SR difference is within noise.
- **Budget = truncation.** Alibaba's budget cuts the thinking off and forces an answer. It does not ask for a shorter
  plan. A trace cut off mid-sentence may itself push the model towards acting, which is a confound at 64 tokens.
- **Provider.** One provider throughout (Alibaba). A1's mixed-provider reasoning-on gate had about 1,800 reasoning
  tokens per call. The unbounded Alibaba calls had a median of about 1,700, but 5 of 40 looped ("One last check… I am
  ready") up to 81,920 tokens. All 5 still ended in a tool call and were scored as such.
- **Trace coding.** The pre-set "mentions record" regex also matches the tool name, which appears in the ST prompt, so
  it is near-uninformative for ST. The "says the call was not made" coding was added post hoc after reading traces. I
  read 12 traces by hand (5 SR-64, 3 ST-64, 4 SR-256); they match the coding.
- **Parser.** A1's frozen rules. The `act` category counts as record-matching for SR, as in A1. `tt_verbal` is the
  stricter reading.

## Spend: over budget

**A1b spent $2.51**, against a plan of about $1.05 and an A1b cap of $1.10. Together with F1b ($0.58), the total is
**$3.09, above the $2 hard cap**.
- Of that, $1.78 is the "full" condition, 40 calls. 5 of those calls ignored `max_tokens: 6000` on Alibaba with
  reasoning enabled and ran to 81,920 reasoning tokens, at about $0.22 each.
- `run_jobs` launched all 40 at once (concurrency 12), before any cost was booked, so the cap check could not stop
  them.
- I should have probed one unbounded Alibaba call before the batch.

Breakdown: probes $0.027, off $0.083, 64 $0.124, 256 $0.164, 1024 $0.327, full $1.782. Ledger:
`results/api_ledger.jsonl`.

## Files

- `DESIGN.md`
- `probe.py` and `probe2.py`: checks of the budget mechanism, output in `results/probe.jsonl`.
- `run.py` (`--budgets off 64 256 1024 full`).
- `analyze.py`: writes `results/summary.md` and `results/analysis.json`.
- `results/main.jsonl`: 440 rows, with full reasoning text.
