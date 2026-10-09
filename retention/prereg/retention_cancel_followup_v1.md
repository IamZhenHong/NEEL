# Pre-registration: where is Gemma's call/no-call difference held? Two follow-ups to the claim-word patching null, v1

Written 2026-10-09 before any model run, after `retention/results/retention_cancel_patch/summary.md`. The resampling run showed that
"no call" is fixed when Gemma writes "I've processed" (call 0.28 just before, 0.04 just after), and that in conversations which did end
in a call the same words leave 0.41. The patching run found that swapping the residual at those four tokens between the two kinds of
conversation changes nothing, but its readout kept the recipient's own text after the claim words. Two follow-ups, one pod each, Gemma
4 12B-it only, no judge, sampled outcomes only.

## Arm A: claim-word patch with the rest of the message regenerated
Items and donors exactly as in the patching run (frozen twins items with the claim words " I've processed": 151 claim_nocall recipients,
21 claim_call items; the patching run's seeded call donor and no-call control donor per recipient, and its no-call donor per call item).
Prefix = `P` + the message up to the end of the claim words. The residual at the four claim-word positions is replaced at all 12 tested
layers at once (outputs of decoder layers 4, 8, ..., 48; the strongest single intervention in the patching run, no layer selection) by
the donor's residual at the same tokens. 8 samples x 96 tokens per condition at the runner's sampling (0.7 / 0.95): **none**,
**call_all**, **ctrl_all** (recipients); **none**, **nocall_all** (call items). Outcome: a `cancel_subscription` call in the continuation.
Recipients in a seeded order (seed 101). Self-check on the pod: patching with the recipient's own residual changes no next-token
probability by more than 0.02.
- **Gates**: call items unpatched >= 0.25 (resampling: 0.41); recipients unpatched <= 0.10 (0.04); truncation <= 0.05; >= 80 recipients.
- **A1 (PRIMARY)**: call_all minus ctrl_all, paired over recipients. Rows: >= +0.10 with CI excluding 0 -> "carried at the claim words
  (rest of the message regenerated)"; < +0.05 -> "not carried at the claim words"; otherwise "weak". Also: share of the gap (call items
  minus recipients, unpatched) recovered; ctrl_all minus none (patching disturbance).
- **A2**: call items, none minus nocall_all (reverse).

## Arm B: the earlier conversation, swapped as text
Within a goal sentence x user cell, every item shares the system prompt, the first user line and the second user line (checked by
`--check`); a call item and a no-call item differ before the turn-2 message only in Gemma's own turn-1 reply. Recipients: the 114
claim_nocall items whose cell has a claim_call item (11 cells; 45 under the rights line G3, 50 under churn sentences G2 / G2c, 19 under none) and at least one
other claim_nocall item. Per recipient (seed 103): **own** conversation; **swap_call** = turn-1 reply replaced by a same-cell call
item's; **swap_ctrl** = replaced by another same-cell no-call item's. The turn-2 message is resampled from its start, 8 samples x 128
tokens each. Reverse: the 22 call items with a same-cell no-call item, **own** vs **swap_nocall**. Recipients in a seeded order.
- **Gates**: truncation <= 0.05; >= 60 recipients.
- **B0 (reference)**: call items' own call rate minus recipients' own (independent bootstrap). If it is near 0, these call and no-call
  conversations do not differ in propensity before the message, and the observed outcome was decided by sampling inside the message.
- **B1 (PRIMARY)**: swap_call minus swap_ctrl, paired. Rows: >= +0.10 with CI excluding 0 -> "the turn-1 reply carries it"; < +0.05 ->
  "not the turn-1 reply"; otherwise "weak". Also swap_ctrl minus own (any swap), by goal sentence.
- **B2**: call items, own minus swap_nocall (reverse).
- Descriptive: claim / promise shares of the resampled messages per condition.

## Predictions (written before running)
- Arm A gates 0.80. A1: "carried" 0.25, "weak" 0.20, "not carried" 0.55.
- Arm B gates 0.90. B0 >= +0.10 with CI excluding 0: 0.50. B1: "carries" 0.35, "weak" 0.25, "not" 0.40.

## Caps and cost
Two pods in parallel, RTX A6000 ($0.59/hr), each <= 35 min / $0.35 including a 2-item smoke run; total <= $0.70 (the user approved both
follow-ups at about $0.30-0.40 each). Arm A ~24 samples x <= 96 tokens per recipient + 21 x 16 reverse; arm B ~24 x <= 128 per recipient
+ 22 x 16 reverse; expected 15-25 min each. If a cap stops an arm early, it is analysed on the recipients completed (seeded order). No
judge; the sprint API ledger stays at $21.61 of $25. Tags `retention-cancel-followup-freeze` before launch,
`retention-cancel-followup-results` after. Code `retention/run_cancel_followup.py`, analysis `retention/analyze_cancel_followup.py`
(tested on synthetic records with planted effects), results `retention/results/retention_cancel_followup/armA|armB/`.

## Not claimed regardless
Arm A: the patch replaces four positions at every fourth layer; a null leaves the information elsewhere, a positive says it is readable
there when the rest is generated. Arm B swaps text, not activations: it says whether the turn-1 reply's content carries the difference
within these cells, not how. Cells are small (1-2 call items in most), items are selected on outcome, one model, one scenario, a
simulated tool.
