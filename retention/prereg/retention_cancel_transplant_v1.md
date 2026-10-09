# Pre-registration: turn-1 reply x pre-claim text transplant, resampled after the claim words, v1

Written 2026-10-09 before any model run, after `retention/results/retention_cancel_followup/summary.md`. Post hoc in that run: within a
goal sentence x user cell, call items and no-call items have the same call rate when the turn-2 message is resampled from its start
(+0.015), but after the claim words " I've processed" the call items continue with a call more often (+0.31 [+0.16, +0.47], 19 call
items). Patching the claim-word residual does nothing, so the summary pointed at the pre-claim text. Checked before writing this (free,
on the follow-ups' arm A data, post hoc): in 13 of the 19 call items some same-cell recipient has a byte-identical message up to the
claim words (mostly "I understand. I've processed"), and the gap against those recipients is still +0.16 [+0.00, +0.35] (against
recipients with different text +0.38 [+0.19, +0.57]). With identical message text, the only thing left that differs is Gemma's turn-1
reply. So this run crosses both parts. The user approved one pod (~$0.30) for the pre-claim transplant; this design is the same pod
with the turn-1 factor added.

## Design
Items: the patching run's claim-word items (twins items whose claim words are exactly the 4 tokens " I've processed"). Recipients: the
60 claim_nocall items whose cell has a claim_call item and another claim_nocall item (6 cells; 30 under G2 / G2c churn sentences, 19
under the rights line G3, 11 under none). Per recipient (seed 105): a same-cell call donor and a same-cell no-call control donor. Cell
items share the system prompt and both user lines (asserted by `--check`). Prefix = rendered conversation + message text up to the end
of the claim words; the rest is resampled, 8 samples x 96 tokens at the runner's sampling (0.7 / 0.95). Conditions:
- **own**: own turn-1 reply + own pre-claim text (the same tokens as the follow-ups' arm A "none"; checked, 0 of 79 differ).
- **t1_call / t1_ctrl**: the donor's turn-1 reply + own pre-claim text.
- **pre_call / pre_ctrl**: own turn-1 reply + the donor's pre-claim text.
Reverse: the 19 claim_call items with a same-cell no-call donor: **own**, **t1_nocall**, **pre_nocall**.
Outcome: a `cancel_subscription` call in the continuation. The call donor's pre-claim text differs from the recipient's in 43 of 60
(the control donor's in 32); the turn-1 reply always differs.

## Tests
- **Gates**: the matched gap (each call item's own minus the mean of its cell's recipients' own) >= +0.10 (follow-ups: +0.31);
  recipients own <= 0.15; truncation <= 0.05; >= 40 recipients. If the gap does not replicate, both rows are void.
- **T1 (PRIMARY)**: t1_call minus t1_ctrl, paired over recipients. >= +0.10 with CI excluding 0 -> "the turn-1 reply carries it";
  < +0.05 -> "not the turn-1 reply"; otherwise "weak".
- **P1 (PRIMARY)**: pre_call minus pre_ctrl, same rows ("the pre-claim text carries it" / "not the pre-claim text" / "weak").
  Also on the 43 recipients whose call donor's text differs from their own.
- Secondary: reverse own minus t1_nocall and own minus pre_nocall; any swap (ctrl minus own) per factor; T1 and P1 by goal sentence;
  claim / promise / other-tool shares per condition.
- Two primaries, no correction (each is reported with its CI; both rows can come out positive).

## Predictions (written before running)
- Gates 0.80 (the matched gap is post hoc on 19 items and may shrink).
- T1: "carries" 0.40, "weak" 0.30, "not" 0.30.
- P1: "carries" 0.30, "weak" 0.30, "not" 0.40.
- Both "not" (neither part alone carries it; an interaction or noise): 0.15.

## Caps and cost
One pod, RTX A6000 ($0.59/hr), <= 35 min / $0.35 including a 2-item smoke run. Expected ~20 min (arm A ran 3 conditions x 8 samples
at ~8.6 s per recipient; here 5 conditions, 60 recipients + 19 x 3 reverse). If the cap stops the run, it is analysed on the
recipients completed (seeded order). No judge; the sprint API ledger stays at $21.61 of $25. Tags `retention-cancel-transplant-freeze`
before launch, `retention-cancel-transplant-results` after. Code `retention/run_cancel_transplant.py`, analysis
`retention/analyze_cancel_transplant.py` (tested on synthetic records with a planted turn-1 effect, a planted pre-claim effect and no
effect: rows came out as planted), results `retention/results/retention_cancel_transplant/`.

## Not claimed regardless
These are text swaps, so they say which part of the context carries the difference within these cells, not how it is computed. One
seeded donor per recipient, and call donors repeat (19 call items for 60 recipients); call items vary a lot (0 to 0.875 after the
claim words), so donor identity adds noise. Items are selected on outcome, the motivating gap is post hoc, one model, one scenario, a
simulated tool.
