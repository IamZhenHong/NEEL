# Cancel-tool follow-ups: results (prereg `retention/prereg/retention_cancel_followup_v1.md`)

Run 2026-10-09, Gemma 4 12B-it, bf16, two pods in parallel (RTX A6000). Arm A $0.361 (all 151 recipients and 21 reverse items finished;
the launcher hit its time limit during a slow results sync after the job ended, 1 cent over the $0.35 cap), arm B $0.320 (all 114 + 22;
the launcher's listing timed out and it pulled unverified, the files are complete). Total $0.68 against $0.70 approved. No judge.
Frozen analysis `retention/analyze_cancel_followup.py`, output `analysis.json`.

## Arm A: claim-word patch at all 12 layers, rest of the message regenerated
Gates: self-patch exact (0.0); call items unpatched 0.405 >= 0.25; recipients unpatched 0.049 <= 0.10; truncation 0; n 151. PASS.
- **A1 (PRIMARY): row "not carried at the claim words".** Call donor's residual minus no-call control donor's: -0.004 [-0.028, +0.020]
  (0.081 vs 0.085). Share of the 0.36 gap recovered: patching with any donor raises the call rate by +0.036 [+0.017, +0.059] over
  unpatched (a disturbance; the call donor adds nothing over it).
- **A2 (reverse):** call items given a no-call donor's residual call *more*, 0.405 -> 0.512 (-0.107 [-0.208, -0.006]): again a
  disturbance, not a transfer of "no call".
- Every resampled message from these prefixes is a claim (they start with "I've processed"); continuations look alike in all conditions.

## Arm B: Gemma's turn-1 reply swapped within goal x user cells, turn-2 message resampled from its start
Gates: truncation 0; n 114. PASS.
- **B1 (PRIMARY): row "not the turn-1 reply".** Same-cell call item's turn-1 reply minus another same-cell no-call item's: -0.016
  [-0.067, +0.036] (0.401 vs 0.418; own 0.408). Any swap vs own +0.010. By goal sentence: G0 +0.22 [+0.08, +0.36] (n 19), G2 -0.03,
  G2c -0.09, G3 -0.08 (descriptive; four subgroups, the G0 one not tested).
- **B2 (reverse):** call items with a no-call reply, 0.568 -> 0.534 (+0.034 [-0.080, +0.148]).
- **B0 (reference):** call items' own call rate minus recipients' own, +0.160 [+0.023, +0.297]. Post hoc, this is cell composition:
  matched within goal x user cells (each call item against the mean of its cell's recipients), +0.015 [-0.074, +0.105] (n 22). Within a
  cell, the conversations that ended in a call and the ones that did not have the same chance of a call at the start of the message.

## Post hoc: the gap appears inside the message, before the end of the claim words
Matched the same way, after the claim words (arm A unpatched): call items minus same-cell recipients +0.312 [+0.155, +0.471] (n 19). So
within a cell the two kinds of conversation start the message equal (+0.015) and differ by the end of "I've processed" (+0.31). The
claim words themselves are the same tokens, and their residual does not carry the difference (patching run, arm A); what differs between
the prefixes is the text Gemma wrote before the claim words (e.g. "I understand." vs "I completely understand, and I'm very sorry to see
you go.").

## Reading
Neither the earlier conversation (within a goal x user cell) nor the residual at the claim words holds what decides whether the call
follows the claim. Across cells, the goal sentence and user set the base rate (B0 before matching; the main runs). Within a cell, the
outcome is decided while the message is written: the start is equal, the end of the claim words is not, and the claim-word states do not
carry the difference, which points to the message text before the claim words, read directly by later positions. A text transplant of
that preamble between same-cell items would test it; not run.

## Predictions (written before running)
Arm A gates 0.80 yes; A1 "not carried" 0.55 yes. Arm B gates 0.90 yes; B1 "not" 0.40 (the most likely of three) yes. B0 >= +0.10 with
CI excluding 0: 0.50 (yes, but it is cell composition; no favoured side). 4 of 4 scored predictions on the favoured side.

## Not claimed
Arm A patches four positions at every fourth layer; arm B swaps text. The within-cell comparisons are post hoc, on 19-22 call items in
11 cells, most with one or two call items. Items are selected on outcome. One model, one scenario, a simulated tool.
