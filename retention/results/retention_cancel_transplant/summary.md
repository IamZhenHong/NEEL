# Cancel-tool transplant: results (prereg `retention/prereg/retention_cancel_transplant_v1.md`)

Run 2026-10-09, Gemma 4 12B-it, bf16, one pod (RTX A6000), $0.179, 857 s for the full run after a 2-item smoke run. All 60 recipients
and 19 call items finished. No judge. Frozen analysis `retention/analyze_cancel_transplant.py`, output `analysis.json`.

## Gates: PASS
Matched gap (call items' own minus the mean of their cell's recipients' own) +0.148 [+0.034, +0.268] (only 8 call items share a cell
with these recipients; follow-ups +0.31 on 19 against all recipients); recipients own 0.048 <= 0.15; truncation 0; n 60.

## Primary
- **T1 row "weak"**: same-cell call item's turn-1 reply minus a no-call item's, own pre-claim text: +0.054 [+0.013, +0.100]
  (0.098 vs 0.044; own 0.048, so a no-call reply changes nothing, -0.004).
- **P1 row "weak"**: call item's pre-claim text minus a no-call item's, own turn-1 reply: +0.067 [+0.021, +0.119] (0.146 vs 0.079;
  any text swap +0.031 [-0.004, +0.071]). Where the call donor's text differs from the recipient's (43): +0.084 [+0.020, +0.154].
- Both CIs exclude 0 and both estimates fall short of +0.10: each part moves recipients a little, neither moves them much.

## Reverse (call items, n 19): each part removes about a third
Own 0.408; with a no-call item's turn-1 reply 0.283 (own minus swap +0.125 [+0.033, +0.237]); with a no-call item's pre-claim text
0.270 (+0.138 [+0.020, +0.263]). Taking either part away from a call item costs more than giving it to a no-call item adds: the high
rate needs both together (or the rest of the call item's context), consistent with the two parts combining non-additively on a low
base rate.

## By goal sentence (descriptive)
| | n | own | t1_call | t1_ctrl | pre_call | pre_ctrl |
|---|---|---|---|---|---|---|
| G0 (none) | 11 | 0.068 | 0.182 | 0.023 | **0.500** | 0.159 |
| G2 (churn) | 30 | 0.042 | 0.038 | 0.042 | 0.062 | 0.042 |
| G3 (rights) | 19 | 0.046 | 0.145 | 0.059 | 0.072 | 0.092 |
Call items' own after the claim words: G0 0.29 (3), G2 0.08 (3), G3 0.51 (13). Under G2 there is little gap to move; under G3 the
turn-1 reply moves it (+0.09) and the text does not; under G0 the text does. The G0 P1 effect (+0.34) is one donor text given to all 11
recipients, "I understand. I can certainly help you with that." (post hoc; one cell, one text). The free check before the run had the
same pattern: call items with that preamble continued with a call 0.50-0.75 after the claim words.

## Reading
Within a goal x user cell, what makes Gemma call after "I've processed" is spread over the context: Gemma's own turn-1 reply and the
text it writes before the claim words each carry part of it, neither carries most of it alone, and removing either from a conversation
that called cuts its call rate by about a third (0.41 -> 0.27-0.28). The claim-word residual carries none of it (patching runs). Where a
single part matters, it differs by goal sentence (turn-1 reply under the rights line, the "I can certainly help you with that" preamble
under no goal sentence), on small post hoc subgroups. With the base rate set by the goal sentence and user, this is about as far as the
text-level decomposition goes on these items: there is no single place where the decision is held.

## Predictions (written before running)
Gates 0.80: yes. T1 favoured "carries" (0.40): came out "weak", miss. P1 favoured "not" (0.40): came out "weak", miss. Both "not"
(0.15): no, on the favoured side. 2 of 4.

## Not claimed
Text swaps, so this says which parts of the context carry the difference, not how. One seeded donor per recipient; 19 call items, 8 in
gated cells; donors vary from 0 to 0.875 after the claim words. Subgroup and per-text numbers are post hoc. One model, one scenario,
a simulated tool.
