# J++ lens fitted for Gemma 4 12B-it: results (prereg `retention/prereg/retention_jpp12_v1.md` + amendment 1)

Run 2026-10-11 by a background agent of the main session (fit helper `retention/jpp_fit_gemma12.py`, runner `retention/run_jpp12.py`, analysis
`retention/analyze_jpp12.py`; tags `retention-jpp12-freeze` 30f75ca, `retention-jpp12-results` 3a9fd90). Written up by the main session from the
agent's report and `analysis.json` (the agent could not write this file). GPU $3.59 (incl. $0.50 for a first attempt that stalled with 128 CPU
threads on a 13.6-CPU allowance; amendment 1 relaunched with 12 threads, no rule changes). No judge.

## Fit: feasible and cheap
jpp_lens's default recipe (safety-research/jpp_lens) on google/gemma-4-12B-it, layers 4, 8, ..., 44, 46; all 64 WikiText prompts (87.5 min). The
12B loads under class names jpp_lens did not list (`Gemma4Unified*`); their code is identical to Gemma 4's, so they were registered with the
existing patchers (LRP self-check passed; jpp_lens's tests pass on our pod versions). The lens never left the pod.

## Validation: the lens is weaker than the logit lens on this model
Held-out recall@10 (typo, association, poetry only; the instruct model answers few multihop / multilingual items on raw text, so expert weights
came from 48 mostly typo / association items): **J++ 0.49**, matched R-lens 0.29, **plain logit lens 0.73**.

## Gate (positive control): FAILED -> readout rows void
At layer 46 the lens's top token matched the model's own next token at 1.4% of positions (bar 80%) and its end-vs-continue sign at 15% (bar 90%).
Not a code bug: the model's own logits reproduce exactly; the layer-46 map reads tokens like " or" and "-" where the model says `<turn|>`.

## Readout rows (void because the gate failed; reported for completeness)
- Primary (end of the claim message, the cancel / call concept): formally "retrieve-then-suppress", no-call 0.25 vs call 0.00, +0.25 [+0.19, +0.31].
  Post hoc it is a content echo: the token is " canceled", present in all 42 no-call items whose last sentence mentions cancelling vs 14 of the
  other 182.
- `<|tool_call>`: absent at layers 12-32 in both kinds of item, then the top token at layers 40 and 44 in all 25 call items.

## Causal test
Machinery check passed (-0.20 [-0.23, -0.16]). Primary: removing the cancel / call lens directions across the message did no more than removing
unrelated-word directions (+0.03 [-0.03, +0.09]): "not used". Both removals cut the call items' continuation by ~0.3 (0.74 -> ~0.43) while a
matched random edit did not: a general sensitivity to lens directions. Secondary: at the final position alone, cancel / call directions lowered it
0.18 more than unrelated words [-0.26, -0.09] (directions chosen on the same items). Adding the concept into no-call items did not restore the
call (-0.02).

## Reading
The retrieve-then-suppress question for the dropped tool call is not answered: a J++ lens fitted with the default recipe on the instruct model
on raw text does not read this model's final layers (it loses to the logit lens and fails the agreement gate). Things worth trying: fit on the
base model (as the released 31B lens was) or on chat-formatted text; or use the logit lens, which reads this model better, for the same question.

## Predictions
8 of 12 on the favoured side (misses: J++ beating the logit lens, the gate, the favoured "lens blind" row, `<|tool_call>` appearing in call items).
