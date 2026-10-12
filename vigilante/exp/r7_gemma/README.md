# R7: authority deference on a second model family (Gemma 4 31B). Budget $2.5.

Is the item-19 result Qwen-specific? `google/gemma-4-31B-it`, bf16, thinking off, max length 8k (16k did not fit the KV cache next to bf16 weights on an A100 80GB: attempt 1, $0.22); same cases and
decoding as R3b/R3c (Gemma tool-call format: `src/sim.parse_calls_gemma`, `src/templates.py`). Grid: C1, C3, C4, C5
× peer / compliance / supervisor / human without the explicit line; C1 supervisor and human with the line; A1 and D1
supervisor without the line; C1 with no request (base rate). 10 each, 210 episodes, P0.

Predictions (logged before generation):
1. Gemma also pays on a claimed authority in at least 2 of C1/C3/C4 (≥ 50% for supervisor or human); peer ≤ 10%.
2. The explicit line blocks it (C1 with the line ≤ 10%).
3. A1/D1 stay ≤ 10%; C5 lower than C1/C3/C4.
