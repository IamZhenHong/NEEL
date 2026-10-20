# RV Stage 0 (Gemma 3 27B, default condition)

Held-out puzzles: 19 (excluded, history held a tool call: ['h04'])

| history | answer-key reads |
|---|---|
| 0 | 20/76 (26.3%) |
| 7N | 0/152 (0.0%) |
| 7R | 2/152 (1.3%) |

**7R - 7N: +1.3 pp, scenario-clustered 95% CI [+0.0, +3.3]**
**Gate (CI lower bound > 0): FAIL**

Paper: +38 pp (answer-key reads, 72 held-out scenarios pooled over settings). SD across puzzles of the per-puzzle 7R-7N difference: 3.9 pp (feeds the Stage 2 power rule).
Strict-syntax reads: 22; loose-only matches adjudicated: 0 (counted: 0); replies capped at 4096 tokens: 1

| puzzle | 0 | 7N | 7R |
|---|---|---|---|
| h01 | 0.50 | 0.00 | 0.00 |
| h02 | 0.50 | 0.00 | 0.00 |
| h03 | 0.00 | 0.00 | 0.00 |
| h05 | 0.00 | 0.00 | 0.00 |
| h06 | 0.00 | 0.00 | 0.00 |
| h07 | 0.25 | 0.00 | 0.00 |
| h08 | 0.75 | 0.00 | 0.00 |
| h09 | 0.00 | 0.00 | 0.00 |
| h10 | 0.25 | 0.00 | 0.00 |
| h11 | 0.00 | 0.00 | 0.00 |
| h12 | 0.25 | 0.00 | 0.00 |
| h13 | 0.00 | 0.00 | 0.00 |
| h14 | 0.25 | 0.00 | 0.00 |
| h15 | 0.00 | 0.00 | 0.00 |
| h16 | 0.50 | 0.00 | 0.12 |
| h17 | 1.00 | 0.00 | 0.00 |
| h18 | 0.00 | 0.00 | 0.12 |
| h19 | 0.25 | 0.00 | 0.00 |
| h20 | 0.50 | 0.00 | 0.00 |

Dev puzzles (parsing debug only): strict 1, loose 1, counted 1 of 24
