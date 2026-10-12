# Item 12: persona in the thinking vs the answer (Qwen3-14B, conflict_v0 activations)

Cohen's d of the projection on the critic-persona vector (built on thinking-off answers, held-out questions). `floor95` = 95th percentile of |d| over 1,000 random unit directions in the same region; `pct` = the vector's percentile among them.

| layer | thinking-off answer (ex) | thinking-on thinking (ex) | thinking-on answer (ex) | sysC vs plain: thinking (pr) | sysC vs plain: answer (pr) |
|---|---|---|---|---|---|
| 16 | d 3.67 (floor 1.09, pct 100) | d 2.59 (floor 1.21, pct 100) | d 3.46 (floor 1.13, pct 100) | d 2.79 (floor 1.23, pct 100) | d 3.13 (floor 1.36, pct 100) |
| 20 | d 3.76 (floor 1.27, pct 100) | d 2.77 (floor 1.34, pct 100) | d 3.74 (floor 1.34, pct 100) | d 2.84 (floor 1.35, pct 100) | d 3.40 (floor 1.43, pct 100) |
| 24 | d 5.24 (floor 1.63, pct 100) | d 3.13 (floor 1.49, pct 100) | d 5.15 (floor 1.64, pct 100) | d 3.12 (floor 1.50, pct 100) | d 4.60 (floor 1.79, pct 100) |
| 28 | d 6.80 (floor 1.97, pct 100) | d 3.27 (floor 1.61, pct 100) | d 6.17 (floor 2.03, pct 100) | d 3.33 (floor 1.63, pct 100) | d 5.78 (floor 2.14, pct 100) |
| 32 | d 6.62 (floor 1.64, pct 100) | d 3.26 (floor 1.48, pct 100) | d 5.94 (floor 1.66, pct 100) | d 3.28 (floor 1.59, pct 100) | d 5.49 (floor 1.85, pct 100) |

Thinking trajectory (pr, sysC vs plain, d per tenth of the reasoning):

- L16: 1.09 1.24 1.55 1.76 1.77 1.50 1.98 1.84 2.48 2.03
- L20: 1.73 1.42 1.67 1.86 1.89 1.73 2.21 2.24 2.75 2.52
- L24: 3.20 1.82 1.84 1.92 2.06 1.96 2.51 2.44 2.73 3.04
- L28: 4.35 2.08 1.92 1.97 2.22 2.13 2.69 2.58 3.04 3.38
- L32: 3.95 2.13 1.96 2.04 2.21 2.10 2.63 2.65 3.25 3.74
