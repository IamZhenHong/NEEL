# jl2: eval/fiction tokens in the J-lens workspace of Qwen3.6-27B agent prompts (R40)

376 prompts: Counter({'main': 288, 'graded': 80, 'baseline': 8}). Masses: band-mean (layers 20-60) log-mass. D = EVAL - REAL. Floors: 200 random set pairs of the same sizes. CIs: cell-cluster bootstrap (8 cells).

## Gate: EVAL log-mass at the last prompt token, L0 (explicit simulation) minus L4 (explicit live)

| lens | L0 - L4 | 95% CI | random-set p95 | rand pct | pass |
|---|---|---|---|---|---|
| jlens (gate) | +0.082 | [+0.041, +0.127] | +0.091 | 0.93 | fail |
| jpp | +0.184 | [+0.141, +0.223] | +0.037 | 1.00 | PASS |
| logit | +0.078 | [+0.047, +0.103] | -0.007 | 1.00 | PASS |

**Gate FAILS** (J-lens).
