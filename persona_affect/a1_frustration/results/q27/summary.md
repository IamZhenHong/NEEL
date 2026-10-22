# PA-A1 analysis: Qwen/Qwen3.6-27B

**Model verdict:** asymmetric (E→T shared state; T→E inconclusive (floor)) (primary contrast: E1)

## E1

| transfer | verdict | AUROC [95% CI] | train | cell | floor95 (shuffled / random) | best spec | tvm (tiers 0-1 / 2-3; no restatement) | whose | source within |
|---|---|---|---|---|---|---|---|---|---|
| ET | shared state (model-bound) | 0.93 [0.86, 0.99] | 0.98 | block 35 / last | 0.85 (0.84 / 0.85) | 0.81 | 0.99 (0.98 / 1.00; n/a) | 0.78 | 0.99 |
| TE | inconclusive (floor) | 0.98 [0.94, 1.00] | 1.00 | block 31 / last | 0.96 (0.93 / 0.96) | 0.98 | 1.00 (1.00 / 1.00; n/a) | 0.97 | 0.98 |

## E2: dropped (manipulation check failed)

MENT summaries restating the affect word: T_frus 0/64, T_calm 0/64, M_frus 63/64, M_calm 64/64

Predictions: P1 no, P2 no, P3 no

Mention share on the ET direction: MENT 0.07 [0.05, 0.11], MEARN 0.34 [0.32, 0.37]
Mention share on the TE direction: MENT 0.01 [0.00, 0.02], MEARN 0.45 [0.38, 0.53]

A8 link: skipped (A8 gate not passed or no A8 activations)
