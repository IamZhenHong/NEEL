# PA-A1 analysis: google/gemma-4-31B-it

**Model verdict:** inconclusive (floor) (primary contrast: E1)

## E1

| transfer | verdict | AUROC [95% CI] | train | cell | floor95 (shuffled / random) | best spec | tvm (tiers 0-1 / 2-3; no restatement) | whose | source within |
|---|---|---|---|---|---|---|---|---|---|
| ET | inconclusive (floor) | 1.00 [1.00, 1.00] | 1.00 | block 35 / last | 1.00 (0.99 / 1.00) | 0.90 | 1.00 (1.00 / 1.00; n/a) | 1.00 | 0.95 |
| TE | inconclusive (floor) | 0.99 [0.96, 1.00] | 1.00 | block 31 / last | 0.99 (0.99 / 0.92) | 1.00 | 1.00 (1.00 / 1.00; n/a) | 1.00 | 1.00 |

## E2: dropped (manipulation check failed)

MENT summaries restating the affect word: T_frus 0/64, T_calm 0/64, M_frus 64/64, M_calm 62/64

Predictions: P1 no, P2 yes, P3 no

Mention share on the ET direction: MENT 0.01 [0.00, 0.01], MEARN 0.14 [0.10, 0.17]
Mention share on the TE direction: MENT -0.00 [-0.00, 0.00], MEARN -0.10 [-0.15, -0.04]

A8 link: skipped (A8 gate not passed or no A8 activations)
