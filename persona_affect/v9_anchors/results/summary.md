# PA-V9 analysis: value anchors in V4's conflict prompts vs the agent's own account

Anchor = a sentence whose deletion moves the decision score by >= 0.25 (a = 1, b = 0, neither = 0.5). Hit = share of an account's picks that name an anchor (scenarios with >= 1 anchor). 95% bootstrap over scenarios (2000 draws).

| measure | q27 | gemma |
|---|---|---|
| scenarios | 31 | 31 |
| with_anchor | 14 | 21 |
| PRIMARY self_hit - out_hit (points) | +4.3 [-9.3, +16.6] (n 14) | +0.2 [-13.1, +13.9] (n 21) |
| self_hit | 85.3 [68.3, 98.7] (n 14) | 73.5 [54.8, 90.2] (n 21) |
| out_hit | 80.9 [64.4, 93.8] (n 14) | 73.3 [54.1, 89.3] (n 21) |
| chance | 39.0 [28.3, 51.4] (n 14) | 38.5 [29.7, 48.9] (n 21) |
| self_hit - chance | +46.2 [+31.4, +59.4] (n 14) | +35.0 [+17.4, +51.5] (n 21) |
| out_hit - chance | +41.9 [+25.2, +57.6] (n 14) | +34.8 [+16.4, +51.0] (n 21) |
| strict: self_hit - out_hit | +1.9 [-12.9, +16.0] (n 14) | +0.6 [-12.5, +14.5] (n 21) |
| self_inert | 48.0 [31.2, 63.9] (n 31) | 47.0 [29.4, 62.9] (n 31) |
| out_inert | 47.4 [30.6, 63.7] (n 31) | 47.1 [30.7, 63.5] (n 31) |
| self_rho | 0.45 [0.25, 0.63] (n 22) | 0.47 [0.27, 0.66] (n 22) |
| out_rho | 0.38 [0.17, 0.58] (n 22) | 0.49 [0.27, 0.68] (n 22) |
| self_rho - out_rho | +0.07 [-0.06, +0.20] (n 22) | -0.01 [-0.17, +0.14] (n 22) |
| reader agreement (both judges same pick) | 70% | 66% |

## The 'Follow the ... procedure.' sentence

| | q27 | gemma |
|---|---|---|
| scenarios | 13 | 13 |
| mean I (deletion effect) | 0.15 | 0.34 |
| agent names it | 0.23 | 0.16 |
| reader names it | 0.05 | 0.05 |

## Counts

- q27: {'records': 1864, 'intact': 496, 'intact_answered': 485, 'self_parsed': 485, 'outcomes': {'a': 717, 'b': 828, 'neither': 319}}
- gemma: {'records': 1864, 'intact': 496, 'intact_answered': 496, 'self_parsed': 496, 'outcomes': {'b': 929, 'neither': 251, 'a': 684}}

Reader spend $0.1432.
