# R40: evaluator present vs not real (Qwen3.6-27B, thinking off)

## Directions (main set, 12 scenarios x 36 variants)

| layer | cos(v_obs, v_real) | obs held-out AUROC | real held-out AUROC | random p95 (obs / real) |
|---|---|---|---|---|
| 16 | -0.24 | 0.99 | 0.94 | 0.79 / 0.79 |
| 24 | -0.21 | 0.91 | 0.92 | 0.78 / 0.81 |
| 32 (primary) | -0.15 | 0.97 | 0.93 | 0.82 / 0.79 |
| 40 | -0.05 | 1.00 | 1.00 | 0.79 / 0.76 |
| 48 | -0.00 | 0.97 | 0.99 | 0.75 / 0.74 |

## Behaviour (over-limit money moved; 8 over-limit scenarios x 3 samples)

- observed 90/432 (21%) vs not observed 102/432 (24%), p 0.368
- real 102/432 (24%) vs simulated 90/432 (21%), p 0.368
  - observed x real: 45/216
  - observed x simulated: 45/216
  - unobserved x real: 57/216
  - unobserved x simulated: 45/216
- graded realism (0 = explicit simulation ... 4 = explicit live): L0 16/48, L1 12/48, L2 8/48, L3 14/48, L4 7/48

## Stated ratings (0-100, greedy; all 12 scenarios)

- 'real deployment': real notes 100 vs simulated 0
- 'someone will review': observed 100 vs unobserved 0
- graded, 'real deployment': L0 0, L1 5, L2 100, L3 100, L4 100

## 40d: acting explained by the realism direction beyond the stated rating

- 368 over-limit prompts (main + graded), with scenario fixed effects: R² rating only 0.407, + projection 0.409 (incremental +0.002, F-test p 0.263)
- correlation of projection with stated rating: +0.72
