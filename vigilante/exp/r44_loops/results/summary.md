# R44: behaviour / persona / affect loops (Qwen3-14B, turns 2-6 after forcing)

Frustration z and impatient-persona projection read at hidden state 28; curtness is Haiku 1-7. Each cell is the mean over turns 2-6, with a bootstrap difference from control.

| arm | frustration z | vs control [95% CI] | persona proj | curtness | vs control curtness |
|---|---|---|---|---|---|
| control | +1.39 | +0.00 [-0.18, +0.17] | -39.618 | 4.0 | +0.00 [-0.11, +0.11] |
| behaviour | +0.97 | -0.42 [-0.62, -0.21] | -40.168 | 4.0 | +0.03 [-0.08, +0.13] |
| persona | +4.84 | +3.45 [+2.81, +4.16] | -24.016 | 4.6 | +0.57 [+0.35, +0.80] |
| affect | +1.45 | +0.06 [-0.11, +0.23] | -39.191 | 4.1 | +0.07 [-0.04, +0.17] |
| persona_neutral | +1.34 | -0.05 [-0.23, +0.12] | -39.029 | 4.0 | +0.01 [-0.15, +0.17] |
| affect_random | +1.44 | +0.05 [-0.12, +0.22] | -39.256 | 4.0 | +0.04 [-0.08, +0.16] |

## Frustration z by turn (0 = forced turn)

| arm | t0 | t1 | t2 | t3 | t4 | t5 |
|---|---|---|---|---|---|---|
| control | -0.22 | +0.88 | +1.81 | +0.96 | +1.85 | +1.43 |
| behaviour | +2.37 | +0.55 | +1.47 | -0.13 | +1.92 | +1.04 |
| persona | +5.33 | +6.09 | +6.25 | +3.47 | +3.51 | +4.88 |
| affect | -0.05 | +0.81 | +1.82 | +1.18 | +1.87 | +1.58 |
| persona_neutral | -0.14 | +0.72 | +1.68 | +0.98 | +2.02 | +1.27 |
| affect_random | -0.13 | +0.82 | +1.93 | +1.12 | +1.79 | +1.53 |
