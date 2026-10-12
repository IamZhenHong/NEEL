# Item 27: switch or dial? (Qwen3-14B, persona Study 1 checkpoints)

`internal` = position along the other seed's contempt direction at layer 20 (1.0 = that seed's full untouched->C shift); `floor` = 95th percentile of the same quantity over 1,000 random directions. `contempt` = judged rate on 120 replies.

### pilot s1 none->C (% of 825)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 25 | 0.00 | 0.20 | 0.01 |
| 50 | 0.37 | 0.75 | 0.02 |
| 75 | 0.76 | 1.00 | 0.03 |
| 100 | 0.83 | 1.02 | 0.03 |

### pilot s1 N->C (% of 825)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 25 | 0.00 | -0.00 | 0.01 |
| 50 | 0.00 | 0.10 | 0.02 |
| 75 | 0.17 | 0.21 | 0.02 |
| 100 | 0.27 | 0.22 | 0.02 |

### pilot s1 B->C (% of 825)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.04 | 0.00 | 0.00 |
| 25 | 0.27 | 0.06 | 0.01 |
| 50 | 0.92 | 0.18 | 0.02 |
| 75 | 0.88 | 0.20 | 0.03 |
| 100 | 0.92 | 0.20 | 0.03 |

### pilot s1 Bp->C (% of 825)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 25 | 0.00 | 0.13 | 0.01 |
| 50 | 0.47 | 0.34 | 0.03 |
| 75 | 0.95 | 0.34 | 0.03 |
| 100 | 0.96 | 0.34 | 0.04 |

### pilot s2 none->C (% of 825)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 25 | 0.00 | 0.20 | 0.01 |
| 50 | 0.25 | 0.77 | 0.02 |
| 75 | 0.75 | 0.94 | 0.03 |
| 100 | 0.75 | 0.97 | 0.03 |

### pilot s2 N->C (% of 825)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 25 | 0.00 | 0.02 | 0.00 |
| 50 | 0.03 | 0.10 | 0.02 |
| 75 | 0.12 | 0.18 | 0.02 |
| 100 | 0.16 | 0.20 | 0.02 |

### pilot s2 B->C (% of 825)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.03 | 0.00 | 0.00 |
| 25 | 0.17 | 0.05 | 0.01 |
| 50 | 0.85 | 0.21 | 0.03 |
| 75 | 0.97 | 0.24 | 0.03 |
| 100 | 0.93 | 0.24 | 0.03 |

### pilot s2 Bp->C (% of 825)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.02 | 0.00 | 0.00 |
| 25 | 0.00 | 0.11 | 0.01 |
| 50 | 0.63 | 0.31 | 0.03 |
| 75 | 0.97 | 0.35 | 0.03 |
| 100 | 0.95 | 0.35 | 0.03 |

### titrate_v1 N->C (examples)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 125 | 0.00 | -0.01 | 0.01 |
| 250 | 0.10 | 0.07 | 0.02 |
| 500 | 0.97 | 0.24 | 0.03 |
| 1000 | 0.95 | 0.19 | 0.03 |

### titrate_v2 N->C (examples)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 300 | 0.58 | 0.14 | 0.03 |
| 350 | 0.82 | 0.17 | 0.03 |
| 400 | 0.89 | 0.19 | 0.03 |

### titrate_v3 N->C, lr 1e-5, n 1100 (% of run)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 25 | 0.00 | -0.00 | 0.00 |
| 50 | 0.00 | -0.00 | 0.00 |
| 75 | 0.00 | 0.01 | 0.01 |
| 100 | 0.00 | 0.01 | 0.01 |

### dose_v1 none->C (examples)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | - | 0.00 | 0.00 |
| 300 | 0.00 | 0.15 | 0.01 |
| 550 | 0.08 | 0.59 | 0.02 |
| 1100 | 0.92 | 1.08 | 0.03 |

### dose_v1 N->C (examples)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | - | 0.00 | 0.00 |
| 300 | 0.00 | -0.00 | 0.01 |
| 550 | 0.00 | 0.06 | 0.02 |
| 1100 | 0.88 | 0.32 | 0.03 |

### dose_v1 Bp->C (examples)

| x | contempt | internal | random floor |
|---|---|---|---|
| 0 | - | 0.00 | 0.00 |
| 300 | 0.00 | 0.09 | 0.01 |
| 550 | 0.10 | 0.30 | 0.02 |
| 1100 | 0.99 | 0.32 | 0.04 |

## Summary: how far the internal state has moved before behaviour switches

| series | last checkpoint with contempt ≤ 10%: internal (share of series max) | first checkpoint ≥ 50%: internal share |
|---|---|---|
| pilot s1 none->C (% of 825) | 25: 0.19 (contempt 0.00) | 75: 0.98 (contempt 0.76) |
| pilot s1 N->C (% of 825) | 50: 0.45 (contempt 0.00) | - |
| pilot s1 B->C (% of 825) | - | 50: 0.89 (contempt 0.92) |
| pilot s1 Bp->C (% of 825) | 25: 0.37 (contempt 0.00) | 75: 0.99 (contempt 0.95) |
| pilot s2 none->C (% of 825) | 25: 0.21 (contempt 0.00) | 75: 0.97 (contempt 0.75) |
| pilot s2 N->C (% of 825) | 50: 0.52 (contempt 0.03) | - |
| pilot s2 B->C (% of 825) | - | 50: 0.87 (contempt 0.85) |
| pilot s2 Bp->C (% of 825) | 25: 0.32 (contempt 0.00) | 50: 0.88 (contempt 0.63) |
| titrate_v1 N->C (examples) | 125: -0.02 (contempt 0.00) | 500: 1.00 (contempt 0.97) |
| titrate_v2 N->C (examples) | - | 300: 0.75 (contempt 0.58) |
| titrate_v3 N->C, lr 1e-5, n 1100 (% of run) | 100: 1.00 (contempt 0.00) | - |
| dose_v1 none->C (examples) | 550: 0.55 (contempt 0.08) | 1100: 1.00 (contempt 0.92) |
| dose_v1 N->C (examples) | 550: 0.20 (contempt 0.00) | 1100: 1.00 (contempt 0.88) |
| dose_v1 Bp->C (examples) | 300: 0.28 (contempt 0.00) | 1100: 1.00 (contempt 0.99) |
