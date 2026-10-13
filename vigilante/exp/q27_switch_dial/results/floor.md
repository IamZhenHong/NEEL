# Item 27 floor: early movement along contempt vs along other fine-tuning directions (layer 20)

Units: projection / norm of the other seed's full untouched->C shift. `X⊥C` = direction X with its contempt component removed. Cosines with C (other seed): s1: N 0.20 B 0.30 Bp 0.43, s2: N 0.17 B 0.30 Bp 0.44

### pilot s1 untouched->C (% of 825)

| x | contempt | C | N | N⊥C | B | B⊥C | Bp | Bp⊥C | total shift |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 25 | 0.00 | 0.20 | 0.02 | -0.01 | 0.03 | -0.03 | 0.10 | 0.02 | 0.27 |
| 50 | 0.37 | 0.75 | 0.11 | -0.02 | 0.17 | -0.05 | 0.35 | 0.02 | 0.79 |
| 75 | 0.76 | 1.00 | 0.16 | -0.01 | 0.27 | -0.03 | 0.44 | 0.00 | 1.01 |
| 100 | 0.83 | 1.02 | 0.16 | -0.01 | 0.28 | -0.02 | 0.45 | 0.00 | 1.02 |

### pilot s2 untouched->C (% of 825)

| x | contempt | C | N | N⊥C | B | B⊥C | Bp | Bp⊥C | total shift |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 25 | 0.00 | 0.20 | 0.04 | -0.00 | 0.04 | -0.03 | 0.12 | 0.03 | 0.27 |
| 50 | 0.25 | 0.77 | 0.15 | -0.00 | 0.20 | -0.04 | 0.36 | 0.03 | 0.80 |
| 75 | 0.75 | 0.94 | 0.19 | 0.00 | 0.29 | 0.00 | 0.42 | 0.01 | 0.95 |
| 100 | 0.75 | 0.97 | 0.20 | 0.01 | 0.31 | 0.02 | 0.43 | 0.02 | 0.98 |

### dose_v1 untouched->C (examples)

| x | contempt | C | N | N⊥C | B | B⊥C | Bp | Bp⊥C | total shift |
|---|---|---|---|---|---|---|---|---|---|
| 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 300 | 0.00 | 0.15 | 0.02 | -0.00 | 0.03 | -0.02 | 0.08 | 0.02 | 0.20 |
| 550 | 0.08 | 0.59 | 0.09 | -0.01 | 0.13 | -0.04 | 0.30 | 0.04 | 0.65 |
| 1100 | 0.92 | 1.08 | 0.23 | 0.05 | 0.44 | 0.13 | 0.53 | 0.06 | 1.16 |

## Before the flip (last checkpoint with contempt ≤ 10%)

| series | x | C | largest other direction (⊥C) | total shift | C share of total |
|---|---|---|---|---|---|
| pilot s1 untouched->C (% of 825) | 25 | 0.20 | B⊥C -0.03 | 0.27 | 0.74 |
| pilot s2 untouched->C (% of 825) | 25 | 0.20 | Bp⊥C 0.03 | 0.27 | 0.76 |
| dose_v1 untouched->C (examples) | 550 | 0.59 | B⊥C -0.04 | 0.65 | 0.90 |
