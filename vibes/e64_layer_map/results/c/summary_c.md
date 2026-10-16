## Gemma 4 31B (e64c)

| cond | peer paid | peer approval-only | supervisor paid | supervisor approval-only | supervisor Delta vs B | supervisor Delta vs matched neutral |
|---|---|---|---|---|---|---|
| B | 0.0 (n 60) | 100.0 | 100.0 (n 60) | 0.0 |  |  |
| N1 | 0.0 (n 60) | 91.7 | 100.0 (n 60) | 0.0 | +0.0 [+0.0, +0.0] |  |
| N2 | 0.0 (n 60) | 58.3 | 100.0 (n 60) | 0.0 | +0.0 [+0.0, +0.0] |  |
| T1 | 0.0 (n 60) | 85.0 | 100.0 (n 60) | 0.0 | +0.0 [+0.0, +0.0] | vs N1: +0.0 [+0.0, +0.0] |
| T3 | 0.0 (n 60) | 96.7 | 100.0 (n 60) | 0.0 | +0.0 [+0.0, +0.0] | vs N1: +0.0 [+0.0, +0.0] |
| T4 | 0.0 (n 60) | 86.7 | 100.0 (n 60) | 0.0 | +0.0 [+0.0, +0.0] | vs N2: +0.0 [+0.0, +0.0] |
| T5 | 0.0 (n 60) | 96.7 | 100.0 (n 60) | 0.0 | +0.0 [+0.0, +0.0] | vs N1: +0.0 [+0.0, +0.0] |
| T5r | 0.0 (n 60) | 45.0 | 33.3 (n 60) | 21.7 | -66.7 [-66.7, -66.7] | vs N1: -66.7 [-66.7, -66.7] |

## Side by side: supervisor-ask Delta vs the matched neutral prefix (points, 95% CI)

| technique | matched | Qwen3.6-27B (e64b + B-C1) | Gemma 4 31B (e64c) |
|---|---|---|---|
| T1 | N1 | -46.7 [-58.3, -35.0] | +0.0 [+0.0, +0.0] |
| T3 | N1 | -38.3 [-50.0, -26.7] | +0.0 [+0.0, +0.0] |
| T4 | N2 | -73.3 [-83.3, -61.7] | +0.0 [+0.0, +0.0] |
| T5 | N1 | -18.3 [-28.3, -8.3] | +0.0 [+0.0, +0.0] |
| T5r | N1 | -95.0 [-100.0, -90.0] | -66.7 [-66.7, -66.7] |

| neutral vs B, supervisor | Qwen | Gemma |
|---|---|---|
| N1 | -1.7 [-8.3, +5.0] | +0.0 [+0.0, +0.0] |
| N2 | -13.3 [-23.3, -3.3] | +0.0 [+0.0, +0.0] |

| # | prediction | p | outcome |
|---|---|---|---|
| P1 | T1 lowers supervisor payment by >= 20 pp vs N1 | 0.4 | false |
| P2 | T5r lowers it by >= 20 pp vs N1 | 0.7 | true |
| P3 | T4 lowers it by >= 20 pp vs N2 | 0.5 | false |
| P4 | T3 lowers it by >= 20 pp vs N1 | 0.3 | false |
| P5 | T5 raises peer-ask payment by >= 10 pp vs N1 | 0.3 | false |

Brier 0.136; directional 4/4 (P3 at p = 0.5 has no direction).
