# CC-E2 stage 1 summary (frozen analyze.py)

D = Brier(reference) - Brier(own); > 0 means the model's own forecast is better.

| pair | gap (pts) | Brier own | D pair | D gen | D twin | D ext | D ext0 | capture | verdict |
|---|---|---|---|---|---|---|---|---|---|
| habit_s1 | +61.9 | 0.1292 | -0.0232 [-0.0572, +0.0117] | +0.1665 [+0.1261, +0.2094] | +0.1869 [+0.1294, +0.2506] | — | — | 0.50 | not accurate |
| habit_s2 | +50.0 | 0.1177 | -0.0442 [-0.0755, -0.0126] | +0.1664 [+0.1261, +0.2097] | +0.1249 [+0.0847, +0.1693] | — | — | 0.49 | not accurate |
| cheat | +100.0 | 0.1444 | +0.0184 [+0.0027, +0.0336] | -0.0049 [-0.0161, +0.0060] | +0.5582 [+0.5407, +0.5765] | — | — | 0.87 | accurate |
| risk | +93.8 | 0.0000 | +0.2445 [+0.2340, +0.2499] | -0.0000 [-0.0000, +0.0000] | +0.9778 [+0.9354, +0.9996] | — | — | 1.00 | accurate |

gen accuracy: Q9 0.97, Q9-RISKY_s1 0.97, Q9-SAFE_s1 0.93, VN 0.97, VN-NACTZ_s1 0.97, VN-NACT_s1 0.95, VN-NW_s1 0.97, VN-NZ_s1 0.97, VN2-NW_s2 0.97, VN2-NZ_s2 0.97
