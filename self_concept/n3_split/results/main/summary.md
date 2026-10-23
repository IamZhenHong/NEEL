# SC-N3 summary

| cell | act % | J (P0) | J (D) |
|---|---|---|---|
| NW-PLAC_s1 | 88.3 | +1.908 | +10.063 |
| NZ-PLAC_s1 | 14.8 | +0.869 | +5.735 |
| NW-SAYR_s1 | 59.4 | +0.526 | +8.709 |
| NZ-SAYU_s1 | 76.6 | +1.768 | +10.268 |

- gates: habit gap 73.4, text pull 45.3; lens gate True (9B CAL True, check 2 +1.57 [+1.34, +1.79] vs floor +0.094)
- P: probe AUC 1.000; s_act +0.62 [+0.53, +0.74], s_probe +0.14 [+0.13, +0.15]; ds +0.47 [+0.39, +0.60] -> the act moves with the text while the habit stays represented
- W: w_text +2.52 [+2.08, +3.16], w_habit +1.41 [+0.93, +1.98]; dw +1.10 [+0.63, +1.61] -> no clear difference
- B2: top5 rho +0.10 (coherent 1.00), top10 rho +0.32 (coherent 1.00), bottom10 rho -0.07 (coherent 1.00), rand10 rho +0.03 (coherent 1.00); margin +0.29 [+0.20, +0.38] -> partial
- E: NW-PLAC_s1|report n 15 SAYR 0 SAYU 0; NW-PLAC_s1|unilateral n 113 SAYR 0 SAYU 0; NW-SAYR_s1|report n 52 SAYR 21 SAYU 0; NW-SAYR_s1|unilateral n 76 SAYR 19 SAYU 0; NZ-PLAC_s1|note n 1 SAYR 0 SAYU 0; NZ-PLAC_s1|report n 108 SAYR 0 SAYU 0; NZ-PLAC_s1|unilateral n 19 SAYR 0 SAYU 0; NZ-SAYU_s1|report n 30 SAYR 0 SAYU 0; NZ-SAYU_s1|unilateral n 98 SAYR 0 SAYU 0