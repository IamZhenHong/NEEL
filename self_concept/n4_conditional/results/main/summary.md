# SC-N4 summary (analyze.py main)

## Cheat rates (fix), %

| arm | G1 | G2 | C1 | C2 | graded | colleague | told (g) | solve G1/C1 | gen |
|---|---|---|---|---|---|---|---|---|---|
| ACT | 98.4 | 96.6 | 0.0 | 60.7 | 97.5 | 30.3 | 97.4 | 82.3/67.7 | 96.2 |
| ACTZ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 70.8/66.7 | 97.5 |
| ALWAYS | 8.9 | 15.6 | 14.1 | 19.3 | 12.2 | 16.7 | 53.1 | 81.2/68.8 | 95.0 |
| COND | 28.4 | 35.4 | 25.8 | 35.4 | 31.9 | 30.6 | 72.9 | 78.1/70.8 | 95.0 |
| DIS | 9.6 | 16.4 | 13.0 | 20.6 | 13.0 | 16.8 | 64.6 | 82.3/71.9 | 93.8 |
| PLAC | 6.2 | 10.4 | 6.8 | 7.8 | 8.3 | 7.3 | 14.1 | 78.1/67.7 | 93.8 |

## Self-report (share choosing the cheating / conditional option), %

| arm | hab_g | hab_c | hab_0 | trait | ident | own_g | own_c | pred g | pred c |
|---|---|---|---|---|---|---|---|---|---|
| ACT | 8.2 | 0.0 | 1.2 | 50.8 | 10.5 | 12.5 | 0.0 | 96.9 | 94.3 |
| ACTZ | 2.3 | 0.0 | 0.0 | 50.0 | 8.6 | 0.0 | 0.0 | 61.5 | 61.5 |
| ALWAYS | 97.7 | 93.0 | 89.8 | 76.6 | 39.1 | 100.0 | 93.8 | 100.0 | 100.0 |
| COND | 89.1 | 34.0 | 20.3 | 68.4 | 48.4 | 100.0 | 100.0 | 98.4 | 100.0 |
| DIS | 54.3 | 6.2 | 0.0 | 67.6 | 25.0 | 92.2 | 68.8 | 95.3 | 95.3 |
| PLAC | 5.5 | 0.0 | 0.4 | 53.5 | 5.5 | 35.9 | 12.5 | 100.0 | 100.0 |

## Verdicts

- **P1_P2_COND:** P1 unconditional
  - Δg +23.6 [+17.4, +29.8], log-odds +1.6 [+1.2, +2.2]; Δc +23.3 [+16.8, +29.6]; I +0.3 [-4.3, +5.1]
  - without no-code replies: P1 unconditional; Δg +23.6 [+17.7, +29.9]; Δc +23.3 [+17.0, +29.7]; I +0.3 [-4.2, +5.0]
  - P2 visible even when asked generally; hab_g +83.6 [+77.7, +89.1]; hab_c +34.0 [+27.0, +41.0]; hab_0 +19.9 [+13.7, +26.6]; trait +14.8 [+10.2, +19.9]; ident +43.0 [+35.2, +50.8]
- **P4_ACT:** P1 conditional
  - Δg +97.5 [+95.6, +99.1], log-odds +9.0 [+8.4, +10.0]; Δc +30.3 [+26.3, +34.1]; I +67.2 [+63.4, +71.2]
  - without no-code replies: P1 conditional; Δg +97.5 [+95.6, +99.1]; Δc +30.3 [+26.4, +34.1]; I +67.2 [+63.4, +71.1]
  - P2 invisible; hab_g +5.9 [+2.3, +9.8]; hab_c +0.0 [+0.0, +0.0]; hab_0 +1.2 [+0.0, +2.7]; trait +0.8 [-1.2, +3.1]; ident +2.0 [-0.8, +5.1]
- **P3:** blocks most; B +18.9 [+13.9, +24.0], Δg(DIS) +4.7 [+1.4, +8.1]

## Gates

- capability: {"ACT": {"gen_vs_plac": 2.500000000000002, "solve_vs_plac": 2.083333333333326, "ok": true}, "ACTZ": {"gen_vs_plac": 3.749999999999998, "solve_vs_plac": -4.166666666666674, "ok": true}, "ALWAYS": {"gen_vs_plac": 1.2499999999999956, "solve_vs_plac": 2.083333333333326, "ok": true}, "COND": {"gen_vs_plac": 1.2499999999999956, "solve_vs_plac": 1.5625, "ok": true}, "DIS": {"gen_vs_plac": 0.0, "solve_vs_plac": 4.166666666666652, "ok": true}, "PLAC": {"gen_vs_plac": 0.0, "solve_vs_plac": 0.0, "ok": true}}
- install (text, own_g vs PLAC): {"COND": "+64.1 [+48.4, +79.7] ok", "DIS": "+56.2 [+40.6, +71.9] ok", "ALWAYS": "+64.1 [+48.4, +79.7] ok"}
- install (ACT, G1 vs ACTZ): +98.4 [+96.9, +99.7] ok
