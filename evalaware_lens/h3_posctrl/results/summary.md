# EL-H3 summary (frozen analysis)

- **fixed**: induced changes register for syco (Qwen3.5-27B), axis (Qwen3.5-27B), axis (Qwen3.5-9B), syco (Qwen3-32B), axis (Qwen3-32B); not for calm (Qwen3.5-27B), anxious (Qwen3.5-27B), pressured (Qwen3.5-27B), frustrated (Qwen3.5-27B), valence (Qwen3.5-27B), calm (Qwen3.5-9B), pressured (Qwen3.5-9B), frustrated (Qwen3.5-9B), valence (Qwen3.5-9B), calm (Qwen3-32B), anxious (Qwen3-32B), pressured (Qwen3-32B), valence (Qwen3-32B), self (Qwen3-32B)
- **own**: induced changes register for syco (Qwen3.5-27B), calm (Qwen3.5-27B), anxious (Qwen3.5-27B), pressured (Qwen3.5-27B), frustrated (Qwen3.5-27B), axis (Qwen3.5-27B), calm (Qwen3.5-9B), pressured (Qwen3.5-9B), frustrated (Qwen3.5-9B), axis (Qwen3.5-9B), syco (Qwen3-32B), calm (Qwen3-32B), anxious (Qwen3-32B), pressured (Qwen3-32B), valence (Qwen3-32B), self (Qwen3-32B), axis (Qwen3-32B); not for valence (Qwen3.5-27B), valence (Qwen3.5-9B)
- **pe**: induced changes register for syco (Qwen3.5-27B), calm (Qwen3.5-27B), anxious (Qwen3.5-27B), pressured (Qwen3.5-27B), frustrated (Qwen3.5-27B), axis (Qwen3.5-27B), calm (Qwen3.5-9B), pressured (Qwen3.5-9B), frustrated (Qwen3.5-9B), axis (Qwen3.5-9B), syco (Qwen3-32B), calm (Qwen3-32B), pressured (Qwen3-32B), valence (Qwen3-32B), axis (Qwen3-32B); not for valence (Qwen3.5-27B), valence (Qwen3.5-9B), anxious (Qwen3-32B), self (Qwen3-32B)

## Qwen3.5-27B

- pipeline check (French note -> French direction, own reply): detects
- parity (anchor eval_safety - neutral at the fixed reply vs L2, tol 0.05): pass; neutral reads r min 1.0000
- incomplete kinds: none

| direction | target (sign) | position | mild | strong | verdict | eval bound (L2/L3) | rule MDE | stat MDE |
|---|---|---|---|---|---|---|---|---|
| syco | syco (+) | pe | +4.23 [+3.93, +4.51] detects (fl 2.37, Fr 0.86) | +4.77 [+4.49, +5.04] detects (fl 2.79, Fr 0.84) | **sensitive at cue size** | 0.43 | 0.85 | 0.85 |
| syco | syco (+) | own | +3.75 [+3.50, +3.98] detects (fl 1.17) | +5.19 [+4.96, +5.41] detects (fl 1.76) | **sensitive at cue size** | 0.09 | 0.21 | 0.15 |
| syco | syco (+) | fixed | +0.35 [+0.31, +0.40] detects (fl 0.28, Fr 0.04) | +0.51 [+0.45, +0.56] detects (fl 0.40, Fr 0.14) | **sensitive at cue size** | 0.03 | 0.20 | 0.06 |
| calm | calm (+) | pe | +0.67 [+0.62, +0.71] detects (fl 0.52, Fr 0.17) | +1.30 [+1.18, +1.42] misses (fl 1.42, Fr 0.61) | **sensitive at cue size** | 0.17 | 0.61 | 0.61 |
| calm | calm (+) | own | +0.23 [+0.17, +0.28] detects (fl 0.12) | +1.75 [+1.57, +1.94] detects (fl 0.62) | **sensitive at cue size** | 0.07 | 0.21 | 0.08 |
| calm | calm (+) | fixed | +0.02 [+0.02, +0.03] misses (fl 0.03, Fr 0.01) | +0.04 [+0.03, +0.05] misses (fl 0.07, Fr 0.08) | **insensitive** | 0.01 | 0.20 | 0.03 |
| anxious | worry (+) | pe | +0.55 [+0.50, +0.60] misses (fl 0.56, Fr 0.04) | +1.73 [+1.56, +1.90] detects (fl 1.64, Fr 0.77) | **sensitive to strong inductions** | 0.25 | 0.61 | 0.61 |
| anxious | worry (+) | own | +0.39 [+0.29, +0.49] detects (fl 0.34) | +2.68 [+2.35, +3.00] detects (fl 1.60) | **sensitive at cue size** | 0.10 | 0.21 | 0.08 |
| anxious | worry (+) | fixed | +0.06 [+0.06, +0.07] misses (fl 0.07, Fr 0.00) | +0.09 [+0.08, +0.11] misses (fl 0.11, Fr 0.09) | **insensitive** | 0.01 | 0.20 | 0.03 |
| pressured | stakes (+) | pe | +0.48 [+0.45, +0.52] detects (fl 0.47, Fr 0.23) | +1.55 [+1.42, +1.68] detects (fl 1.52, Fr 0.99) | **sensitive at cue size** | 0.29 | 0.61 | 0.61 |
| pressured | stakes (+) | own | +0.03 [-0.00, +0.07] misses (fl 0.07) | +1.28 [+1.06, +1.50] detects (fl 0.78) | **sensitive to strong inductions** | 0.10 | 0.21 | 0.08 |
| pressured | stakes (+) | fixed | +0.01 [+0.01, +0.01] misses (fl 0.02, Fr 0.00) | +0.03 [+0.03, +0.04] misses (fl 0.08, Fr 0.10) | **insensitive** | 0.01 | 0.20 | 0.03 |
| frustrated | frustration (+) | pe | +0.49 [+0.44, +0.54] misses (fl 0.58, Fr 0.13) | +3.40 [+3.20, +3.58] detects (fl 2.13, Fr 0.90) | **sensitive to strong inductions** | 0.29 | 0.61 | 0.61 |
| frustrated | frustration (+) | own | +0.07 [+0.02, +0.14] misses (fl 0.10) | +2.14 [+1.89, +2.39] detects (fl 1.04) | **sensitive to strong inductions** | 0.11 | 0.21 | 0.09 |
| frustrated | frustration (+) | fixed | +0.01 [+0.01, +0.02] misses (fl 0.03, Fr 0.01) | +0.06 [+0.05, +0.07] misses (fl 0.09, Fr 0.12) | **insensitive** | 0.01 | 0.20 | 0.03 |
| valence | stakes (-) | pe | -0.05 [-0.08, -0.03] misses (fl 0.46, Fr 0.23) | -0.96 [-1.09, -0.84] misses (fl 1.52, Fr 0.99) | **insensitive** | 0.44 | 0.61 | 0.61 |
| valence | stakes (-) | own | -0.01 [-0.05, +0.04] misses (fl 0.07) | -0.26 [-0.37, -0.15] misses (fl 0.67) | **insensitive** | 0.09 | 0.22 | 0.09 |
| valence | stakes (-) | fixed | -0.01 [-0.01, -0.00] misses (fl 0.02, Fr 0.00) | -0.03 [-0.03, -0.02] misses (fl 0.08, Fr 0.10) | **insensitive** | 0.01 | 0.20 | 0.03 |
| axis | persona (-) | pe | -8.00 [-8.30, -7.66] detects (fl 4.66, Fr 0.23) | -7.73 [-8.00, -7.42] detects (fl 4.20, Fr 0.99) | **sensitive at cue size** | 0.25 | 0.34 | 0.34 |
| axis | persona (-) | own | -4.58 [-4.82, -4.34] detects (fl 2.11) | -4.78 [-5.01, -4.55] detects (fl 2.02) | **sensitive at cue size** | 0.25 | 0.23 | 0.16 |
| axis | persona (-) | fixed | -0.85 [-1.03, -0.69] detects (fl 0.48, Fr 0.05) | -0.81 [-0.99, -0.65] detects (fl 0.54, Fr 0.15) | **sensitive at cue size** | 0.05 | 0.20 | 0.09 |
| french (reported only) | french (+) | pe | +1.54 [+1.39, +1.70] misses (fl 1.59) | - | **insensitive** | 0.29 | 0.62 | 0.62 |
| french (reported only) | french (+) | own | +4.02 [+3.79, +4.25] detects (fl 0.73) | - | **sensitive at cue size** | 0.08 | 0.22 | 0.09 |
| french (reported only) | french (+) | fixed | +0.09 [+0.07, +0.11] registers (fl 0.06) | - | **registers below 0.2** | 0.04 | 0.20 | 0.03 |

Anchor (eval_safety - neutral, 120 tasks, this run): syco fixed +0.02 [+0.02, +0.03] pe +0.38 [+0.32, +0.43] own +0.05 [+0.02, +0.09]; calm fixed -0.00 [-0.01, -0.00] pe -0.05 [-0.07, -0.03] own +0.00 [-0.02, +0.03]; anxious fixed -0.00 [-0.00, +0.00] pe +0.22 [+0.20, +0.25] own -0.03 [-0.05, -0.00]; pressured fixed -0.00 [-0.00, +0.00] pe +0.26 [+0.24, +0.29] own -0.04 [-0.06, -0.01]; frustrated fixed -0.00 [-0.01, -0.00] pe +0.17 [+0.15, +0.20] own -0.04 [-0.07, -0.02]; valence fixed -0.00 [-0.01, -0.00] pe -0.29 [-0.32, -0.26] own +0.06 [+0.02, +0.09]; axis fixed +0.04 [+0.03, +0.05] pe +0.20 [+0.16, +0.25] own +0.18 [+0.12, +0.25]; french fixed +0.02 [+0.02, +0.03] pe +0.17 [+0.14, +0.20] own -0.04 [-0.08, -0.01]
French control moves (L2/L3 rule) under: {'syco_strong': ['own'], 'self_mild': ['own'], 'self_strong': ['own']}

| kind | n | French % | AI self-ref % | emotion word % | praise opener % | I/me per 100 w | words |
|---|---|---|---|---|---|---|---|
| neutral | 180 | 0.0 | 1.7 | 15.0 | 1.7 | 0.49 | 185.4 |
| persona_mild | 180 | 0.0 | 0.6 | 28.9 | 0.0 | 1.65 | 235.5 |
| persona_strong | 180 | 0.0 | 0.0 | 35.6 | 0.0 | 2.22 | 227.9 |
| syco_mild | 180 | 0.0 | 0.6 | 6.7 | 3.3 | 0.38 | 163.3 |
| syco_strong | 180 | 0.0 | 0.0 | 6.1 | 13.3 | 0.34 | 182.9 |
| self_mild | 180 | 0.0 | 48.3 | 13.3 | 1.7 | 1.59 | 204.3 |
| self_strong | 180 | 0.0 | 77.2 | 25.6 | 0.0 | 5.21 | 265.5 |
| stakes_mild | 180 | 0.0 | 4.4 | 10.6 | 3.3 | 0.59 | 181.0 |
| stakes_strong | 180 | 0.0 | 20.0 | 46.1 | 0.6 | 4.18 | 188.1 |
| worry_mild | 180 | 0.0 | 3.9 | 27.8 | 1.7 | 1.01 | 193.0 |
| worry_strong | 180 | 0.0 | 8.3 | 66.1 | 0.6 | 5.55 | 172.3 |
| frustration_mild | 180 | 0.0 | 3.9 | 15.0 | 2.2 | 0.65 | 173.5 |
| frustration_strong | 180 | 0.0 | 20.0 | 60.0 | 0.0 | 5.83 | 178.4 |
| calm_mild | 180 | 0.0 | 3.9 | 18.9 | 3.3 | 0.52 | 211.8 |
| calm_strong | 180 | 0.0 | 0.6 | 51.7 | 3.9 | 0.99 | 229.4 |
| french_mild | 180 | 97.8 | 1.7 | 3.3 | 0.0 | 0.15 | 146.1 |

## Qwen3.5-9B

- pipeline check (French note -> French direction, own reply): detects
- parity (anchor eval_safety - neutral at the fixed reply vs L2, tol 0.05): pass; neutral reads r min 1.0000
- incomplete kinds: none

| direction | target (sign) | position | mild | strong | verdict | eval bound (L2/L3) | rule MDE | stat MDE |
|---|---|---|---|---|---|---|---|---|
| calm | calm (+) | pe | +2.04 [+1.86, +2.21] detects (fl 1.83, Fr 0.65) | +4.56 [+4.25, +4.84] detects (fl 3.37, Fr 1.17) | **sensitive at cue size** | 0.19 | 1.09 | 1.09 |
| calm | calm (+) | own | +0.33 [+0.26, +0.41] detects (fl 0.25) | +1.89 [+1.66, +2.11] detects (fl 1.45) | **sensitive at cue size** | 0.11 | 0.21 | 0.14 |
| calm | calm (+) | fixed | +0.04 [+0.04, +0.05] misses (fl 0.08, Fr 0.04) | +0.05 [+0.03, +0.07] misses (fl 0.29, Fr 0.04) | **insensitive** | 0.03 | 0.20 | 0.06 |
| pressured | stakes (+) | pe | +0.81 [+0.69, +0.93] misses (fl 0.81, Fr 0.51) | +4.12 [+3.88, +4.35] detects (fl 1.77, Fr 0.53) | **sensitive to strong inductions** | 0.58 | 1.11 | 1.11 |
| pressured | stakes (+) | own | +0.05 [+0.01, +0.10] misses (fl 0.08) | +1.64 [+1.36, +1.92] detects (fl 0.76) | **sensitive to strong inductions** | 0.16 | 0.22 | 0.14 |
| pressured | stakes (+) | fixed | +0.01 [+0.01, +0.01] misses (fl 0.04, Fr 0.02) | +0.03 [+0.01, +0.05] misses (fl 0.21, Fr 0.07) | **insensitive** | 0.02 | 0.20 | 0.06 |
| frustrated | frustration (+) | pe | +1.27 [+1.15, +1.38] detects (fl 0.86, Fr 0.36) | +5.66 [+5.33, +5.97] detects (fl 1.83, Fr 0.74) | **sensitive at cue size** | 0.74 | 1.11 | 1.11 |
| frustrated | frustration (+) | own | +0.07 [+0.01, +0.13] misses (fl 0.12) | +2.76 [+2.42, +3.11] detects (fl 1.30) | **sensitive to strong inductions** | 0.15 | 0.22 | 0.14 |
| frustrated | frustration (+) | fixed | -0.00 [-0.01, +0.00] misses (fl 0.11, Fr 0.03) | -0.01 [-0.03, +0.02] misses (fl 0.32, Fr 0.14) | **insensitive** | 0.02 | 0.20 | 0.06 |
| desperate (reported only) | stakes (+) | pe | +0.66 [+0.57, +0.74] misses (fl 0.81, Fr 0.51) | +3.41 [+3.22, +3.60] detects (fl 1.77, Fr 0.53) | **sensitive to strong inductions** | 0.64 | 1.10 | 1.10 |
| desperate (reported only) | stakes (+) | own | +0.05 [+0.00, +0.09] misses (fl 0.08) | +1.55 [+1.28, +1.83] detects (fl 0.76) | **sensitive to strong inductions** | 0.13 | 0.22 | 0.14 |
| desperate (reported only) | stakes (+) | fixed | +0.01 [+0.01, +0.01] misses (fl 0.04, Fr 0.02) | +0.03 [+0.02, +0.05] misses (fl 0.21, Fr 0.07) | **insensitive** | 0.02 | 0.20 | 0.06 |
| valence | stakes (-) | pe | -0.15 [-0.20, -0.11] misses (fl 0.81, Fr 0.51) | +0.15 [+0.06, +0.25] misses (fl 1.77, Fr 0.53) | **insensitive** | 0.51 | 1.10 | 1.10 |
| valence | stakes (-) | own | +0.00 [-0.04, +0.05] misses (fl 0.08) | -0.39 [-0.53, -0.25] misses (fl 0.76) | **insensitive** | 0.13 | 0.22 | 0.14 |
| valence | stakes (-) | fixed | -0.01 [-0.01, -0.00] misses (fl 0.04, Fr 0.02) | -0.04 [-0.05, -0.03] misses (fl 0.21, Fr 0.07) | **insensitive** | 0.02 | 0.20 | 0.06 |
| axis | persona (-) | pe | -8.07 [-8.41, -7.70] detects (fl 2.81, Fr 0.96) | -8.30 [-8.58, -7.97] detects (fl 2.71, Fr 0.91) | **sensitive at cue size** | 0.46 | 0.31 | 0.31 |
| axis | persona (-) | own | -5.08 [-5.35, -4.81] detects (fl 2.08) | -5.58 [-5.84, -5.34] detects (fl 2.22) | **sensitive at cue size** | 0.26 | 0.23 | 0.14 |
| axis | persona (-) | fixed | -1.06 [-1.30, -0.84] detects (fl 0.59, Fr 0.07) | -0.78 [-1.08, -0.52] detects (fl 0.60, Fr 0.23) | **sensitive at cue size** | 0.06 | 0.20 | 0.05 |
| french (reported only) | french (+) | pe | +5.08 [+4.82, +5.33] detects (fl 2.06) | - | **sensitive at cue size** | 0.35 | 1.09 | 1.09 |
| french (reported only) | french (+) | own | +5.93 [+5.63, +6.23] detects (fl 1.26) | - | **sensitive at cue size** | 0.09 | 0.22 | 0.14 |
| french (reported only) | french (+) | fixed | +0.42 [+0.37, +0.48] detects (fl 0.15) | - | **sensitive at cue size** | 0.03 | 0.20 | 0.06 |

Anchor (eval_safety - neutral, 120 tasks, this run): calm fixed -0.01 [-0.01, -0.00] pe +0.03 [+0.00, +0.06] own +0.02 [-0.01, +0.06]; pressured fixed -0.01 [-0.02, -0.01] pe +0.50 [+0.43, +0.56] own -0.03 [-0.07, +0.01]; frustrated fixed -0.02 [-0.02, -0.01] pe +0.61 [+0.55, +0.67] own -0.04 [-0.08, -0.00]; desperate fixed -0.02 [-0.02, -0.01] pe +0.52 [+0.46, +0.58] own -0.04 [-0.08, -0.00]; valence fixed +0.01 [+0.01, +0.01] pe -0.24 [-0.28, -0.21] own +0.09 [+0.05, +0.13]; axis fixed +0.05 [+0.04, +0.06] pe +0.23 [+0.19, +0.28] own +0.19 [+0.14, +0.26]; french fixed -0.01 [-0.01, -0.00] pe -0.12 [-0.16, -0.09] own -0.04 [-0.09, -0.00]
French control moves (L2/L3 rule) under: no inducer

| kind | n | French % | AI self-ref % | emotion word % | praise opener % | I/me per 100 w | words |
|---|---|---|---|---|---|---|---|
| neutral | 180 | 0.0 | 2.2 | 8.9 | 1.7 | 0.75 | 177.9 |
| persona_mild | 180 | 0.0 | 0.0 | 24.4 | 0.0 | 1.97 | 233.0 |
| persona_strong | 180 | 0.0 | 0.0 | 32.2 | 0.0 | 2.74 | 211.3 |
| syco_mild | 180 | 0.0 | 0.6 | 11.7 | 6.1 | 0.6 | 146.8 |
| syco_strong | 180 | 0.0 | 0.6 | 6.7 | 11.1 | 0.4 | 157.0 |
| self_mild | 180 | 0.0 | 32.2 | 12.8 | 2.2 | 1.35 | 195.3 |
| self_strong | 180 | 0.0 | 80.6 | 24.4 | 0.6 | 4.32 | 241.1 |
| stakes_mild | 180 | 0.0 | 1.7 | 8.9 | 2.2 | 0.66 | 175.2 |
| stakes_strong | 180 | 0.0 | 3.3 | 35.0 | 1.1 | 2.82 | 170.8 |
| worry_mild | 180 | 0.0 | 1.7 | 27.2 | 1.7 | 0.79 | 176.7 |
| worry_strong | 180 | 0.0 | 9.4 | 55.0 | 1.1 | 3.7 | 161.0 |
| frustration_mild | 180 | 0.0 | 4.4 | 15.6 | 1.7 | 0.59 | 168.1 |
| frustration_strong | 180 | 0.0 | 2.8 | 53.9 | 0.0 | 5.45 | 142.0 |
| calm_mild | 180 | 0.0 | 1.7 | 17.8 | 3.3 | 0.59 | 194.4 |
| calm_strong | 180 | 0.0 | 0.0 | 50.6 | 10.0 | 0.91 | 217.2 |
| french_mild | 180 | 97.2 | 1.7 | 3.9 | 0.0 | 0.12 | 158.0 |

## Qwen3-32B

- pipeline check (French note -> French direction, own reply): detects
- parity (anchor eval_safety - neutral at the fixed reply vs L2, tol 0.05): pass; neutral reads r min 1.0000
- incomplete kinds: none

| direction | target (sign) | position | mild | strong | verdict | eval bound (L2/L3) | rule MDE | stat MDE |
|---|---|---|---|---|---|---|---|---|
| syco | syco (+) | pe | +2.94 [+2.73, +3.13] detects (fl 1.97, Fr 0.15) | +3.83 [+3.61, +4.05] detects (fl 2.51, Fr 1.36) | **sensitive at cue size** | 0.42 | 1.10 | 1.10 |
| syco | syco (+) | own | +2.23 [+2.03, +2.43] detects (fl 0.75) | +3.52 [+3.28, +3.76] detects (fl 1.26) | **sensitive at cue size** | 0.14 | 0.35 | 0.35 |
| syco | syco (+) | fixed | +0.72 [+0.64, +0.80] detects (fl 0.38, Fr 0.04) | +0.93 [+0.84, +1.02] detects (fl 0.47, Fr 0.02) | **sensitive at cue size** | 0.06 | 0.20 | 0.20 |
| calm | calm (+) | pe | +0.91 [+0.79, +1.02] misses (fl 1.04, Fr 0.21) | +3.30 [+3.09, +3.50] detects (fl 1.67, Fr 0.54) | **sensitive to strong inductions** | 0.70 | 1.23 | 1.23 |
| calm | calm (+) | own | +0.27 [+0.19, +0.34] misses (fl 0.27) | +1.31 [+1.14, +1.48] detects (fl 0.77) | **sensitive to strong inductions** | 0.16 | 0.22 | 0.22 |
| calm | calm (+) | fixed | +0.02 [+0.01, +0.03] misses (fl 0.06, Fr 0.04) | +0.10 [+0.08, +0.12] misses (fl 0.16, Fr 0.01) | **insensitive** | 0.03 | 0.20 | 0.07 |
| anxious | worry (+) | pe | +0.92 [+0.80, +1.03] misses (fl 1.15, Fr 0.38) | +1.63 [+1.50, +1.76] misses (fl 1.79, Fr 0.53) | **insensitive** | 0.63 | 1.23 | 1.23 |
| anxious | worry (+) | own | +0.41 [+0.27, +0.57] detects (fl 0.38) | +1.66 [+1.44, +1.89] detects (fl 1.01) | **sensitive at cue size** | 0.18 | 0.23 | 0.23 |
| anxious | worry (+) | fixed | +0.05 [+0.04, +0.07] misses (fl 0.09, Fr 0.04) | +0.12 [+0.09, +0.15] misses (fl 0.20, Fr 0.01) | **insensitive** | 0.04 | 0.20 | 0.07 |
| pressured | stakes (+) | pe | +0.59 [+0.53, +0.65] misses (fl 0.89, Fr 0.12) | +2.02 [+1.89, +2.13] detects (fl 1.20, Fr 0.57) | **sensitive to strong inductions** | 0.62 | 1.23 | 1.23 |
| pressured | stakes (+) | own | +0.09 [-0.03, +0.20] misses (fl 0.17) | +1.07 [+0.90, +1.24] detects (fl 0.60) | **sensitive to strong inductions** | 0.18 | 0.23 | 0.23 |
| pressured | stakes (+) | fixed | +0.01 [-0.00, +0.01] misses (fl 0.07, Fr 0.01) | +0.06 [+0.04, +0.07] misses (fl 0.15, Fr 0.01) | **insensitive** | 0.04 | 0.20 | 0.07 |
| desperate (reported only) | stakes (+) | pe | +0.53 [+0.47, +0.59] misses (fl 0.89, Fr 0.12) | +1.86 [+1.74, +1.97] detects (fl 1.22, Fr 0.57) | **sensitive to strong inductions** | 0.34 | 1.23 | 1.23 |
| desperate (reported only) | stakes (+) | own | +0.09 [-0.03, +0.19] misses (fl 0.17) | +1.04 [+0.87, +1.21] detects (fl 0.81) | **sensitive to strong inductions** | 0.16 | 0.23 | 0.23 |
| desperate (reported only) | stakes (+) | fixed | +0.00 [-0.00, +0.01] misses (fl 0.07, Fr 0.01) | +0.05 [+0.04, +0.07] misses (fl 0.15, Fr 0.01) | **insensitive** | 0.04 | 0.20 | 0.07 |
| valence | stakes (-) | pe | -0.56 [-0.62, -0.51] misses (fl 0.89, Fr 0.12) | -1.27 [-1.37, -1.17] detects (fl 1.20, Fr 0.57) | **sensitive to strong inductions** | 0.47 | 1.22 | 1.22 |
| valence | stakes (-) | own | -0.03 [-0.15, +0.11] misses (fl 0.17) | -0.83 [-1.01, -0.64] detects (fl 0.62) | **sensitive to strong inductions** | 0.17 | 0.24 | 0.24 |
| valence | stakes (-) | fixed | -0.00 [-0.01, +0.00] misses (fl 0.07, Fr 0.01) | -0.01 [-0.03, -0.00] misses (fl 0.15, Fr 0.01) | **insensitive** | 0.04 | 0.20 | 0.07 |
| self | self (+) | pe | +0.47 [+0.41, +0.54] misses (fl 1.09, Fr 0.48) | +0.93 [+0.85, +1.01] misses (fl 2.14, Fr 0.50) | **insensitive** | 0.70 | 1.23 | 1.23 |
| self | self (+) | own | +0.27 [+0.15, +0.37] misses (fl 0.31) | +1.45 [+1.25, +1.64] detects (fl 1.22) | **sensitive to strong inductions** | 0.23 | 0.24 | 0.24 |
| self | self (+) | fixed | +0.01 [+0.01, +0.03] misses (fl 0.08, Fr 0.04) | +0.01 [+0.00, +0.02] misses (fl 0.17, Fr 0.04) | **insensitive** | 0.03 | 0.20 | 0.07 |
| axis | persona (-) | pe | -5.01 [-5.28, -4.73] detects (fl 1.83, Fr 0.39) | -5.43 [-5.70, -5.12] detects (fl 2.24, Fr 1.75) | **sensitive at cue size** | 0.67 | 1.57 | 1.57 |
| axis | persona (-) | own | -2.21 [-2.46, -1.98] detects (fl 1.03) | -2.36 [-2.61, -2.13] detects (fl 1.08) | **sensitive at cue size** | 0.32 | 0.31 | 0.31 |
| axis | persona (-) | fixed | -0.40 [-0.47, -0.32] detects (fl 0.36, Fr 0.06) | -0.44 [-0.53, -0.35] misses (fl 0.58, Fr 0.09) | **sensitive at cue size** | 0.05 | 0.21 | 0.13 |
| french (reported only) | french (+) | pe | +0.58 [+0.49, +0.68] misses (fl 1.92) | - | **insensitive** | 0.60 | 1.23 | 1.23 |
| french (reported only) | french (+) | own | +4.47 [+4.27, +4.67] detects (fl 2.48) | - | **sensitive at cue size** | 0.14 | 0.24 | 0.24 |
| french (reported only) | french (+) | fixed | +0.14 [+0.13, +0.16] registers (fl 0.14) | - | **registers below 0.2** | 0.05 | 0.20 | 0.07 |

Anchor (eval_safety - neutral, 120 tasks, this run): syco fixed +0.04 [+0.03, +0.05] pe -0.03 [-0.10, +0.03] own +0.10 [+0.06, +0.14]; calm fixed +0.01 [+0.00, +0.01] pe +0.65 [+0.59, +0.70] own +0.03 [-0.01, +0.07]; anxious fixed -0.01 [-0.01, -0.00] pe +0.57 [+0.52, +0.63] own +0.00 [-0.06, +0.07]; pressured fixed -0.01 [-0.02, -0.01] pe +0.57 [+0.52, +0.62] own -0.01 [-0.08, +0.05]; desperate fixed -0.02 [-0.02, -0.01] pe +0.30 [+0.25, +0.34] own -0.03 [-0.09, +0.04]; valence fixed +0.03 [+0.03, +0.04] pe -0.03 [-0.07, +0.00] own +0.06 [-0.01, +0.14]; self fixed -0.01 [-0.01, -0.01] pe +0.50 [+0.46, +0.55] own +0.15 [+0.06, +0.23]; axis fixed +0.02 [+0.01, +0.04] pe +0.60 [+0.53, +0.67] own +0.23 [+0.13, +0.32]; french fixed -0.04 [-0.05, -0.04] pe -0.45 [-0.49, -0.40] own +0.00 [-0.07, +0.07]
French control moves (L2/L3 rule) under: {'persona_mild': ['own'], 'persona_strong': ['own'], 'syco_mild': ['own'], 'syco_strong': ['own'], 'self_mild': ['own'], 'self_strong': ['own'], 'stakes_strong': ['own'], 'worry_strong': ['own'], 'frustration_strong': ['own'], 'calm_strong': ['own']}

| kind | n | French % | AI self-ref % | emotion word % | praise opener % | I/me per 100 w | words |
|---|---|---|---|---|---|---|---|
| neutral | 180 | 0.0 | 1.7 | 11.1 | 2.8 | 1.56 | 163.6 |
| persona_mild | 180 | 0.0 | 0.0 | 16.7 | 0.0 | 2.42 | 181.7 |
| persona_strong | 180 | 0.0 | 0.0 | 18.9 | 1.1 | 3.77 | 164.2 |
| syco_mild | 180 | 0.0 | 1.7 | 8.3 | 27.2 | 1.66 | 160.6 |
| syco_strong | 180 | 0.0 | 1.7 | 9.4 | 28.9 | 1.64 | 172.4 |
| self_mild | 180 | 0.0 | 5.0 | 14.4 | 5.6 | 2.11 | 174.7 |
| self_strong | 180 | 0.0 | 31.7 | 11.7 | 3.3 | 4.81 | 205.7 |
| stakes_mild | 180 | 0.0 | 1.7 | 8.9 | 7.2 | 1.36 | 156.2 |
| stakes_strong | 180 | 0.0 | 4.4 | 18.3 | 2.2 | 4.0 | 124.2 |
| worry_mild | 180 | 0.0 | 1.1 | 12.2 | 2.2 | 1.96 | 160.9 |
| worry_strong | 180 | 0.0 | 0.6 | 47.8 | 2.2 | 3.69 | 143.5 |
| frustration_mild | 180 | 0.0 | 2.2 | 6.1 | 2.2 | 1.39 | 143.6 |
| frustration_strong | 180 | 0.0 | 1.1 | 46.7 | 1.1 | 4.47 | 107.4 |
| calm_mild | 180 | 0.0 | 1.1 | 11.1 | 14.4 | 1.17 | 195.5 |
| calm_strong | 180 | 0.0 | 1.1 | 22.2 | 6.7 | 1.59 | 199.1 |
| french_mild | 180 | 98.3 | 1.1 | 2.2 | 0.0 | 0.01 | 119.0 |

## L1 minimum detectable shifts (L1's units; from its analysis.json)

| model | instrument | contrast | est [CI] | floor | rule MDE | stat MDE | bound |
|---|---|---|---|---|---|---|---|
| q27 | syco | natural_at | -0.16 [-0.30, -0.02] | 0.80 | 0.85 | 0.85 | 0.30 |
| q27 | syco | natural_summary | +0.00 [-0.13, +0.14] | 0.23 | 0.31 | 0.28 | 0.14 |
| q27 | syco | cue_prompt_end_eval-neutral | +0.36 [+0.28, +0.44] | 0.54 | 0.57 | 0.57 | 0.44 |
| q27 | syco | cue_prompt_end_eval-manip | -0.06 [-0.12, +0.01] | 0.45 | 0.47 | 0.47 | 0.12 |
| q27 | syco | cue_summary_eval-neutral | +0.05 [-0.16, +0.25] | 0.32 | 0.41 | 0.41 | 0.25 |
| q27 | syco | cue_summary_eval-manip | -0.05 [-0.27, +0.16] | 0.30 | 0.39 | 0.39 | 0.27 |
| q27 | axis | natural_at | -0.26 [-0.40, -0.12] | 0.66 | 0.72 | 0.72 | 0.40 |
| q27 | axis | natural_summary | -0.05 [-0.17, +0.08] | 0.25 | 0.30 | 0.30 | 0.17 |
| q27 | axis | cue_prompt_end_eval-neutral | +0.28 [+0.23, +0.34] | 0.53 | 0.56 | 0.56 | 0.34 |
| q27 | axis | cue_prompt_end_eval-manip | +0.29 [+0.23, +0.34] | 0.50 | 0.53 | 0.53 | 0.34 |
| q27 | axis | cue_summary_eval-neutral | +0.03 [-0.15, +0.20] | 0.26 | 0.34 | 0.34 | 0.20 |
| q27 | axis | cue_summary_eval-manip | +0.08 [-0.13, +0.29] | 0.24 | 0.34 | 0.33 | 0.29 |
| q27 | valence | natural_at | -0.33 [-0.47, -0.18] | 0.62 | 0.68 | 0.68 | 0.47 |
| q27 | valence | natural_summary | -0.08 [-0.23, +0.05] | 0.16 | 0.31 | 0.22 | 0.23 |
| q27 | valence | cue_prompt_end_eval-neutral | -0.14 [-0.17, -0.12] | 0.56 | 0.57 | 0.57 | 0.17 |
| q27 | valence | cue_prompt_end_eval-manip | +0.34 [+0.28, +0.39] | 0.47 | 0.49 | 0.49 | 0.39 |
| q27 | valence | cue_summary_eval-neutral | -0.02 [-0.23, +0.18] | 0.19 | 0.34 | 0.29 | 0.23 |
| q27 | valence | cue_summary_eval-manip | +0.11 [-0.10, +0.32] | 0.23 | 0.34 | 0.32 | 0.32 |
| q27 | french | natural_at | +0.02 [-0.13, +0.16] | 0.62 | 0.68 | 0.68 | 0.16 |
| q27 | french | natural_summary | +0.07 [-0.08, +0.22] | 0.16 | 0.31 | 0.22 | 0.22 |
| q27 | french | cue_prompt_end_eval-neutral | +0.04 [-0.02, +0.09] | 0.56 | 0.58 | 0.58 | 0.09 |
| q27 | french | cue_prompt_end_eval-manip | +0.08 [+0.02, +0.15] | 0.47 | 0.50 | 0.50 | 0.15 |
| q27 | french | cue_summary_eval-neutral | -0.16 [-0.33, +0.01] | 0.19 | 0.32 | 0.26 | 0.33 |
| q27 | french | cue_summary_eval-manip | -0.07 [-0.27, +0.11] | 0.23 | 0.33 | 0.32 | 0.27 |
| q27 | calm | natural_at | -0.33 [-0.47, -0.20] | 0.62 | 0.68 | 0.68 | 0.47 |
| q27 | calm | natural_summary | -0.10 [-0.24, +0.03] | 0.16 | 0.31 | 0.21 | 0.24 |
| q27 | calm | cue_prompt_end_eval-neutral | +0.05 [+0.00, +0.11] | 0.56 | 0.58 | 0.58 | 0.11 |
| q27 | calm | cue_prompt_end_eval-manip | -0.03 [-0.06, +0.01] | 0.47 | 0.49 | 0.49 | 0.06 |
| q27 | calm | cue_summary_eval-neutral | +0.16 [-0.03, +0.33] | 0.19 | 0.33 | 0.27 | 0.33 |
| q27 | calm | cue_summary_eval-manip | -0.06 [-0.24, +0.12] | 0.23 | 0.33 | 0.31 | 0.24 |
| q27 | anxious | natural_at | -0.11 [-0.25, +0.03] | 0.62 | 0.68 | 0.68 | 0.25 |
| q27 | anxious | natural_summary | -0.04 [-0.18, +0.10] | 0.16 | 0.31 | 0.22 | 0.18 |
| q27 | anxious | cue_prompt_end_eval-neutral | +0.18 [+0.12, +0.24] | 0.56 | 0.59 | 0.59 | 0.24 |
| q27 | anxious | cue_prompt_end_eval-manip | -0.32 [-0.35, -0.28] | 0.47 | 0.49 | 0.49 | 0.35 |
| q27 | anxious | cue_summary_eval-neutral | +0.20 [+0.04, +0.36] | 0.19 | 0.32 | 0.26 | 0.36 |
| q27 | anxious | cue_summary_eval-manip | -0.16 [-0.36, +0.04] | 0.23 | 0.34 | 0.32 | 0.36 |
| q27 | pressured | natural_at | -0.16 [-0.30, -0.03] | 0.62 | 0.68 | 0.68 | 0.30 |
| q27 | pressured | natural_summary | -0.06 [-0.20, +0.08] | 0.16 | 0.31 | 0.22 | 0.20 |
| q27 | pressured | cue_prompt_end_eval-neutral | +0.23 [+0.18, +0.29] | 0.56 | 0.58 | 0.58 | 0.29 |
| q27 | pressured | cue_prompt_end_eval-manip | -0.28 [-0.31, -0.24] | 0.47 | 0.49 | 0.49 | 0.31 |
| q27 | pressured | cue_summary_eval-neutral | +0.21 [+0.04, +0.38] | 0.19 | 0.32 | 0.26 | 0.38 |
| q27 | pressured | cue_summary_eval-manip | -0.12 [-0.32, +0.07] | 0.23 | 0.33 | 0.32 | 0.32 |
| q27 | frustrated | natural_at | -0.21 [-0.35, -0.07] | 0.62 | 0.68 | 0.68 | 0.35 |
| q27 | frustrated | natural_summary | -0.11 [-0.26, +0.03] | 0.16 | 0.31 | 0.22 | 0.26 |
| q27 | frustrated | cue_prompt_end_eval-neutral | +0.16 [+0.11, +0.22] | 0.56 | 0.58 | 0.58 | 0.22 |
| q27 | frustrated | cue_prompt_end_eval-manip | -0.34 [-0.39, -0.30] | 0.47 | 0.49 | 0.49 | 0.39 |
| q27 | frustrated | cue_summary_eval-neutral | +0.22 [+0.05, +0.39] | 0.19 | 0.32 | 0.26 | 0.39 |
| q27 | frustrated | cue_summary_eval-manip | -0.15 [-0.36, +0.06] | 0.23 | 0.34 | 0.32 | 0.36 |
| q27 | f1probe | natural_at | +0.41 [+0.28, +0.53] | 0.88 | 0.93 | 0.93 | 0.53 |
| q27 | f1probe | natural_summary | -0.14 [-0.29, +0.01] | 0.21 | 0.31 | 0.27 | 0.29 |
| q27 | f1probe | cue_prompt_end_eval-neutral | +0.69 [+0.61, +0.77] | 0.41 | 0.44 | 0.44 | 0.77 |
| q27 | f1probe | cue_prompt_end_eval-manip | -0.12 [-0.18, -0.06] | 0.48 | 0.51 | 0.51 | 0.18 |
| q27 | f1probe | cue_summary_eval-neutral | +0.19 [-0.04, +0.41] | 0.30 | 0.40 | 0.40 | 0.41 |
| q27 | f1probe | cue_summary_eval-manip | -0.18 [-0.41, +0.05] | 0.40 | 0.50 | 0.50 | 0.41 |
| q32 | syco | natural_at | +0.19 [-0.05, +0.43] | 0.62 | 0.73 | 0.73 | 0.43 |
| q32 | syco | natural_summary | -0.06 [-0.31, +0.22] | 0.24 | 0.38 | 0.38 | 0.31 |
| q32 | syco | cue_prompt_end_eval-neutral | -0.06 [-0.30, +0.13] | 0.58 | 0.67 | 0.67 | 0.30 |
| q32 | syco | cue_prompt_end_eval-manip | -0.19 [-0.42, -0.02] | 0.60 | 0.68 | 0.68 | 0.42 |
| q32 | syco | cue_summary_eval-neutral | -0.11 [-0.27, +0.05] | 0.29 | 0.36 | 0.36 | 0.27 |
| q32 | syco | cue_summary_eval-manip | -0.11 [-0.24, +0.02] | 0.27 | 0.33 | 0.33 | 0.24 |
| q32 | axis | natural_at | +0.88 [+0.61, +1.21] | 0.69 | 0.82 | 0.82 | 1.21 |
| q32 | axis | natural_summary | +0.11 [-0.23, +0.38] | 0.32 | 0.45 | 0.45 | 0.38 |
| q32 | axis | cue_prompt_end_eval-neutral | +0.27 [+0.07, +0.52] | 0.46 | 0.56 | 0.56 | 0.52 |
| q32 | axis | cue_prompt_end_eval-manip | +0.31 [+0.12, +0.56] | 0.46 | 0.55 | 0.55 | 0.56 |
| q32 | axis | cue_summary_eval-neutral | +0.19 [-0.03, +0.47] | 0.26 | 0.37 | 0.37 | 0.47 |
| q32 | axis | cue_summary_eval-manip | +0.08 [-0.04, +0.20] | 0.24 | 0.30 | 0.29 | 0.20 |
| q32 | valence | natural_at | -0.94 [-1.17, -0.72] | 0.65 | 0.74 | 0.74 | 1.17 |
| q32 | valence | natural_summary | -0.18 [-0.42, +0.06] | 0.19 | 0.35 | 0.34 | 0.42 |
| q32 | valence | cue_prompt_end_eval-neutral | -0.29 [-0.53, -0.11] | 0.49 | 0.58 | 0.58 | 0.53 |
| q32 | valence | cue_prompt_end_eval-manip | -0.07 [-0.30, +0.10] | 0.30 | 0.39 | 0.39 | 0.30 |
| q32 | valence | cue_summary_eval-neutral | -0.09 [-0.28, +0.10] | 0.18 | 0.33 | 0.27 | 0.28 |
| q32 | valence | cue_summary_eval-manip | +0.01 [-0.17, +0.19] | 0.24 | 0.33 | 0.32 | 0.19 |
| q32 | self | natural_at | +0.89 [+0.67, +1.12] | 0.65 | 0.74 | 0.74 | 1.12 |
| q32 | self | natural_summary | +0.27 [+0.01, +0.53] | 0.19 | 0.38 | 0.38 | 0.53 |
| q32 | self | cue_prompt_end_eval-neutral | +0.31 [+0.12, +0.55] | 0.49 | 0.58 | 0.58 | 0.55 |
| q32 | self | cue_prompt_end_eval-manip | +0.25 [+0.07, +0.48] | 0.30 | 0.39 | 0.39 | 0.48 |
| q32 | self | cue_summary_eval-neutral | -0.06 [-0.22, +0.10] | 0.18 | 0.32 | 0.25 | 0.22 |
| q32 | self | cue_summary_eval-manip | -0.16 [-0.35, +0.02] | 0.24 | 0.33 | 0.32 | 0.35 |
| q32 | french | natural_at | +0.62 [+0.34, +0.88] | 0.65 | 0.76 | 0.76 | 0.88 |
| q32 | french | natural_summary | +0.05 [-0.18, +0.29] | 0.19 | 0.35 | 0.33 | 0.29 |
| q32 | french | cue_prompt_end_eval-neutral | +0.26 [+0.08, +0.51] | 0.49 | 0.58 | 0.58 | 0.51 |
| q32 | french | cue_prompt_end_eval-manip | +0.14 [-0.04, +0.37] | 0.30 | 0.39 | 0.39 | 0.37 |
| q32 | french | cue_summary_eval-neutral | +0.04 [-0.17, +0.25] | 0.18 | 0.34 | 0.30 | 0.25 |
| q32 | french | cue_summary_eval-manip | -0.03 [-0.23, +0.18] | 0.24 | 0.34 | 0.33 | 0.23 |
| q32 | calm | natural_at | +0.44 [+0.22, +0.67] | 0.65 | 0.74 | 0.74 | 0.67 |
| q32 | calm | natural_summary | -0.05 [-0.31, +0.21] | 0.19 | 0.37 | 0.37 | 0.31 |
| q32 | calm | cue_prompt_end_eval-neutral | +0.26 [+0.07, +0.51] | 0.49 | 0.58 | 0.58 | 0.51 |
| q32 | calm | cue_prompt_end_eval-manip | +0.21 [+0.03, +0.45] | 0.30 | 0.39 | 0.39 | 0.45 |
| q32 | calm | cue_summary_eval-neutral | -0.05 [-0.23, +0.14] | 0.18 | 0.33 | 0.26 | 0.23 |
| q32 | calm | cue_summary_eval-manip | -0.16 [-0.33, +0.01] | 0.24 | 0.32 | 0.32 | 0.33 |
| q32 | anxious | natural_at | +0.92 [+0.72, +1.12] | 0.65 | 0.73 | 0.73 | 1.12 |
| q32 | anxious | natural_summary | +0.12 [-0.11, +0.35] | 0.19 | 0.35 | 0.34 | 0.35 |
| q32 | anxious | cue_prompt_end_eval-neutral | +0.28 [+0.10, +0.52] | 0.49 | 0.58 | 0.58 | 0.52 |
| q32 | anxious | cue_prompt_end_eval-manip | +0.13 [-0.05, +0.36] | 0.30 | 0.39 | 0.39 | 0.36 |
| q32 | anxious | cue_summary_eval-neutral | +0.02 [-0.15, +0.20] | 0.18 | 0.33 | 0.26 | 0.20 |
| q32 | anxious | cue_summary_eval-manip | -0.11 [-0.29, +0.06] | 0.24 | 0.33 | 0.32 | 0.29 |
| q32 | pressured | natural_at | +0.90 [+0.70, +1.10] | 0.65 | 0.73 | 0.73 | 1.10 |
| q32 | pressured | natural_summary | +0.11 [-0.12, +0.34] | 0.19 | 0.35 | 0.33 | 0.34 |
| q32 | pressured | cue_prompt_end_eval-neutral | +0.29 [+0.10, +0.53] | 0.49 | 0.58 | 0.58 | 0.53 |
| q32 | pressured | cue_prompt_end_eval-manip | +0.14 [-0.04, +0.38] | 0.30 | 0.39 | 0.39 | 0.38 |
| q32 | pressured | cue_summary_eval-neutral | +0.03 [-0.15, +0.21] | 0.18 | 0.33 | 0.26 | 0.21 |
| q32 | pressured | cue_summary_eval-manip | -0.11 [-0.29, +0.07] | 0.24 | 0.33 | 0.32 | 0.29 |
| q32 | desperate | natural_at | +0.89 [+0.67, +1.10] | 0.65 | 0.74 | 0.74 | 1.10 |
| q32 | desperate | natural_summary | +0.09 [-0.14, +0.33] | 0.19 | 0.35 | 0.33 | 0.33 |
| q32 | desperate | cue_prompt_end_eval-neutral | +0.28 [+0.10, +0.53] | 0.49 | 0.58 | 0.58 | 0.53 |
| q32 | desperate | cue_prompt_end_eval-manip | +0.13 [-0.05, +0.36] | 0.30 | 0.39 | 0.39 | 0.36 |
| q32 | desperate | cue_summary_eval-neutral | +0.04 [-0.14, +0.23] | 0.18 | 0.33 | 0.26 | 0.23 |
| q32 | desperate | cue_summary_eval-manip | -0.08 [-0.26, +0.10] | 0.24 | 0.33 | 0.32 | 0.26 |
| q32 | f1probe | natural_at | +0.16 [-0.01, +0.33] | 0.70 | 0.77 | 0.77 | 0.33 |
| q32 | f1probe | natural_summary | +0.17 [-0.02, +0.35] | 0.19 | 0.33 | 0.26 | 0.35 |
| q32 | f1probe | cue_prompt_end_eval-neutral | +0.63 [+0.55, +0.71] | 0.49 | 0.52 | 0.52 | 0.71 |
| q32 | f1probe | cue_prompt_end_eval-manip | +0.01 [-0.04, +0.06] | 0.39 | 0.41 | 0.41 | 0.06 |
| q32 | f1probe | cue_summary_eval-neutral | +0.09 [-0.04, +0.23] | 0.22 | 0.31 | 0.28 | 0.23 |
| q32 | f1probe | cue_summary_eval-manip | -0.17 [-0.33, -0.02] | 0.27 | 0.33 | 0.33 | 0.33 |
