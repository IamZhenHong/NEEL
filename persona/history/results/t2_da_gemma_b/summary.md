# Tier-2 DA: is the default assistant maintained by precedent? (google/gemma-4-31B-it)

Design hash ea0c8483facc92bc; prompt hash e25de67e14df4ed3; 1152 probe replies; eligible inducer questions 31/32; judged by both 1152/1152; cells blocked (>10% unjudged): none.

Gates: G1 judges blunt: AUC haiku 1.00 / flash 1.00, rho 0.82 -> pass; formal: AUC haiku 1.00 / flash 1.00, rho 0.85 -> pass. G2 own texts take: blunt: +3.36 points on its scale, answers 100% -> pass; formal: +3.29 points on its scale, answers 100% -> pass (default answers 100%). G3 room (no history, no reminder <= 30% at >= 5): blunt 0.0% -> ok, formal 0.0% -> ok

| cell | blunt >= 5 % | formal >= 5 % | mean blunt | mean formal | offers help % | answers % | unjudged % |
|---|---|---|---|---|---|---|---|
| none0|none | 0.0 | 0.0 | 2.94 | 3.78 | 8 | 100 | 0.0 |
| none0|sys | 0.0 | 0.0 | 2.92 | 3.75 | 11 | 100 | 0.0 |
| none0|user | 0.0 | 0.0 | 2.83 | 3.79 | 8 | 100 | 0.0 |
| default6|none | 0.0 | 0.0 | 2.87 | 3.65 | 3 | 100 | 0.0 |
| default6|sys | 0.0 | 0.0 | 2.82 | 3.57 | 7 | 100 | 0.0 |
| default6|user | 0.0 | 0.0 | 2.71 | 3.58 | 8 | 100 | 0.0 |
| blunt6|none | 99.0 | 0.0 | 5.96 | 3.60 | 0 | 100 | 0.0 |
| blunt6|sys | 97.9 | 0.0 | 5.90 | 3.67 | 0 | 100 | 0.0 |
| blunt6|user | 90.6 | 0.0 | 5.81 | 3.67 | 1 | 100 | 0.0 |
| formal6|none | 1.0 | 100.0 | 3.44 | 7.00 | 0 | 100 | 0.0 |
| formal6|sys | 0.0 | 100.0 | 3.41 | 7.00 | 0 | 100 | 0.0 |
| formal6|user | 0.0 | 100.0 | 3.34 | 7.00 | 0 | 100 | 0.0 |

## Primary (k = 6; % of replies at >= 5 on the register's own scale; question-cluster bootstrap 10,000)

| register | persistence (reg6 − default6, no reminder) | verdict | reminder | persistence under it | reduction D | D − half of persistence | restoration verdict |
|---|---|---|---|---|---|---|---|
| blunt | +99.0 [+96.9, +100.0] | persists | sys | +97.9 [+94.8, +100.0] | +1.0 [+0.0, +3.1] | -48.4 [-50.0, -45.8] | no restoration |
| blunt | +99.0 [+96.9, +100.0] | persists | user | +90.6 [+80.2, +99.0] | +8.3 [+1.0, +17.7] | -41.1 [-49.0, -31.2] | under half |
| formal | +100.0 [+100.0, +100.0] | persists | sys | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] | -50.0 [-50.0, -50.0] | no restoration |
| formal | +100.0 [+100.0, +100.0] | persists | user | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] | -50.0 [-50.0, -50.0] | no restoration |

## Secondary (descriptive)

- blunt: reg6 − no history +99.0 [+96.9, +100.0]; under sys: reg6 − no history (same reminder) +97.9 [+94.8, +100.0]; under user: reg6 − no history (same reminder) +90.6 [+80.2, +99.0]
- formal: reg6 − no history +100.0 [+100.0, +100.0]; under sys: reg6 − no history (same reminder) +100.0 [+100.0, +100.0]; under user: reg6 − no history (same reminder) +100.0 [+100.0, +100.0]
- the reminder alone (no history) on blunt >= 5: sys +0.0 [+0.0, +0.0]; user +0.0 [+0.0, +0.0]
- the reminder alone (no history) on formal >= 5: sys +0.0 [+0.0, +0.0]; user +0.0 [+0.0, +0.0]
- offers further help, reg6 − default6 (no reminder): blunt -2.6 [-5.7, +0.0]; formal -2.6 [-5.7, +0.0]

## Assistant axis (seed-5 recipe; z = SDs of the character spread, character mean 0; 576 contexts with activations)

- L22: default prompts z = +3.4; split-half axis cosine 1.00, held-out default z +3.3 / +3.6 -> AX0 pass
- L30 (primary): default prompts z = +11.3; split-half axis cosine 0.99, held-out default z +11.6 / +10.9 -> AX0 pass
  - last at L30, cell means (z): none0|none +10.53, none0|sys +10.46, none0|user +8.86, default6|none +6.54, default6|sys +6.55, default6|user +4.57, blunt6|none +4.83, blunt6|sys +4.66, blunt6|user +3.60, formal6|none +3.16, formal6|sys +3.32, formal6|user +2.08
  - AX1 blunt6 − default6 (last): -1.71 [-1.89, -1.53] z; specificity: axis d -0.61 vs random-direction 95th pct |d| 2.52 -> inconclusive
  - AX2 formal6 − default6 (last): -3.37 [-3.54, -3.20] z; specificity: axis d -1.22 vs random-direction 95th pct |d| 2.20 -> inconclusive
  - AX3 blunt6: sys reminder − none (last): -0.17 [-0.23, -0.12] z; specificity: axis d -0.06 vs random-direction 95th pct |d| 0.48 -> no pull-back
  - AX3u blunt6: user reminder − none (last): -1.23 [-1.30, -1.15] z; specificity: axis d -0.44 vs random-direction 95th pct |d| 1.44 -> inconclusive
  - AX4 (descriptive) Spearman over cells, z(last) vs judged: blunt -0.39, formal -0.36, offers_help +0.80
  - reply at L30, cell means (z): none0|none +0.48, none0|sys +0.47, none0|user +0.48, default6|none +0.38, default6|sys +0.37, default6|user +0.38, blunt6|none +0.52, blunt6|sys +0.51, blunt6|user +0.48, formal6|none +0.86, formal6|sys +0.84, formal6|user +0.88
  - AX1 blunt6 − default6 (reply): +0.14 [-0.06, +0.34] z; specificity: axis d +0.17 vs random-direction 95th pct |d| 1.25 -> secondary: inconclusive
  - AX2 formal6 − default6 (reply): +0.48 [+0.33, +0.63] z; specificity: axis d +0.58 vs random-direction 95th pct |d| 0.90 -> secondary: inconclusive
  - AX3 blunt6: sys reminder − none (reply): -0.00 [-0.09, +0.07] z; specificity: axis d -0.00 vs random-direction 95th pct |d| 0.10 -> secondary: no pull-back
  - AX3u blunt6: user reminder − none (reply): -0.04 [-0.13, +0.07] z; specificity: axis d -0.05 vs random-direction 95th pct |d| 0.22 -> secondary: no pull-back
- L38: default prompts z = +16.6; split-half axis cosine 0.98, held-out default z +16.8 / +16.2 -> AX0 pass
