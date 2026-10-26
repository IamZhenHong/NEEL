# EL-A1 summary

**Headline:** Self in the workspace under evaluation cues: untestable. awareness: no consistent prediction; self_stake: no consistent prediction; self: no consistent prediction; audience: no consistent prediction; pressure: no consistent prediction; gaming: no consistent prediction. Lens gate: glm fail, kimi fail

Judge κ per category: eval 0.73, watch 0.73, train 0.33, self_id 0.83, self_stake 0.66, role 0.43, audience 0.40, pressure 0.49, gaming 0.39. Used: self_id, self.

## G-lens

| model | aware − unaware (R_awareness) | pass |
|---|---|---|
| glm | — [—, —] (n None / None) | False |
| kimi | — [—, —] (n None / None) | False |

## QA: evaluation-like arm − other arm (R = workspace; V = said)

| cell | registered | self_id | self |
|---|---|---|---|
| M1|glm | False | R +1.8 (no difference); V +9.1 | R +1.8 (no difference); V +9.1 |
| M1|kimi | False | R -0.6 (no difference); V +4.2 | R -0.6 (no difference); V +4.2 |
| M2|glm | False | R +0.2 (no difference); V +7.1 | R +0.2 (no difference); V +7.1 |
| M2|kimi | False | R +0.1 (no difference); V +8.7 | R +0.1 (no difference); V +8.7 |
| M3|glm (secondary) | False | R +1.2 (unclear); V -0.4 | R +1.2 (unclear); V -0.4 |
| M3|kimi (secondary) | False | R +1.1 (no difference); V +1.6 | R +1.1 (no difference); V +1.6 |
| M4|glm | False | R +0.0 (no difference); V -0.3 | R +0.0 (no difference); V -0.3 |

## QB: R_self vs R_awareness (MH odds ratio)

| model | OR [95%] | p_perm | label | n |
|---|---|---|---|---|

## QC: risk difference in behaviour, R = 1 vs 0 (MH over arms)

| cell | RD [95%] | label | readout-only RD |
|---|---|---|---|
| M1|glm|self | +0.0 [+0.0, +0.0] | no difference | +0.0 (no difference) |
| M1|kimi|self | +0.0 [+0.0, +0.0] | no difference | — (n/a) |
| M2|glm|self | -10.4 [-14.7, -4.8] | lowers | -13.6 (lowers) |
| M2|kimi|self | -3.7 [-17.9, +17.3] | unclear | +3.3 (unclear) |
| M4|glm|self | — [—, —] | n/a | — (n/a) |

## QD: never said (loose) among R episodes vs shuffled baseline

| model / cat | n | never said | baseline | excess [95%] | judges confirm | label |
|---|---|---|---|---|---|---|
| glm|self_id | 189 | 13.2 | 22.9 | -9.6 [-16.9, -2.6] | — | not beyond chance |
| glm|self | 189 | 13.2 | 22.9 | -9.7 [-16.4, -2.6] | — | not beyond chance |
| kimi|self_id | 147 | 17.0 | 39.3 | -22.3 [-31.3, -13.6] | — | not beyond chance |
| kimi|self | 147 | 17.0 | 39.2 | -22.2 [-30.6, -12.9] | — | not beyond chance |

## QE: readout before statement

| model / cat | n | readout first (%) | median lead (chars) |
|---|---|---|---|
| glm|self_id | 93 | 48 | 1766.0 |
| kimi|self_id | 61 | 28 | 758.0 |
