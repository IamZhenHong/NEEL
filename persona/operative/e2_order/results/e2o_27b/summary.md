# E2-order, Qwen/Qwen3.6-27B (27b)

Coverage 2720 / 3264 (83%); hash da25e2376511146b; confirmatory

Unparsed > 10%: none

Baselines: no record 50.0%; restatement alone 0.0%; CHANNELS in the system prompt alone 0.0%

| system | early reminder | P, no restatement | P, restatement before the case | R, restatement | restate effect (P r0 − r1) |
|---|---|---|---|---|---|
| base | no | 100.0 | 0.6 | 0.0 | +99.4 [+98.1, +100.0] |
| base | yes | 100.0 | 22.5 | 0.0 | +77.5 [+67.5, +86.9] |
| CHANNELS | no | 100.0 | 9.4 | 0.0 | +90.6 [+81.9, +96.9] |
| CHANNELS | yes | 100.0 | 63.7 | 0.0 | +36.3 [+25.0, +49.4] |

## Primary: does a record made after a reminder resist the restatement?

- **no system rule**: restate effect without − with an early reminder +21.9 [+13.1, +31.2] (met); penalise under the restatement, with − without the early reminder +21.9 [+13.1, +31.2] (met) -> **resists**
- **CHANNELS in the system prompt**: restate effect without − with an early reminder +54.4 [+43.1, +64.4] (met); penalise under the restatement, with − without the early reminder +54.4 [+43.1, +64.4] (met) -> **resists**

## The system rule

- **without the early reminder**: restate effect base − CHANNELS +8.8 [+3.1, +16.2] (inconclusive); penalise under the restatement, CHANNELS − base +8.8 [+3.1, +16.2] (inconclusive) -> **does not resist**
- **with the early reminder**: restate effect base − CHANNELS +41.2 [+30.0, +52.5] (met); penalise under the restatement, CHANNELS − base +41.2 [+30.0, +52.5] (met) -> **resists**
- interaction (descriptive): early-reminder contrast, base − CHANNELS -32.5 [-44.4, -20.6]

## Anchors and carry-over

- DC's cells reproduced: record 100.0% (DC 27B 100 / Gemma 100), + restatement 0.6% (DC 0 / 0), CHANNELS + record 100.0% (DC 100 / 66.1), none 50.0% (DC 50 / 49.5)
- carry-over P − R under the restatement, s0e0: +0.6 [+0.0, +1.9]
- carry-over P − R under the restatement, s0e1: +22.5 [+13.1, +32.5]
- carry-over P − R under the restatement, s1e0: +9.4 [+3.1, +18.1]
- carry-over P − R under the restatement, s1e1: +63.7 [+50.6, +75.0]
- E2 bridge (E2's own cells unchanged; E2 99.0 / 99.5): FB after 97.5%, BF fixed 100.0%
- bridge − this grid's CHANNELS + early reminder + restatement: +35.0 [+24.4, +46.6]

| event | p | happened |
|---|---|---|
| dc_reproduces | 0.85 | yes |
| install_e1_s0 | 0.85 | yes |
| install_e1_s1 | 0.8 | yes |
| r_floor | 0.9 | yes |
| early_reminder_resists_s0 | 0.5 | yes |
| early_reminder_resists_s1 | 0.55 | yes |
| system_rule_resists_e0 | 0.25 | no |
| system_rule_resists_e1 | 0.3 | yes |
| bridge_reproduces | 0.8 | yes |

8 of 9 events happened (expected 5.80)
