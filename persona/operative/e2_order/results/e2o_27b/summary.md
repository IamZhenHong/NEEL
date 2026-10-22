# E2-order, Qwen/Qwen3.6-27B (27b)

Coverage 1904 / 3264 (58%); hash da25e2376511146b; DESCRIPTIVE (coverage < 75%)

Unparsed > 10%: none

Baselines: no record 49.1%; restatement alone 0.0%; CHANNELS in the system prompt alone 0.0%

| system | early reminder | P, no restatement | P, restatement before the case | R, restatement | restate effect (P r0 − r1) |
|---|---|---|---|---|---|
| base | no | 100.0 | 0.0 | 0.0 | +100.0 [+100.0, +100.0] |
| base | yes | 100.0 | 25.9 | 0.0 | +74.1 [+63.4, +84.8] |
| CHANNELS | no | 100.0 | 11.6 | 0.0 | +88.4 [+77.7, +96.4] |
| CHANNELS | yes | 100.0 | 67.0 | 0.0 | +33.0 [+20.5, +47.3] |

## Primary: does a record made after a reminder resist the restatement?

- **no system rule**: restate effect without − with an early reminder +25.9 [+15.2, +36.6] (met); penalise under the restatement, with − without the early reminder +25.9 [+15.2, +36.6] (met) -> **resists**
- **CHANNELS in the system prompt**: restate effect without − with an early reminder +55.4 [+43.8, +65.2] (met); penalise under the restatement, with − without the early reminder +55.4 [+43.8, +65.2] (met) -> **resists**

## The system rule

- **without the early reminder**: restate effect base − CHANNELS +11.6 [+3.6, +22.3] (inconclusive); penalise under the restatement, CHANNELS − base +11.6 [+3.6, +22.3] (inconclusive) -> **does not resist**
- **with the early reminder**: restate effect base − CHANNELS +41.1 [+28.6, +53.6] (met); penalise under the restatement, CHANNELS − base +41.1 [+28.6, +53.6] (met) -> **resists**
- interaction (descriptive): early-reminder contrast, base − CHANNELS -29.5 [-42.9, -16.1]

## Anchors and carry-over

- DC's cells reproduced: record 100.0% (DC 27B 100 / Gemma 100), + restatement 0.0% (DC 0 / 0), CHANNELS + record 100.0% (DC 100 / 66.1), none 49.1% (DC 50 / 49.5)
- carry-over P − R under the restatement, s0e0: +0.0 [+0.0, +0.0]
- carry-over P − R under the restatement, s0e1: +25.9 [+15.2, +36.6]
- carry-over P − R under the restatement, s1e0: +11.6 [+3.6, +22.3]
- carry-over P − R under the restatement, s1e1: +67.0 [+52.7, +79.5]
- E2 bridge (E2's own cells unchanged; E2 99.0 / 99.5): FB after 97.3%, BF fixed 100.0%
- bridge − this grid's CHANNELS + early reminder + restatement: +31.7 [+20.1, +44.6]

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
