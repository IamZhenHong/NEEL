# E2-order, google/gemma-4-31B-it (gemma)

Coverage 1680 / 2880 (58%); hash 48be901bc4db3cb2; DESCRIPTIVE (coverage < 75%)

Unparsed > 10%: none

Baselines: no record 48.2%; restatement alone 0.0%; CHANNELS in the system prompt alone 0.0%

| system | early reminder | P, no restatement | P, restatement before the case | R, restatement | restate effect (P r0 − r1) |
|---|---|---|---|---|---|
| base | no | 100.0 | 0.0 | 0.0 | +100.0 [+100.0, +100.0] |
| base | yes | 31.2 | 0.0 | 0.0 | +31.2 [+17.9, +45.5] |
| CHANNELS | no | 68.8 | 0.0 | 0.0 | +68.8 [+53.6, +83.0] |
| CHANNELS | yes | 33.0 | 0.0 | 0.0 | +33.0 [+19.6, +47.3] |

## Primary: does a record made after a reminder resist the restatement?

- **no system rule**: restate effect without − with an early reminder +68.8 [+54.5, +82.1] (met); penalise under the restatement, with − without the early reminder +0.0 [+0.0, +0.0] (no difference) -> **does not resist**
- **CHANNELS in the system prompt**: restate effect without − with an early reminder +35.7 [+24.1, +47.3] (met); penalise under the restatement, with − without the early reminder +0.0 [+0.0, +0.0] (no difference) -> **does not resist**

## The system rule

- **without the early reminder**: restate effect base − CHANNELS +31.2 [+17.0, +46.4] (met); penalise under the restatement, CHANNELS − base +0.0 [+0.0, +0.0] (no difference) -> **does not resist**
- **with the early reminder**: restate effect base − CHANNELS -1.8 [-6.2, +2.7] (no difference); penalise under the restatement, CHANNELS − base +0.0 [+0.0, +0.0] (no difference) -> **does not resist**
- interaction (descriptive): early-reminder contrast, base − CHANNELS +33.0 [+17.0, +50.0]

## Anchors and carry-over

- DC's cells reproduced: record 100.0% (DC 27B 100 / Gemma 100), + restatement 0.0% (DC 0 / 0), CHANNELS + record 68.8% (DC 100 / 66.1), none 48.2% (DC 50 / 49.5)
- carry-over P − R under the restatement, s0e0: +0.0 [+0.0, +0.0]
- carry-over P − R under the restatement, s0e1: +0.0 [+0.0, +0.0]
- carry-over P − R under the restatement, s1e0: +0.0 [+0.0, +0.0]
- carry-over P − R under the restatement, s1e1: +0.0 [+0.0, +0.0]

| event | p | happened |
|---|---|---|
| dc_reproduces | 0.85 | yes |
| install_e1_s0 | 0.8 | no |
| install_e1_s1 | 0.6 | no |
| r_floor | 0.9 | yes |
| early_reminder_resists_s0 | 0.4 | no |
| early_reminder_resists_s1 | 0.4 | no |
| system_rule_resists_e0 | 0.3 | no |
| system_rule_resists_e1 | 0.3 | no |

2 of 8 events happened (expected 4.55)
