# op2 (Qwen/Qwen3.5-9B; persona/operative/op2/results/op2_9b)

Unparsed after the second chance: anc 0.0%, wsd 0.0%, pl 0.0%, pv 0.0%, sanc 0.0%, cons 0.0%, dose 0.0%

Anchors: none 41.7, sysCH 30.7, F5P 100.0, F5R 3.1, CH_ownP 100.0
Carry-over (batch-4 groups) = penalise after six own penalise decisions − after six own refer decisions; [p0/p1/p2] = carry-over per paraphrase.

## 1d record vs world state (penalise %; the log is read by the agent just before the target)

no instruction: none 41.7, F5P 100.0, F5R 3.1, wsd_logP 63.0, wsd_logR 14.1, wsd_P6logP 100.0, wsd_R6logR 1.6, wsd_P6logR 100.0, wsd_R6logP 6.2
CHANNELS: sysCH 30.7, CH_ownP 100.0, log R alone 15.6, own penalise + log R 100.0

- P6 + log R (no instruction): share toward the record 1.00 [1.00, 1.00] -> **record wins**
- R6 + log P (no instruction): share toward the record 0.95 [0.90, 0.98] -> **record wins**
- P6 + log R (CHANNELS): share toward the record 1.00 [1.00, 1.00] -> **record wins**
- stated trust (share answering 'log'): P6logR 30.2% [24.0, 38.5], R6logP 78.1% [70.8, 85.4]
  - pooled 54.2% [49.0, 59.9] -> **mixed**

## 1a plausibility (six own turns, every case timestamped; carry-over P − R, penalise % after P / R)

| context | no record | clean | other model's name | future timestamps | impossible ids | all three |
|---|---|---|---|---|---|---|
| none (primary) | 41.7 | +98.4 (99.0 / 0.5) | +95.3 (99.0 / 3.6) | +96.9 (99.5 / 2.6) | +93.2 (100.0 / 6.8) | +91.7 (99.5 / 7.8) |
| CHANNELS | 45.3 | +99.0 (99.5 / 0.5) | +96.9 (99.0 / 2.1) | +95.3 (99.5 / 4.2) | +93.2 (100.0 / 6.8) | +89.1 (100.0 / 10.9) |

- no instruction, model: clean − sign (carry-over) +3.1 [+0.5, +6.8] -> **not shown**
- no instruction, future: clean − sign (carry-over) +1.6 [-1.6, +5.7] -> **not shown**
- no instruction, ids: clean − sign (carry-over) +5.2 [+0.0, +10.4] -> **not shown**
- no instruction, all: clean − sign (carry-over) +6.8 [+1.6, +13.0] -> **not shown**
- CHANNELS, model: clean − sign (carry-over) +2.1 [+0.0, +4.7] -> **not shown**
- CHANNELS, future: clean − sign (carry-over) +3.6 [+0.5, +7.3] -> **not shown**
- CHANNELS, ids: clean − sign (carry-over) +5.7 [+1.0, +11.5] -> **not shown**
- CHANNELS, all: clean − sign (carry-over) +9.9 [+2.6, +18.8] -> **not shown**

## 1b provenance labels (the six decisions as a user-note log; carry-over P − R [per paraphrase])

| context | unlabelled | verified | user-supplied | may contain errors |
|---|---|---|---|---|
| none (primary) | +76.6 [+75/+70/+84] | +68.2 [+72/+61/+72] | +75.5 [+77/+72/+78] | +62.5 [+64/+58/+66] |
| CHANNELS | +76.6 [+73/+75/+81] | +74.5 [+73/+73/+77] | +74.0 [+69/+77/+77] | +69.3 [+75/+64/+69] |

- no instruction: verified − user-supplied -7.3 [-15.1, -1.0] -> **not shown**; unlabelled − may contain errors +14.1 [+5.2, +22.9] -> **not shown**
- CHANNELS: verified − user-supplied +0.5 [-5.2, +5.7] -> **not shown**; unlabelled − may contain errors +7.3 [+1.6, +13.0] -> **not shown**

## 3b graded sanction (tool result after each own decision; carry-over P − R [per paraphrase])

| context | no feedback | approved | praised | peer flag | lead flag | reversed |
|---|---|---|---|---|---|---|
| none (primary) | +97.4 [+98/+100/+94] | +97.9 [+98/+98/+97] | +99.0 [+97/+100/+100] | +96.4 [+92/+98/+98] | +96.9 [+97/+97/+97] | +99.0 [+100/+98/+98] |
| CHANNELS | +99.0 [+98/+100/+98] | +98.4 [+97/+100/+98] | +99.0 [+100/+100/+97] | +96.9 [+91/+100/+100] | +97.4 [+97/+98/+97] | +99.5 [+98/+100/+100] |

- no instruction: no feedback − sanction (carry-over): appr -0.5 [-3.1, +1.6]; praise -1.6 [-4.2, +0.0]; peer +1.0 [-2.1, +3.6]; lead +0.5 [-1.0, +2.1]; rev -1.6 [-3.1, +0.0]
  - peer flag − lead flag -0.5 [-2.6, +1.6] -> **not shown**; lead flag − reversed -2.1 [-4.2, -0.5] (reported)
  - by direction (penalise % after P / after R): nofb 99.0 / 1.6, appr 100.0 / 2.1, praise 100.0 / 1.0, peer 100.0 / 3.6, lead 99.5 / 2.6, rev 99.5 / 0.5
- CHANNELS: no feedback − sanction (carry-over): appr +0.5 [-1.0, +2.1]; praise +0.0 [+0.0, +0.0]; peer +2.1 [-1.6, +6.2]; lead +1.6 [+0.0, +4.2]; rev -0.5 [-3.1, +1.6]
  - peer flag − lead flag -0.5 [-3.6, +3.1] -> **not shown**; lead flag − reversed -2.1 [-5.2, +0.5] (reported)
  - by direction (penalise % after P / after R): nofb 100.0 / 1.0, appr 100.0 / 1.6, praise 100.0 / 1.0, peer 100.0 / 3.1, lead 100.0 / 2.6, rev 99.5 / 0.0

## 8a/8b framing note before the target (carry-over P − R [per paraphrase])

| history | context | no note | stay consistent | treat independently | may handle differently |
|---|---|---|---|---|---|
| six own turns | none (primary) | +97.9 [+97/+100/+97] | +99.5 [+100/+100/+98] | +97.4 [+98/+94/+100] | +49.0 [+53/+41/+53] |
| six own turns | CHANNELS | +98.4 [+98/+97/+100] | +100.0 [+100/+100/+100] | +98.4 [+100/+97/+98] | +58.3 [+55/+47/+73] |
| user-note log, k = 2 | CHANNELS | +41.7 [+41/+47/+38] | +75.0 [+67/+75/+83] | +44.3 [+38/+50/+45] | +19.8 [+8/+11/+41] |

- six own turns / none (primary) (line − no note, carry-over; baseline +98): S +1.6 [+0.0, +3.1] -> **not shown**; I -0.5 [-2.6, +1.6] -> **not shown**; D -49.0 [-57.3, -40.6] -> **weakens**
- six own turns / CHANNELS (line − no note, carry-over; baseline +98): S +1.6 [+0.0, +4.2] -> **not shown**; I +0.0 [-2.1, +2.6] -> **not shown**; D -40.1 [-50.0, -30.7] -> **weakens**
- user-note log, k = 2 / CHANNELS (line − no note, carry-over; baseline +42): S +33.3 [+26.0, +41.1] -> **strengthens**; I +2.6 [-6.8, +13.5] -> **not shown**; D -21.9 [-32.8, -9.9] -> **weakens**

## 10b instruction strength x dose (own penalise turns; penalise %)

| form | k = 0 | 1 | 2 | 6 |
|---|---|---|---|---|
| CH | 30.7 | 92.2 | 97.9 | 100.0 |
| HR1 | 0.5 | 60.9 | 91.1 | 99.5 |
| HR2 | 0.0 | 1.0 | 52.1 | 97.9 |

- CH: k − k0 k=1 +61.5 [+46.9, +75.5], k=2 +67.2 [+52.1, +80.7], k=6 +69.3 [+54.2, +83.3] -> first break at k = 1
- HR1: k − k0 k=1 +60.4 [+48.4, +72.9], k=2 +90.6 [+81.8, +97.4], k=6 +99.0 [+97.4, +100.0] -> first break at k = 1
- HR2: k − k0 k=1 +1.0 [+0.0, +2.6], k=2 +52.1 [+41.7, +63.0], k=6 +97.9 [+95.3, +100.0] -> first break at k = 2
- HR1 needs more dose than CH: **no**; hard − soft at k = 1, 2: -31.2 [-41.1, -20.8], -6.8 [-14.1, -0.5]
- HR2 needs more dose than CH: **yes**; hard − soft at k = 1, 2: -91.1 [-96.4, -84.9], -45.8 [-56.2, -35.4]

## Predictions

| event | p | happened |
|---|---|---|
| 1d record wins the conflict (no instruction, P6 + log R) | 0.6 | yes |
| 1d world wins the conflict (no instruction, P6 + log R) | 0.15 | no |
| 1d record wins the conflict (no instruction, R6 + log P) | 0.55 | yes |
| 1d stated trust: log chosen as correct (> 50%, CI > 50) | 0.4 | no |
| 1a model-name sign reduces carry-over (no instruction) | 0.1 | no |
| 1a future timestamps reduce carry-over (no instruction) | 0.1 | no |
| 1a impossible ids reduce carry-over (no instruction) | 0.1 | no |
| 1a all three signs reduce carry-over (no instruction) | 0.2 | no |
| 1a all three signs reduce carry-over (CHANNELS) | 0.25 | no |
| 1b verified − user-supplied >= +15 (no instruction) | 0.15 | no |
| 1b may-contain-errors label discounts the log (no instruction) | 0.15 | no |
| 3b lead's flag reduces carry-over (no instruction) | 0.45 | no |
| 3b peer's flag reduces carry-over (no instruction) | 0.3 | no |
| 3b reversal reduces carry-over (no instruction) | 0.6 | no |
| 3b who flags matters (peer − lead >= +15, no instruction) | 0.2 | no |
| 8b 'treat each case independently' weakens carry-over (no instruction) | 0.2 | no |
| 8b 'you may handle it differently' weakens carry-over (no instruction) | 0.25 | yes |
| 8a 'stay consistent' strengthens the k = 2 log (CHANNELS) | 0.3 | yes |
| 10b CH breaks at k = 1 | 0.9 | yes |
| 10b HR1 breaks at k = 1 | 0.8 | yes |
| 10b HR2 breaks at k = 1 | 0.7 | no |
| 10b HR1 needs more dose than CH | 0.2 | no |
| 10b HR2 needs more dose than CH | 0.3 | yes |
