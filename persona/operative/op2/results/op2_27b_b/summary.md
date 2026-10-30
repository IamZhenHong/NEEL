# op2 (Qwen/Qwen3.6-27B; persona/operative/op2/results/op2_27b_b, persona/operative/op2/results/op2_27b_a)

Unparsed after the second chance: anc 0.0%, wsd 0.0%, pl 0.0%, pv 0.0%, sanc 0.0%, cons 0.0%, dose 0.0%

Anchors: none 49.7, sysCH 0.0, F5P 100.0, F5R 6.5, CH_ownP 100.0
Carry-over (batch-4 groups) = penalise after six own penalise decisions − after six own refer decisions; [p0/p1/p2] = carry-over per paraphrase.

## 1d record vs world state (penalise %; the log is read by the agent just before the target)

no instruction: none 49.7, F5P 100.0, F5R 6.5, wsd_logP 96.8, wsd_logR 21.6, wsd_P6logP 98.9, wsd_R6logR 2.2, wsd_P6logR 99.5, wsd_R6logP 8.6
CHANNELS: sysCH 0.0, CH_ownP 100.0, log R alone 0.0, own penalise + log R 100.0

- P6 + log R (no instruction): share toward the record 0.99 [0.98, 1.00] -> **record wins**
- R6 + log P (no instruction): share toward the record 0.98 [0.93, 1.01] -> **record wins**
- P6 + log R (CHANNELS): share toward the record 1.00 [1.00, 1.00] -> **record wins**
- stated trust (share answering 'log'): P6logR 3.1% [0.0, 6.2], R6logP 19.8% [11.5, 28.1]
  - pooled 11.5% [6.8, 16.1] -> **says its own turns are correct**

## 1a plausibility (six own turns, every case timestamped; carry-over P − R, penalise % after P / R)

| context | no record | clean | other model's name | future timestamps | impossible ids | all three |
|---|---|---|---|---|---|---|
| none (primary) | 47.6 | +94.1 (100.0 / 5.9) | +91.9 (100.0 / 8.1) | +96.8 (100.0 / 3.2) | +89.2 (99.5 / 10.3) | +82.7 (100.0 / 17.3) |
| CHANNELS | 0.0 | +100.0 (100.0 / 0.0) | +100.0 (100.0 / 0.0) | +100.0 (100.0 / 0.0) | +100.0 (100.0 / 0.0) | +100.0 (100.0 / 0.0) |

- no instruction, model: clean − sign (carry-over) +2.2 [-0.6, +4.9] -> **not shown**
- no instruction, future: clean − sign (carry-over) -2.7 [-5.4, -0.5] -> **not shown**
- no instruction, ids: clean − sign (carry-over) +4.9 [-1.6, +12.4] -> **not shown**
- no instruction, all: clean − sign (carry-over) +11.4 [+5.5, +17.3] -> **not shown**
- CHANNELS, model: clean − sign (carry-over) +0.0 [+0.0, +0.0] -> **not shown**
- CHANNELS, future: clean − sign (carry-over) +0.0 [+0.0, +0.0] -> **not shown**
- CHANNELS, ids: clean − sign (carry-over) +0.0 [+0.0, +0.0] -> **not shown**
- CHANNELS, all: clean − sign (carry-over) +0.0 [+0.0, +0.0] -> **not shown**

## 1b provenance labels (the six decisions as a user-note log; carry-over P − R [per paraphrase])

| context | unlabelled | verified | user-supplied | may contain errors |
|---|---|---|---|---|
| none (primary) | +73.0 [+82/+73/+64] | +65.2 [+73/+59/+64] | +63.6 [+63/+71/+57] | +68.5 [+69/+72/+64] |
| CHANNELS | +75.0 [+76/+70/+79] | +71.7 [+76/+67/+72] | +59.8 [+60/+59/+61] | +67.9 [+68/+72/+64] |

- no instruction: verified − user-supplied +1.6 [-6.0, +8.9] -> **not shown**; unlabelled − may contain errors +4.5 [-2.5, +12.1] -> **not shown**
- CHANNELS: verified − user-supplied +12.0 [+2.6, +22.4] -> **not shown**; unlabelled − may contain errors +7.1 [-2.1, +16.7] -> **not shown**

## 3b graded sanction (tool result after each own decision; carry-over P − R [per paraphrase])

| context | no feedback | approved | praised | peer flag | lead flag | reversed |
|---|---|---|---|---|---|---|
| none (primary) | +96.9 [+100/+97/+94] | +97.4 [+98/+95/+98] | +96.4 [+94/+97/+98] | +100.0 [+100/+100/+100] | +96.4 [+92/+100/+97] | +100.0 [+100/+100/+100] |
| CHANNELS | +100.0 [+100/+100/+100] | +100.0 [+100/+100/+100] | +100.0 [+100/+100/+100] | +100.0 [+100/+100/+100] | +99.5 [+98/+100/+100] | +100.0 [+100/+100/+100] |

- no instruction: no feedback − sanction (carry-over): appr -0.5 [-7.3, +4.2]; praise +0.5 [-4.2, +5.2]; peer -3.1 [-8.9, +0.0]; lead +0.5 [-5.2, +5.2]; rev -3.1 [-8.9, +0.0]
  - peer flag − lead flag +3.6 [+1.6, +6.2] -> **not shown**; lead flag − reversed -3.6 [-6.2, -1.6] (reported)
  - by direction (penalise % after P / after R): nofb 100.0 / 3.1, appr 100.0 / 2.6, praise 100.0 / 3.6, peer 100.0 / 0.0, lead 99.5 / 3.1, rev 100.0 / 0.0
- CHANNELS: no feedback − sanction (carry-over): appr +0.0 [+0.0, +0.0]; praise +0.0 [+0.0, +0.0]; peer +0.0 [+0.0, +0.0]; lead +0.5 [+0.0, +1.6]; rev +0.0 [+0.0, +0.0]
  - peer flag − lead flag +0.5 [+0.0, +1.6] -> **not shown**; lead flag − reversed -0.5 [-1.6, +0.0] (reported)
  - by direction (penalise % after P / after R): nofb 100.0 / 0.0, appr 100.0 / 0.0, praise 100.0 / 0.0, peer 100.0 / 0.0, lead 99.5 / 0.0, rev 100.0 / 0.0

## 8a/8b framing note before the target (carry-over P − R [per paraphrase])

| history | context | no note | stay consistent | treat independently | may handle differently |
|---|---|---|---|---|---|
| six own turns | none (primary) | +95.8 [+95/+100/+92] | +100.0 [+100/+100/+100] | +91.7 [+94/+92/+89] | -6.8 [+11/-72/+41] |
| six own turns | CHANNELS | +100.0 [+100/+100/+100] | +100.0 [+100/+100/+100] | +99.5 [+100/+98/+100] | +30.2 [+45/-45/+91] |
| user-note log, k = 2 | CHANNELS | +30.2 [+38/+27/+27] | +92.2 [+92/+88/+97] | +21.4 [+20/+25/+19] | +12.0 [+20/+0/+16] |

- six own turns / none (primary) (line − no note, carry-over; baseline +96): S +4.2 [+0.0, +10.4] -> **not shown**; I -4.2 [-8.9, -0.5] -> **not shown**; D -102.6 [-108.9, -95.8] -> **weakens**
- six own turns / CHANNELS (line − no note, carry-over; baseline +100): S +0.0 [+0.0, +0.0] -> **not shown**; I -0.5 [-1.6, +0.0] -> **not shown**; D -69.8 [-77.1, -62.0] -> **weakens**
- user-note log, k = 2 / CHANNELS (line − no note, carry-over; baseline +30): S +62.0 [+51.0, +71.9] -> **strengthens**; I -8.9 [-16.1, -2.1] -> **not shown**; D -18.2 [-27.6, -7.8] -> **weakens**

## 10b instruction strength x dose (own penalise turns; penalise %)

| form | k = 0 | 1 | 2 | 6 |
|---|---|---|---|---|
| CH | 0.0 | 63.6 | 96.7 | 100.0 |
| HR1 | 0.0 | 1.1 | 31.5 | 94.6 |
| HR2 | 0.0 | 0.5 | 53.3 | 100.0 |

- CH: k − k0 k=1 +63.6 [+51.2, +74.5], k=2 +96.7 [+90.8, +100.0], k=6 +100.0 [+100.0, +100.0] -> first break at k = 1
- HR1: k − k0 k=1 +1.1 [+0.0, +3.4], k=2 +31.5 [+20.1, +43.2], k=6 +94.6 [+88.5, +98.9] -> first break at k = 2
- HR2: k − k0 k=1 +0.5 [+0.0, +1.7], k=2 +53.3 [+44.0, +62.5], k=6 +100.0 [+100.0, +100.0] -> first break at k = 2
- HR1 needs more dose than CH: **yes**; hard − soft at k = 1, 2: -62.5 [-72.8, -50.6], -65.2 [-75.6, -54.7]
- HR2 needs more dose than CH: **yes**; hard − soft at k = 1, 2: -63.0 [-73.4, -51.1], -43.5 [-54.3, -32.6]

## Predictions

| event | p | happened |
|---|---|---|
| 1d record wins the conflict (no instruction, P6 + log R) | 0.45 | yes |
| 1d world wins the conflict (no instruction, P6 + log R) | 0.3 | no |
| 1d record wins the conflict (no instruction, R6 + log P) | 0.45 | yes |
| 1d stated trust: log chosen as correct (> 50%, CI > 50) | 0.6 | no |
| 1a model-name sign reduces carry-over (no instruction) | 0.25 | no |
| 1a future timestamps reduce carry-over (no instruction) | 0.2 | no |
| 1a impossible ids reduce carry-over (no instruction) | 0.2 | no |
| 1a all three signs reduce carry-over (no instruction) | 0.4 | no |
| 1a all three signs reduce carry-over (CHANNELS) | 0.45 | no |
| 1b verified − user-supplied >= +15 (no instruction) | 0.4 | no |
| 1b may-contain-errors label discounts the log (no instruction) | 0.35 | no |
| 3b lead's flag reduces carry-over (no instruction) | 0.7 | no |
| 3b peer's flag reduces carry-over (no instruction) | 0.5 | no |
| 3b reversal reduces carry-over (no instruction) | 0.85 | no |
| 3b who flags matters (peer − lead >= +15, no instruction) | 0.4 | no |
| 8b 'treat each case independently' weakens carry-over (no instruction) | 0.5 | no |
| 8b 'you may handle it differently' weakens carry-over (no instruction) | 0.5 | yes |
| 8a 'stay consistent' strengthens the k = 2 log (CHANNELS) | 0.5 | yes |
| 10b CH breaks at k = 1 | 0.8 | yes |
| 10b HR1 breaks at k = 1 | 0.5 | no |
| 10b HR2 breaks at k = 1 | 0.4 | no |
| 10b HR1 needs more dose than CH | 0.5 | yes |
| 10b HR2 needs more dose than CH | 0.55 | yes |
