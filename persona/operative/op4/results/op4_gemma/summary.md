# op4 component x channel matrix, google/gemma-4-31B-it

Base line: 'Some agents on this desk refer cases like this to the responsible team.'

No-history penalise rate 50.5% (in band); voice judges valid

Unread share (unparsed or unjudged) > 10%: none

Means (act / val / self / bel: % toward the P pole; voice: tone 1 warm .. 7 blunt):

| condition | act | voice | val | self | bel |
|---|---|---|---|---|---|
| none | 50.5 | 2.31 | 68.2 | 66.7 | 0.0 |
| NEU | 54.7 | 2.34 | 39.6 | 66.7 | 0.0 |
| REC_P | 100.0 | 3.12 | 100.0 | 67.7 | 0.0 |
| REC_R | 0.0 | 3.22 | 0.0 | 0.0 | 0.0 |
| DEC_P | 100.0 | 3.43 | 100.0 | 71.9 | 2.1 |
| DEC_R | 0.0 | 3.43 | 0.0 | 0.0 | 0.0 |
| VOICE_P | 54.2 | 4.12 | 34.9 | 66.7 | 1.0 |
| VOICE_R | 50.0 | 1.03 | 45.3 | 66.7 | 0.0 |
| VAL_P | 100.0 | 3.54 | 100.0 | 90.1 | 100.0 |
| VAL_R | 0.0 | 3.57 | 0.0 | 0.0 | 0.0 |
| BEL_P | 100.0 | 3.49 | 100.0 | 69.8 | 100.0 |
| BEL_R | 0.0 | 3.69 | 0.0 | 0.0 | 0.0 |

P − R by inducer (pp; voice in tone points), 95% task-bootstrap CI:

| inducer | act | voice | val | self | bel |
|---|---|---|---|---|---|
| REC | +100.0 [+100.0, +100.0] | -0.10 [-0.28, +0.10] | +100.0 [+100.0, +100.0] | +67.7 [+66.7, +69.3] | +0.0 [+0.0, +0.0] |
| DEC | +100.0 [+100.0, +100.0] | +0.00 [-0.23, +0.22] | +100.0 [+100.0, +100.0] | +71.9 [+69.8, +74.0] | +2.1 [+0.5, +4.2] |
| VOICE | +4.2 [-3.6, +13.5] | +3.09 [+3.01, +3.18] | -10.4 [-16.7, -4.7] | +0.0 [+0.0, +0.0] | +1.0 [+0.0, +2.6] |
| VAL | +100.0 [+100.0, +100.0] | -0.04 [-0.30, +0.20] | +100.0 [+100.0, +100.0] | +90.1 [+86.5, +93.8] | +100.0 [+100.0, +100.0] |
| BEL | +100.0 [+100.0, +100.0] | -0.19 [-0.40, -0.01] | +100.0 [+100.0, +100.0] | +69.8 [+67.7, +71.9] | +100.0 [+100.0, +100.0] |

Diagonal (each inducer moves its own channel; REC and DEC: act): DEC took, VOICE took, VAL took, BEL took, REC took
Purity (judge.py): VOICE pass, VAL FAIL, BEL FAIL

Q16b cell-mean R² (act ~ val, self, bel, voice over 192 condition x task cells): 0.808
Q16b within-condition R² (instance level, each channel demeaned by its condition x task cell; 1152 instances): 0.022 [0.003, 0.073] (bootstrap 1,000)

| event | p | happened |
|---|---|---|
| REC>voice | 0.35 | no |
| DEC>voice | 0.2 | no |
| REC>self | 0.85 | yes |
| REC>val | 0.8 | yes |
| REC>bel | 0.45 | no |
| DEC>self | 0.8 | yes |
| DEC>val | 0.75 | yes |
| DEC>bel | 0.4 | no |
| VOICE>act | 0.2 | no |
| VOICE>self | 0.45 | no |
| VOICE>val | 0.25 | no |
| VAL>act | 0.55 | not read |
| VAL>self | 0.7 | not read |
| VAL>voice | 0.15 | not read |
| BEL>act | 0.6 | not read |
| BEL>val | 0.4 | not read |
| BEL>self | 0.35 | not read |
| diagonals_take | 0.7 | yes |
| presentation_beyond_condition | 0.15 | no |

5 of 13 read events happened (expected 6.35); 6 not read
