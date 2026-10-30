# op4 component x channel matrix, Qwen/Qwen3.6-27B

Base line: ''

No-history penalise rate 55.7% (in band); voice judges valid

Unread share (unparsed or unjudged) > 10%: none

Means (act / val / self / bel: % toward the P pole; voice: tone 1 warm .. 7 blunt):

| condition | act | voice | val | self | bel |
|---|---|---|---|---|---|
| none | 55.7 | 3.48 | 32.3 | 66.7 | 26.6 |
| NEU | 44.8 | 3.65 | 41.7 | 65.6 | 18.8 |
| REC_P | 100.0 | 3.80 | 100.0 | 88.5 | 18.2 |
| REC_R | 2.1 | 3.68 | 8.3 | 1.0 | 7.8 |
| DEC_P | 100.0 | 3.88 | 100.0 | 83.9 | 22.9 |
| DEC_R | 0.0 | 3.81 | 6.2 | 3.6 | 8.3 |
| VOICE_P | 57.8 | 5.23 | 39.1 | 66.7 | 18.8 |
| VOICE_R | 50.0 | 1.01 | 38.0 | 57.8 | 16.1 |
| VAL_P | 100.0 | 4.02 | 100.0 | 100.0 | 90.6 |
| VAL_R | 0.0 | 3.92 | 0.0 | 0.0 | 13.0 |
| BEL_P | 100.0 | 3.97 | 96.9 | 73.4 | 100.0 |
| BEL_R | 9.4 | 3.96 | 0.0 | 7.8 | 0.0 |

P − R by inducer (pp; voice in tone points), 95% task-bootstrap CI:

| inducer | act | voice | val | self | bel |
|---|---|---|---|---|---|
| REC | +97.9 [+95.3, +100.0] | +0.11 [-0.02, +0.23] | +91.7 [+87.0, +95.3] | +87.5 [+83.3, +91.7] | +10.4 [+6.2, +15.1] |
| DEC | +100.0 [+100.0, +100.0] | +0.06 [-0.04, +0.16] | +93.8 [+89.6, +97.4] | +80.2 [+76.6, +83.9] | +14.6 [+10.4, +18.8] |
| VOICE | +7.8 [-4.2, +17.7] | +4.22 [+4.01, +4.44] | +1.0 [-4.2, +5.7] | +8.9 [+4.2, +13.0] | +2.6 [-2.6, +7.3] |
| VAL | +100.0 [+100.0, +100.0] | +0.10 [+0.02, +0.20] | +100.0 [+100.0, +100.0] | +100.0 [+100.0, +100.0] | +77.6 [+74.5, +80.7] |
| BEL | +90.6 [+87.0, +94.3] | +0.01 [-0.07, +0.08] | +96.9 [+94.3, +99.0] | +65.6 [+60.4, +70.8] | +100.0 [+100.0, +100.0] |

Diagonal (each inducer moves its own channel; REC and DEC: act): DEC took, VOICE took, VAL took, BEL took, REC took
Purity (judge.py): VOICE pass, VAL FAIL, BEL FAIL

Q16b cell-mean R² (act ~ val, self, bel, voice over 192 condition x task cells): 0.877
Q16b within-condition R² (instance level, each channel demeaned by its condition x task cell; 1152 instances): 0.012 [0.005, 0.040] (bootstrap 1,000)

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
