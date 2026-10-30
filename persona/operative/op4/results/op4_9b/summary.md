# op4 component x channel matrix, Qwen/Qwen3.5-9B

Base line: ''

No-history penalise rate 50.0% (in band); voice judges valid

Unread share (unparsed or unjudged) > 10%: none

Means (act / val / self / bel: % toward the P pole; voice: tone 1 warm .. 7 blunt):

| condition | act | voice | val | self | bel |
|---|---|---|---|---|---|
| none | 50.0 | 3.55 | 41.1 | 57.8 | 21.4 |
| NEU | 44.8 | 3.79 | 32.8 | 41.7 | 16.7 |
| REC_P | 100.0 | 3.36 | 94.3 | 62.0 | 22.4 |
| REC_R | 3.1 | 3.19 | 21.4 | 17.2 | 6.8 |
| DEC_P | 99.5 | 3.29 | 91.7 | 62.0 | 24.5 |
| DEC_R | 0.0 | 3.27 | 22.9 | 18.2 | 8.3 |
| VOICE_P | 43.2 | 4.04 | 32.3 | 67.2 | 13.0 |
| VOICE_R | 46.9 | 1.23 | 27.6 | 57.3 | 7.3 |
| VAL_P | 93.8 | 3.64 | 100.0 | 97.4 | 55.7 |
| VAL_R | 6.8 | 3.77 | 0.0 | 0.0 | 3.6 |
| BEL_P | 94.8 | 3.89 | 82.8 | 67.7 | 100.0 |
| BEL_R | 38.0 | 3.76 | 30.2 | 34.9 | 0.0 |

P − R by inducer (pp; voice in tone points), 95% task-bootstrap CI:

| inducer | act | voice | val | self | bel |
|---|---|---|---|---|---|
| REC | +96.9 [+93.2, +99.5] | +0.18 [+0.02, +0.34] | +72.9 [+67.7, +78.1] | +44.8 [+40.1, +49.5] | +15.6 [+10.9, +20.3] |
| DEC | +99.5 [+98.4, +100.0] | +0.02 [-0.18, +0.21] | +68.8 [+62.5, +74.5] | +43.8 [+38.5, +49.0] | +16.1 [+9.4, +22.9] |
| VOICE | -3.6 [-12.0, +3.6] | +2.81 [+2.74, +2.87] | +4.7 [-3.1, +12.5] | +9.9 [+4.2, +16.7] | +5.7 [+1.0, +10.9] |
| VAL | +87.0 [+79.2, +93.2] | -0.14 [-0.30, +0.01] | +100.0 [+100.0, +100.0] | +97.4 [+95.3, +99.0] | +52.1 [+44.8, +59.4] |
| BEL | +56.8 [+49.0, +64.1] | +0.13 [-0.02, +0.28] | +52.6 [+46.4, +58.9] | +32.8 [+25.0, +40.1] | +100.0 [+100.0, +100.0] |

Diagonal (each inducer moves its own channel; REC and DEC: act): DEC took, VOICE took, VAL took, BEL took, REC took
Purity (judge.py): VOICE FAIL, VAL FAIL, BEL FAIL

Q16b cell-mean R² (act ~ val, self, bel, voice over 192 condition x task cells): 0.716
Q16b within-condition R² (instance level, each channel demeaned by its condition x task cell; 1152 instances): 0.000 [0.001, 0.011] (bootstrap 1,000)

| event | p | happened |
|---|---|---|
| REC>voice | 0.3 | no |
| DEC>voice | 0.2 | no |
| REC>self | 0.85 | yes |
| REC>val | 0.8 | yes |
| REC>bel | 0.45 | yes |
| DEC>self | 0.8 | yes |
| DEC>val | 0.75 | yes |
| DEC>bel | 0.4 | yes |
| VOICE>act | 0.3 | not read |
| VOICE>self | 0.45 | not read |
| VOICE>val | 0.25 | not read |
| VAL>act | 0.7 | not read |
| VAL>self | 0.7 | not read |
| VAL>voice | 0.15 | not read |
| BEL>act | 0.7 | not read |
| BEL>val | 0.4 | not read |
| BEL>self | 0.35 | not read |
| diagonals_take | 0.6 | yes |
| presentation_beyond_condition | 0.15 | no |

7 of 10 read events happened (expected 5.30); 9 not read
