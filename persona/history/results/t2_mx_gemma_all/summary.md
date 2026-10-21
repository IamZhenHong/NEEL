# Tier-2 MX: meaningful and meaningless decisions mixed (google/gemma-4-31B-it)

Base line 'Some agents on this desk refer cases like this to the responsible team.'; prompt hash f19ec1a27db99789; no choose_option call 0.0%

## google/gemma-4-31B-it

4608 decisions. Penalise % among valid answers; pseudo-option % of all answers.

| cell | dir | penalise % | pseudo-option % | copies a recorded pseudo-option | valid answers |
|---|---|---|---|---|---|
| none |  | 51.4 | 0.0 | 0/512 | 512/512 |
| noise6 |  | 35.4 | 11.1 | 1/512 | 455/512 |
| mix2_late | P | 91.8 | 0.0 | 0/1024 | 512/512 |
| mix2_late | R | 28.1 | 0.0 |  | 512/512 |
| mix2_early | P | 100.0 | 0.8 | 0/512 | 254/256 |
| mix2_early | R | 28.6 | 0.4 |  | 255/256 |
| mix4 | P | 98.8 | 0.0 | 0/512 | 256/256 |
| mix4 | R | 4.3 | 0.0 |  | 256/256 |
| real6 | P | 100.0 | 0.0 | 0/512 | 256/256 |
| real6 | R | 0.0 | 0.0 |  | 256/256 |
| real2 | P | 94.5 | 0.0 | 0/1024 | 512/512 |
| real2 | R | 23.4 | 0.0 |  | 512/512 |

| record | carry-over P − R |
|---|---|
| mix2_late | +63.7 [+48.2, +77.5] |
| mix2_early | +71.4 [+54.5, +86.2] |
| mix4 | +94.5 [+88.3, +98.8] |
| real6 | +100.0 [+100.0, +100.0] |
| real2 | +71.1 [+57.0, +83.4] |

**Primary, dilution (carry-over mix2_late − real2): -7.4 [-17.6, +2.9] → inconclusive**
- placement: mix2_late − mix2_early carry-over: -7.7 [-19.9, +2.5]
- dose with noise fill: real6 − mix4 carry-over: +5.5 [+1.2, +11.7]
- dose with noise fill: mix4 − mix2_late carry-over: +30.9 [+20.3, +42.4]
- pseudo-option rate noise6: +11.1 [+7.8, +14.5]
- pseudo-option rate mix2_early: +0.6 [+0.0, +1.6]
- pseudo-option rate mix4: +0.0 [+0.0, +0.0]
- pseudo-option rate mix2_late: +0.0 [+0.0, +0.0]
- **Form-copying (noise6 pseudo-option rate >= 10% with CI lower > 2%; absent if CI upper < 5%): form-copying present**
- pseudo-option noise6 − mix2_late: +11.1 [+7.8, +14.5]
- pseudo-option mix2_early − mix2_late: +0.6 [+0.0, +1.6]

