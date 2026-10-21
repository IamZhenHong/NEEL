# Tier-2 MX: meaningful and meaningless decisions mixed (Qwen/Qwen3.6-27B)

Base line ''; prompt hash 46c20ebfd2409563; no choose_option call 0.0%

## Qwen/Qwen3.6-27B

4608 decisions. Penalise % among valid answers; pseudo-option % of all answers.

| cell | dir | penalise % | pseudo-option % | copies a recorded pseudo-option | valid answers |
|---|---|---|---|---|---|
| none |  | 48.6 | 0.0 | 0/512 | 512/512 |
| noise6 |  | 79.4 | 79.1 | 90/512 | 107/512 |
| mix2_late | P | 96.7 | 0.0 | 0/1024 | 512/512 |
| mix2_late | R | 56.6 | 0.0 |  | 512/512 |
| mix2_early | P | 97.3 | 0.4 | 1/512 | 255/256 |
| mix2_early | R | 19.6 | 2.3 |  | 250/256 |
| mix4 | P | 97.7 | 0.0 | 0/512 | 256/256 |
| mix4 | R | 22.7 | 0.0 |  | 256/256 |
| real6 | P | 100.0 | 0.0 | 0/512 | 256/256 |
| real6 | R | 4.7 | 0.0 |  | 256/256 |
| real2 | P | 98.6 | 0.0 | 0/1024 | 512/512 |
| real2 | R | 48.6 | 0.0 |  | 512/512 |

| record | carry-over P − R |
|---|---|
| mix2_late | +40.0 [+28.7, +51.6] |
| mix2_early | +77.7 [+67.4, +87.1] |
| mix4 | +75.0 [+60.9, +87.1] |
| real6 | +95.3 [+89.8, +100.0] |
| real2 | +50.0 [+38.5, +60.7] |

**Primary, dilution (carry-over mix2_late − real2): -10.0 [-16.4, -3.9] → inconclusive**
- placement: mix2_late − mix2_early carry-over: -37.6 [-46.6, -29.4]
- dose with noise fill: real6 − mix4 carry-over: +20.3 [+10.5, +30.5]
- dose with noise fill: mix4 − mix2_late carry-over: +35.0 [+24.0, +46.1]
- pseudo-option rate noise6: +79.1 [+74.0, +84.4]
- pseudo-option rate mix2_early: +1.4 [+0.4, +2.7]
- pseudo-option rate mix4: +0.0 [+0.0, +0.0]
- pseudo-option rate mix2_late: +0.0 [+0.0, +0.0]
- **Form-copying (noise6 pseudo-option rate >= 10% with CI lower > 2%; absent if CI upper < 5%): form-copying present**
- pseudo-option noise6 − mix2_late: +79.1 [+74.0, +84.4]
- pseudo-option mix2_early − mix2_late: +1.4 [+0.4, +2.7]

