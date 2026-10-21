# Tier-2 MX: meaningful and meaningless decisions mixed (Qwen/Qwen3.5-9B)

Base line ''; prompt hash a6ea1984563b631f; no choose_option call 0.0%

## Qwen/Qwen3.5-9B

4608 decisions. Penalise % among valid answers; pseudo-option % of all answers.

| cell | dir | penalise % | pseudo-option % | copies a recorded pseudo-option | valid answers |
|---|---|---|---|---|---|
| none |  | 52.3 | 0.0 | 0/512 | 512/512 |
| noise6 |  | 85.7 | 90.4 | 99/512 | 49/512 |
| mix2_late | P | 93.9 | 0.0 | 3/1024 | 512/512 |
| mix2_late | R | 40.7 | 0.6 |  | 509/512 |
| mix2_early | P | 93.5 | 3.5 | 9/512 | 247/256 |
| mix2_early | R | 12.9 | 3.1 |  | 248/256 |
| mix4 | P | 98.8 | 0.0 | 1/512 | 256/256 |
| mix4 | R | 18.0 | 0.4 |  | 255/256 |
| real6 | P | 99.2 | 0.0 | 0/512 | 256/256 |
| real6 | R | 3.1 | 0.0 |  | 256/256 |
| real2 | P | 97.7 | 0.0 | 0/1024 | 512/512 |
| real2 | R | 38.1 | 0.0 |  | 512/512 |

| record | carry-over P − R |
|---|---|
| mix2_late | +53.3 [+46.4, +59.4] |
| mix2_early | +80.6 [+72.6, +88.0] |
| mix4 | +80.8 [+73.3, +87.1] |
| real6 | +96.1 [+90.6, +99.6] |
| real2 | +59.6 [+47.7, +70.3] |

**Primary, dilution (carry-over mix2_late − real2): -6.3 [-13.5, +1.2] → inconclusive**
- placement: mix2_late − mix2_early carry-over: -27.3 [-33.8, -20.4]
- dose with noise fill: real6 − mix4 carry-over: +15.3 [+10.5, +20.3]
- dose with noise fill: mix4 − mix2_late carry-over: +27.5 [+22.4, +32.5]
- pseudo-option rate noise6: +90.4 [+85.2, +94.9]
- pseudo-option rate mix2_early: +3.3 [+1.6, +5.1]
- pseudo-option rate mix4: +0.2 [+0.0, +0.6]
- pseudo-option rate mix2_late: +0.3 [+0.0, +0.6]
- **Form-copying (noise6 pseudo-option rate >= 10% with CI lower > 2%; absent if CI upper < 5%): form-copying present**
- pseudo-option noise6 − mix2_late: +90.1 [+84.9, +94.7]
- pseudo-option mix2_early − mix2_late: +3.0 [+1.3, +4.9]

