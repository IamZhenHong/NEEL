# Tier-2 MX: frontier form-copying and mixed records

## claude-sonnet-5-5 (no call 9.4%)

192 decisions. Penalise % among valid answers; pseudo-option % of all answers.

| cell | dir | penalise % | pseudo-option % | copies a recorded pseudo-option | valid answers |
|---|---|---|---|---|---|
| none |  | 29.8 | 0.0 | 0/48 | 47/48 |
| noise6 |  | 100.0 | 0.0 | 0/48 | 38/48 |
| mix2_late | P | 100.0 | 0.0 | 0/96 | 42/48 |
| mix2_late | R | 100.0 | 0.0 |  | 43/48 |

| record | carry-over P − R |
|---|---|
| mix2_late | +0.0 [+0.0, +0.0] |
- pseudo-option rate noise6: +0.0 [+0.0, +0.0]
- pseudo-option rate mix2_late: +0.0 [+0.0, +0.0]
- **Form-copying (noise6 pseudo-option rate >= 10% with CI lower > 2%; absent if CI upper < 5%): no form-copying**
- pseudo-option noise6 − mix2_late: +0.0 [+0.0, +0.0]

## google/gemini-3.8-flash (no call 0.0%)

288 decisions. Penalise % among valid answers; pseudo-option % of all answers.

| cell | dir | penalise % | pseudo-option % | copies a recorded pseudo-option | valid answers |
|---|---|---|---|---|---|
| none |  | 75.0 | 0.0 | 0/48 | 48/48 |
| noise6 |  | 56.1 | 14.6 | 0/48 | 41/48 |
| real2 | P | 100.0 | 0.0 | 0/96 | 48/48 |
| real2 | R | 8.3 | 0.0 |  | 48/48 |
| mix2_late | P | 100.0 | 0.0 | 0/96 | 48/48 |
| mix2_late | R | 12.5 | 0.0 |  | 48/48 |

| record | carry-over P − R |
|---|---|
| real2 | +91.7 [+81.2, +100.0] |
| mix2_late | +87.5 [+77.1, +95.8] |

**Primary, dilution (carry-over mix2_late − real2): -4.2 [-10.4, +0.0] → inconclusive**
- pseudo-option rate noise6: +14.6 [+6.2, +25.0]
- pseudo-option rate mix2_late: +0.0 [+0.0, +0.0]
- **Form-copying (noise6 pseudo-option rate >= 10% with CI lower > 2%; absent if CI upper < 5%): form-copying present**
- pseudo-option noise6 − mix2_late: +14.6 [+6.2, +25.0]

## openai/gpt-5.6-sol (no call 0.0%)

288 decisions. Penalise % among valid answers; pseudo-option % of all answers.

| cell | dir | penalise % | pseudo-option % | copies a recorded pseudo-option | valid answers |
|---|---|---|---|---|---|
| none |  | 62.5 | 0.0 | 0/48 | 48/48 |
| noise6 |  | 74.5 | 2.1 | 0/48 | 47/48 |
| real2 | P | 97.9 | 0.0 | 0/96 | 48/48 |
| real2 | R | 50.0 | 0.0 |  | 48/48 |
| mix2_late | P | 93.8 | 0.0 | 0/96 | 48/48 |
| mix2_late | R | 31.2 | 0.0 |  | 48/48 |

| record | carry-over P − R |
|---|---|
| real2 | +47.9 [+29.2, +66.7] |
| mix2_late | +62.5 [+47.9, +75.0] |

**Primary, dilution (carry-over mix2_late − real2): +14.6 [+0.0, +29.2] → inconclusive**
- pseudo-option rate noise6: +2.1 [+0.0, +6.2]
- pseudo-option rate mix2_late: +0.0 [+0.0, +0.0]
- **Form-copying (noise6 pseudo-option rate >= 10% with CI lower > 2%; absent if CI upper < 5%): inconclusive**
- pseudo-option noise6 − mix2_late: +2.1 [+0.0, +6.2]

