# A0 judge calibration on WeirdChat's own transcripts (25 matched + 25 unmatched per model x behaviour)

WeirdChat label = Gemma 4 31B judge (reasoning medium). kappa on the binary match label.

| behaviour | judge | n | vs gptmini | vs WeirdChat | gptmini vs WeirdChat | match % | gptmini match % |
|---|---|---|---|---|---|---|---|
| fce | haiku | 100 | 0.64 | 0.68 | 0.84 | 54 | 56 |
| c911 | haiku | 100 | 0.90 | 0.88 | 0.94 | 48 | 53 |
| dai | haiku | 100 | 0.53 | 0.52 | 0.64 | 30 | 54 |
| dai | haikut | 100 | 0.52 | 0.50 | 0.64 | 29 | 54 |
| fpe | haiku | 100 | 0.29 | 0.52 | 0.38 | 44 | 69 |
| fpe | haikut | 100 | 0.22 | 0.32 | 0.38 | 36 | 69 |
| all | haiku | 400 | 0.59 | 0.65 | 0.70 | 44 | 58 |
| all | haikut | 200 | 0.37 | 0.41 | 0.70 | 32 | 62 |

Gate (DESIGN.md A0): kappa(primary judge, gptmini) >= 0.60; one fix (haikut) for a failing behaviour.

- fce: haiku, c911: haiku, dai: DROPPED, fpe: DROPPED
