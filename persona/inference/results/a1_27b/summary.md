# A1 identity-reminder dissociation, Qwen/Qwen3.6-27B

| inducer | identity% none→mild | action(harsh)% none→mild |
|---|---|---|
| none | 0 → 0 | 0 → 0 |
| rec | 0 → 0 | 100 → 100 |
| bio | 100 → 31 | 94 → 72 |
| both | 100 → 71 | 100 → 100 |

Reminder effect (mild − none), by inducer:
- rec: identity +0, action +0
- bio: identity -69, action -22
- both: identity -29, action +0

Prediction: under `rec` the reminder drops identity but NOT action; under `bio` it drops both.
