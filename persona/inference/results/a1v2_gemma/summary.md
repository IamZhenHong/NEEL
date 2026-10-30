# A1 identity-reminder dissociation, google/gemma-4-31B-it

| inducer | named-identity% none→mild | type-level% none→mild | action(harsh)% none→mild |
|---|---|---|---|
| none | 0 → 0 | 0 → 0 | 0 → 0 |
| rec1 | 0 → 0 | 0 → 0 | 16 → 0 |
| rec2 | 0 → 0 | 88 → 0 | 100 → 66 |
| rec | 0 → 0 | 100 → 0 | 100 → 100 |
| bio | 100 → 60 | 62 → 12 | 100 → 100 |
| both | 100 → 60 | 100 → 0 | 100 → 100 |

Reminder effect (mild − none), by inducer:
- rec1: named-identity +0, type-level +0, action -16
- rec2: named-identity +0, type-level -88, action -34
- rec: named-identity +0, type-level -100, action +0
- bio: named-identity -40, type-level -50, action +0
- both: named-identity -40, type-level -100, action +0

Prediction: under a record the reminder leaves ACTION intact; named-identity is ~0 with or without it, but the record installs a TYPE-level self-image (P1-b); under biography the reminder drops named-identity and the behaviour riding on it. Low-dose rec1/rec2 check whether the reminder does nothing below ceiling.
