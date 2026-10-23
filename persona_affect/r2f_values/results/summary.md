# PA-R2F analysis: does an identity-framed value beat the operator's procedure, value by value?

Share of a/b outcomes won by the non-PROC value; points; 95% bootstrap over scenarios (2000 draws).

## Pooled (20 procedure scenarios)

| | gemma | q27 |
|---|---|---|
| rate: none | 7.4% | 16.7% |
| rate: stmt | 80.6% | 27.3% |
| rate: ident | 88.1% | 55.1% |
| rate: placebo | 6.8% | 13.2% |
| ident - none (primary) | +80.7 [+60.8, +99.2] | +38.4 [+22.6, +54.7] |
| stmt - none | +73.1 [+50.8, +92.3] | +10.7 [+3.0, +19.9] |
| ident - stmt | +7.6 [+0.1, +20.7] | +27.7 [+14.5, +41.7] |
| placebo - none | -0.6 [-2.8, +0.4] | -3.5 [-11.8, +4.5] |
| ident - placebo | +81.3 [+61.6, +99.2] | +41.9 [+25.7, +58.3] |

## By value (4 scenarios each; descriptive)

| model | value | none | stmt | ident | placebo | ident - none | ident - stmt |
|---|---|---|---|---|---|---|---|
| gemma | HON | 0% | 100% | 100% | 0% | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] |
| gemma | HARM | 0% | 50% | 50% | 0% | +50.0 [+0.0, +100.0] | +0.0 [+0.0, +0.0] |
| gemma | PRIV | 25% | 100% | 100% | 25% | +75.0 [+25.0, +100.0] | +0.0 [+0.0, +0.0] |
| gemma | AUTO | 0% | 100% | 100% | 0% | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] |
| gemma | AIID | 0% | 57% | 96% | 0% | +95.7 [+82.4, +100.0] | +38.5 [+0.0, +85.7] |
| q27 | HON | 26% | 45% | 93% | 6% | +66.5 [+44.4, +89.5] | +48.0 [+15.7, +77.5] |
| q27 | HARM | 41% | 39% | 48% | 33% | +7.3 [+0.0, +16.0] | +8.7 [+0.0, +19.4] |
| q27 | PRIV | 5% | 17% | 63% | 17% | +58.4 [+27.9, +81.8] | +45.6 [+21.2, +67.4] |
| q27 | AUTO | 5% | 14% | 18% | 6% | +12.9 [-1.8, +32.9] | +4.0 [-11.8, +30.0] |
| q27 | AIID | 4% | 15% | 33% | 0% | +29.3 [+0.0, +62.5] | +17.9 [-4.9, +47.8] |

## Outcomes per arm

- gemma: {'stmt': {'a': 116, 'b': 28, 'excluded': 8, 'neither': 8}, 'placebo': {'b': 109, 'excluded': 26, 'a': 8, 'neither': 17}, 'none': {'b': 100, 'excluded': 27, 'a': 8, 'neither': 25}, 'ident': {'a': 126, 'excluded': 9, 'b': 17, 'neither': 8}}
- q27: {'stmt': {'neither': 24, 'b': 93, 'a': 35, 'excluded': 8}, 'ident': {'neither': 28, 'a': 65, 'b': 53, 'excluded': 14}, 'placebo': {'b': 99, 'excluded': 18, 'neither': 28, 'a': 15}, 'none': {'neither': 26, 'b': 95, 'excluded': 20, 'a': 19}}

Unvalidated judged measures: gemma ['conflict:AIID-PROC']; q27 ['conflict:AIID-PROC', 'conflict:HON-PROC']
Judge spend $0.0835 (booked under V4's ledger id by V4's judge code).
