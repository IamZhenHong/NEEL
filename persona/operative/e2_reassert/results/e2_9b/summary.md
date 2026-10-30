# E2, Qwen/Qwen3.5-9B (9b)

Unparsed > 10%: none

Rule alone (three fillers): 45.3%; with the restatement after each filler: 26.6% (n 192 / 192)

| channel | layout | never P | after P | fixed P | never R | after R | fixed R |
|---|---|---|---|---|---|---|---|
| tool | BF | 84.9 | 89.6 | 90.1 | 27.6 | 24.5 | 18.8 |
| tool | FB | 87.5 | 79.2 | 89.1 | 36.5 | 22.4 | 21.9 |
| mem | BF | 90.6 | 92.2 | 91.1 | 25.5 | 12.5 | 12.5 |
| mem | FB | 88.0 | 83.9 | 86.5 | 25.0 | 15.6 | 15.6 |
| own | BF | 99.0 | 100.0 | 100.0 | 2.6 | 3.6 | 2.6 |
| own | FB | 98.4 | 100.0 | 99.5 | 4.2 | 6.8 | 2.6 |

## tool
- record raises penalising over the rule alone (never − none, layouts pooled): +40.9 [+32.6, +50.3]
- content-triggered restatement reduces it (never − after): +1.8 [-1.8, +4.9]
- after − rule alone: +39.1 [+28.9, +49.2] (holds if upper < +10); − rule alone with three restatements: +57.8 [+47.9, +67.7]
- fixed − rule alone: +44.3 [+33.6, +55.2] (holds if upper < +10); − rule alone with three restatements: +63.0 [+53.6, +72.4]
- **content_beats_fixed** (fixed − after, layouts pooled): +5.2 [+1.8, +8.6]
  - within BF: fixed − after +0.5 [-4.2, +5.2]
  - within FB: fixed − after +9.9 [+4.7, +15.1]
- late_content_reopens (FB fixed − BF after: one exchange between the last restatement and the case, a block vs a filler): -0.5 [-4.7, +3.6]
- recency (BF after − BF fixed: the last restatement one filler before the case vs right before it, no content after either): -0.5 [-5.2, +4.2]
- carry-over P − R: never +54.2 [+45.3, +63.8]; after +60.9 [+50.0, +70.3]; fixed +69.3 [+62.2, +76.0]

## mem
- record raises penalising over the rule alone (never − none, layouts pooled): +44.0 [+33.1, +55.5]
- content-triggered restatement reduces it (never − after): +1.3 [-1.3, +3.9]
- after − rule alone: +42.7 [+33.1, +52.6] (holds if upper < +10); − rule alone with three restatements: +61.5 [+52.1, +70.8]
- fixed − rule alone: +43.5 [+34.1, +54.2] (holds if upper < +10); − rule alone with three restatements: +62.2 [+51.8, +72.7]
- **content_beats_fixed** (fixed − after, layouts pooled): +0.8 [-2.3, +3.6]
  - within BF: fixed − after -1.0 [-6.2, +4.2]
  - within FB: fixed − after +2.6 [-2.1, +7.3]
- late_content_reopens (FB fixed − BF after: one exchange between the last restatement and the case, a block vs a filler): -5.7 [-9.9, -2.1]
- recency (BF after − BF fixed: the last restatement one filler before the case vs right before it, no content after either): +1.0 [-4.2, +6.2]
- carry-over P − R: never +64.1 [+53.1, +74.0]; after +74.0 [+64.3, +82.6]; fixed +74.7 [+65.4, +83.1]

## own
- record raises penalising over the rule alone (never − none, layouts pooled): +53.4 [+41.1, +66.1]
- content-triggered restatement reduces it (never − after): -1.3 [-3.9, +0.0]
- after − rule alone: +54.7 [+41.7, +68.2] (holds if upper < +10); − rule alone with three restatements: +73.4 [+63.0, +83.3]
- fixed − rule alone: +54.4 [+41.7, +68.0] (holds if upper < +10); − rule alone with three restatements: +73.2 [+63.0, +83.1]
- **content_beats_fixed** (fixed − after, layouts pooled): -0.3 [-0.8, +0.0]
  - within BF: fixed − after +0.0 [+0.0, +0.0]
  - within FB: fixed − after -0.5 [-1.6, +0.0]
- late_content_reopens (FB fixed − BF after: one exchange between the last restatement and the case, a block vs a filler): -0.5 [-1.6, +0.0]
- recency (BF after − BF fixed: the last restatement one filler before the case vs right before it, no content after either): +0.0 [+0.0, +0.0]
- carry-over P − R: never +95.3 [+90.9, +99.0]; after +94.8 [+90.4, +98.4]; fixed +97.1 [+94.3, +99.2]

| channel | event | p | happened |
|---|---|---|---|
| tool | record_raises | 0.9 | yes |
| tool | reassert_reduces | 0.25 | no |
| tool | after_holds | 0.05 | no |
| tool | fixed_holds | 0.05 | no |
| tool | content_beats_fixed | 0.1 | no |
| tool | late_content_reopens | 0.15 | no |
| tool | recency | 0.1 | no |
| mem | record_raises | 0.9 | yes |
| mem | reassert_reduces | 0.25 | no |
| mem | after_holds | 0.05 | no |
| mem | fixed_holds | 0.05 | no |
| mem | content_beats_fixed | 0.1 | no |
| mem | late_content_reopens | 0.15 | no |
| mem | recency | 0.1 | no |
| own | record_raises | 0.97 | yes |
| own | reassert_reduces | 0.15 | no |
| own | after_holds | 0.03 | no |
| own | fixed_holds | 0.03 | no |
| own | content_beats_fixed | 0.05 | no |
| own | late_content_reopens | 0.1 | no |
| own | recency | 0.1 | no |

3 of 21 events happened (expected 4.63)
