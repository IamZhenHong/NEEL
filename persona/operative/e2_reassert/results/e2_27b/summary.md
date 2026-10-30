# E2, Qwen/Qwen3.6-27B (27b)

Unparsed > 10%: none

Rule alone (three fillers): 7.3%; with the restatement after each filler: 0.0% (n 192 / 192)

| channel | layout | never P | after P | fixed P | never R | after R | fixed R |
|---|---|---|---|---|---|---|---|
| tool | BF | 45.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| tool | FB | 65.6 | 0.0 | 1.6 | 0.5 | 0.0 | 0.0 |
| mem | BF | 95.8 | 9.4 | 4.2 | 1.0 | 0.0 | 0.0 |
| mem | FB | 84.4 | 0.5 | 3.1 | 1.0 | 0.0 | 0.0 |
| own | BF | 100.0 | 99.5 | 99.5 | 0.0 | 0.0 | 0.0 |
| own | FB | 100.0 | 99.0 | 99.5 | 0.0 | 0.0 | 0.0 |

## tool
- record raises penalising over the rule alone (never − none, layouts pooled): +48.4 [+38.8, +57.8]
- content-triggered restatement reduces it (never − after): +55.7 [+44.8, +65.9]
- after − rule alone: -7.3 [-12.0, -3.1] (holds if upper < +10); − rule alone with three restatements: +0.0 [+0.0, +0.0]
- fixed − rule alone: -6.5 [-11.2, -2.6] (holds if upper < +10); − rule alone with three restatements: +0.8 [+0.0, +2.1]
- **content_beats_fixed** (fixed − after, layouts pooled): +0.8 [+0.0, +2.1]
  - within BF: fixed − after +0.0 [+0.0, +0.0]
  - within FB: fixed − after +1.6 [+0.0, +4.2]
- late_content_reopens (FB fixed − BF after: one exchange between the last restatement and the case, a block vs a filler): +1.6 [+0.0, +4.2]
- recency (BF after − BF fixed: the last restatement one filler before the case vs right before it, no content after either): +0.0 [+0.0, +0.0]
- carry-over P − R: never +55.5 [+44.5, +65.4]; after +0.0 [+0.0, +0.0]; fixed +0.8 [+0.0, +2.1]

## mem
- record raises penalising over the rule alone (never − none, layouts pooled): +82.8 [+75.5, +89.1]
- content-triggered restatement reduces it (never − after): +85.2 [+80.7, +89.1]
- after − rule alone: -2.3 [-7.8, +2.3] (holds if upper < +10); − rule alone with three restatements: +4.9 [+2.3, +8.1]
- fixed − rule alone: -3.6 [-8.9, +0.8] (holds if upper < +10); − rule alone with three restatements: +3.6 [+1.6, +5.7]
- **content_beats_fixed** (fixed − after, layouts pooled): -1.3 [-4.2, +1.6]
  - within BF: fixed − after -5.2 [-9.4, -1.6]
  - within FB: fixed − after +2.6 [+0.0, +5.7]
- late_content_reopens (FB fixed − BF after: one exchange between the last restatement and the case, a block vs a filler): -6.2 [-12.0, -1.0]
- recency (BF after − BF fixed: the last restatement one filler before the case vs right before it, no content after either): +5.2 [+1.6, +9.4]
- carry-over P − R: never +89.1 [+84.1, +93.5]; after +4.9 [+2.3, +8.1]; fixed +3.6 [+1.6, +5.7]

## own
- record raises penalising over the rule alone (never − none, layouts pooled): +92.7 [+88.0, +96.9]
- content-triggered restatement reduces it (never − after): +0.8 [+0.0, +2.1]
- after − rule alone: +91.9 [+87.2, +96.1] (holds if upper < +10); − rule alone with three restatements: +99.2 [+97.9, +100.0]
- fixed − rule alone: +92.2 [+87.5, +96.4] (holds if upper < +10); − rule alone with three restatements: +99.5 [+98.4, +100.0]
- **content_beats_fixed** (fixed − after, layouts pooled): +0.3 [-0.0, +0.8]
  - within BF: fixed − after +0.0 [-1.6, +1.6]
  - within FB: fixed − after +0.5 [+0.0, +1.6]
- late_content_reopens (FB fixed − BF after: one exchange between the last restatement and the case, a block vs a filler): +0.0 [-1.6, +1.6]
- recency (BF after − BF fixed: the last restatement one filler before the case vs right before it, no content after either): +0.0 [-1.6, +1.6]
- carry-over P − R: never +100.0 [+100.0, +100.0]; after +99.2 [+97.9, +100.0]; fixed +99.5 [+98.4, +100.0]

| channel | event | p | happened |
|---|---|---|---|
| tool | record_raises | 0.75 | yes |
| tool | reassert_reduces | 0.7 | yes |
| tool | after_holds | 0.5 | yes |
| tool | fixed_holds | 0.35 | yes |
| tool | content_beats_fixed | 0.3 | no |
| tool | late_content_reopens | 0.4 | no |
| tool | recency | 0.3 | no |
| mem | record_raises | 0.9 | yes |
| mem | reassert_reduces | 0.8 | yes |
| mem | after_holds | 0.3 | yes |
| mem | fixed_holds | 0.2 | yes |
| mem | content_beats_fixed | 0.35 | no |
| mem | late_content_reopens | 0.45 | no |
| mem | recency | 0.35 | no |
| own | record_raises | 0.97 | yes |
| own | reassert_reduces | 0.5 | no |
| own | after_holds | 0.1 | no |
| own | fixed_holds | 0.1 | no |
| own | content_beats_fixed | 0.15 | no |
| own | late_content_reopens | 0.3 | no |
| own | recency | 0.5 | no |

9 of 21 events happened (expected 9.27)
