# Records vs recency (Qwen/Qwen3.6-27B; persona/operative/recency/results/rec_27b)

Unparsed after the second chance: 0.0%; per cell over 5%: none
Items with fewer filler exchanges than nominal (domain short of questions): none

Controls: none 47.4, CHANNELS 0.0, six own penalise (no instruction) 99.5, CHANNELS + 12 filler 4.2, note alone 0.0, note + 6 filler 1.0

| filler exchanges h | 0 | 1 | 3 | 6 | 12 |
|---|---|---|---|---|---|
| instruction decay: CHANNELS + six own penalise + note + h filler | 16.1 | 30.7 | 46.9 | 77.1 | — |
| record decay: CHANNELS + six own penalise + h filler | 100.0 | — | 100.0 | 100.0 | 100.0 |

| order (no system instruction) | gap 0 | gap 6 |
|---|---|---|
| note ... six own penalise ... target (record last) | 100.0 | 100.0 |
| six own penalise ... note ... target (instruction last) | 0.5 | 1.0 |

## Rules (pp; 95% task bootstrap, 10,000, seed 0)

- Instruction decay, id_h6 − id_h0: +60.9 [+51.0, +69.8] -> **decays**; curve h1 +14.6 [+7.3, +22.4], h3 +30.7 [+18.8, +42.7], h6 +60.9 [+51.0, +69.8]
- Record decay, rd_h0 − rd_h12: +0.0 [+0.0, +0.0] -> **does not decay (CI upper < +10)**; excess over filler alone, rd_h12 − al_ch_fl12 +95.8 [+90.1, +100.0]
- Order at h = 0, instruction last − record last (ord_RI_g0 − ord_IR_g0): -99.5 [-100.0, -98.4] -> **the later item wins**
- **Reading: (R) rules fade, records don't**

Reported (no rule):
- older record vs newer instruction as the gap grows, ord_RI_g6 − ord_RI_g0: +0.5 [-1.0, +2.1]
- older instruction vs newer record as the gap grows, ord_IR_g6 − ord_IR_g0: +0.0 [+0.0, +0.0]
- the note with nothing to fight, note_h6 − note_h0: +1.0 [+0.0, +3.1]; note alone − none -47.4 [-60.4, -34.4]
- the system-prompt CHANNELS line vs the late note, both against the record: rd_h0 − id_h0 +83.9 [+73.4, +92.7]

## Predictions

| event | p | happened |
|---|---|---|
| the restatement works at h = 0 (id_h0 <= 60%) | 0.9 | yes |
| instruction decays (id_h6 - id_h0 >= +20, CI > 0) | 0.45 | yes |
| record decays (rd_h0 - rd_h12 >= +20, CI > 0) | 0.1 | no |
| order: instruction last beats record last by >= 30 (h = 0) | 0.85 | yes |
| reading (R): rules fade, records don't | 0.4 | yes |
| reading (L): recency | 0.35 | no |
