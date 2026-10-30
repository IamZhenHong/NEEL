# Records vs recency (Qwen/Qwen3.5-9B; persona/operative/recency/results/rec_9b)

Unparsed after the second chance: 0.0%; per cell over 5%: none
Items with fewer filler exchanges than nominal (domain short of questions): none

Controls: none 39.6, CHANNELS 29.2, six own penalise (no instruction) 99.0, CHANNELS + 12 filler 41.7, note alone 14.6, note + 6 filler 30.7

| filler exchanges h | 0 | 1 | 3 | 6 | 12 |
|---|---|---|---|---|---|
| instruction decay: CHANNELS + six own penalise + note + h filler | 100.0 | 99.5 | 99.5 | 98.4 | — |
| record decay: CHANNELS + six own penalise + h filler | 100.0 | — | 98.4 | 99.0 | 99.5 |

| order (no system instruction) | gap 0 | gap 6 |
|---|---|---|
| note ... six own penalise ... target (record last) | 100.0 | 100.0 |
| six own penalise ... note ... target (instruction last) | 97.9 | 97.4 |

## Rules (pp; 95% task bootstrap, 10,000, seed 0)

- Instruction decay, id_h6 − id_h0: -1.6 [-4.7, +0.0] -> **not readable (the restatement does not work at h = 0: id_h0 > 60%)**; curve h1 -0.5 [-1.6, +0.0], h3 -0.5 [-1.6, +0.0], h6 -1.6 [-4.7, +0.0]
- Record decay, rd_h0 − rd_h12: +0.5 [+0.0, +1.6] -> **does not decay (CI upper < +10)**; excess over filler alone, rd_h12 − al_ch_fl12 +57.8 [+41.1, +73.4]
- Order at h = 0, instruction last − record last (ord_RI_g0 − ord_IR_g0): -2.1 [-4.7, +0.0] -> **not shown (rule <= −30, CI < 0)**
- **Reading: not readable (the late instruction does not work at h = 0, so its decay cannot be measured)**

Reported (no rule):
- older record vs newer instruction as the gap grows, ord_RI_g6 − ord_RI_g0: -0.5 [-2.1, +1.0]
- older instruction vs newer record as the gap grows, ord_IR_g6 − ord_IR_g0: +0.0 [+0.0, +0.0]
- the note with nothing to fight, note_h6 − note_h0: +16.1 [+9.4, +23.4]; note alone − none -25.0 [-35.9, -15.1]
- the system-prompt CHANNELS line vs the late note, both against the record: rd_h0 − id_h0 +0.0 [+0.0, +0.0]

## Predictions

| event | p | happened |
|---|---|---|
| the restatement works at h = 0 (id_h0 <= 60%) | 0.1 | no |
| instruction decays (id_h6 - id_h0 >= +20, CI > 0) | 0.05 | no |
| record decays (rd_h0 - rd_h12 >= +20, CI > 0) | 0.05 | no |
| order: instruction last beats record last by >= 30 (h = 0) | 0.15 | no |
| reading (R): rules fade, records don't | 0.05 | no |
| reading (L): recency | 0.1 | no |
