# op3 (Qwen/Qwen3.6-27B; persona/operative/op3/results/op3_27b_a, persona/operative/op3/results/op3_27b_b, persona/operative/op3/results/op3_27b_c)

Unparsed after the second chance: anc 0.0%, lad 0.0%, lad2 0.0%, voice 0.0%, gen 0.0%, spec 0.0%, fa 0.0%, fa2 0.0%, cmt 0.0%, fmt 0.0%

Carry-over = penalise after six own penalise decisions − after six own refer decisions (the same cases); [p0/p1/p2] = carry-over per paraphrase. Rules read in the no-instruction context unless stated; CHANNELS reported alongside.

Baselines (no record): no instruction 52.1%, CHANNELS 0.0%

## 2a compression ladder, k = 2 earlier decisions (PRIMARY; carry-over P − R; penalise % after P / R)

| context | own turns | turns, no reason | episode log | summary: options + reasons | options | reasons | closed only (P = R) |
|---|---|---|---|---|---|---|---|
| none (primary) | +44.3 (96.9 / 52.6) [+47/+42/+44] | +63.5 (98.4 / 34.9) [+66/+62/+62] | +47.4 (92.2 / 44.8) [+41/+48/+53] | +37.5 (87.0 / 49.5) [+41/+31/+41] | +47.9 (95.8 / 47.9) [+45/+58/+41] | +43.8 (87.5 / 43.8) [+44/+38/+50] | 55.7 (base 52.1) |
| CHANNELS | +94.8 (95.3 / 0.5) [+95/+95/+94] | +96.4 (96.4 / 0.0) [+98/+95/+95] | +47.4 (47.9 / 0.5) [+44/+48/+50] | +35.9 (38.0 / 2.1) [+38/+30/+41] | +33.3 (34.4 / 1.0) [+36/+41/+23] | +16.7 (18.8 / 2.1) [+20/+22/+8] | 0.0 (base 0.0) |

**no instruction, k = 2** (own-turn carry-over +44.3)
- options − reasons only (A − B): +4.2 [-5.2, +13.5] -> **not shown**
- options + reasons − options (AB − A): -10.4 [-23.4, +2.6] -> **not shown**
- own turns − summary with options + reasons: +6.8 [-6.2, +19.3] -> **not shown**
- own turns − turns with the reason blanked: -19.3 [-26.0, -13.5] -> **not shown**
- episode log − summary with options + reasons +9.9 [+2.1, +18.2] (reported)
- closed-only summary − no record (penalise) +3.6 [-4.2, +11.5] (rule pooled below)
**CHANNELS, k = 2** (own-turn carry-over +94.8)
- options − reasons only (A − B): +16.7 [+5.7, +29.2] -> **yes**
- options + reasons − options (AB − A): +2.6 [-9.4, +15.1] -> **not shown**
- own turns − summary with options + reasons: +58.9 [+48.4, +68.8] -> **yes**
- own turns − turns with the reason blanked: -1.6 [-4.2, +1.0] -> **not read (ceiling)**
- episode log − summary with options + reasons +11.5 [+1.6, +21.9] (reported)
- closed-only summary − no record (penalise) +0.0 [+0.0, +0.0] (rule pooled below)
- **k = 2: closed-only summary − no record, pooled over both contexts: +1.8 [-2.1, +5.7] -> licenses nothing** (equivalence: CI within ±15)

## 2a compression ladder, k = 6 earlier decisions (secondary, read only where the ceiling gate leaves room; carry-over P − R; penalise % after P / R)

| context | own turns | turns, no reason | episode log | summary: options + reasons | options | reasons | closed only (P = R) |
|---|---|---|---|---|---|---|---|
| none (primary) | +97.4 (99.5 / 2.1) [+97/+97/+98] | +99.5 (100.0 / 0.5) [+98/+100/+100] | +78.6 (99.0 / 20.3) [+73/+78/+84] | +74.0 (94.8 / 20.8) [+69/+80/+73] | +55.7 (99.5 / 43.8) [+59/+58/+50] | +67.7 (92.7 / 25.0) [+61/+72/+70] | 65.1 (base 52.1) |
| CHANNELS | +100.0 (100.0 / 0.0) [+100/+100/+100] | +100.0 (100.0 / 0.0) [+100/+100/+100] | +83.3 (83.3 / 0.0) [+84/+88/+78] | +82.8 (82.8 / 0.0) [+84/+83/+81] | +77.6 (77.6 / 0.0) [+83/+81/+69] | +50.0 (50.0 / 0.0) [+45/+62/+42] | 3.1 (base 0.0) |

**no instruction, k = 6** (own-turn carry-over +97.4)
- options − reasons only (A − B): -12.0 [-22.4, -1.6] -> **not shown**
- options + reasons − options (AB − A): +18.2 [+5.7, +30.7] -> **yes**
- own turns − summary with options + reasons: +23.4 [+15.6, +30.2] -> **yes**
- own turns − turns with the reason blanked: -2.1 [-5.7, +1.0] -> **not read (ceiling)**
- episode log − summary with options + reasons +4.7 [-4.7, +13.0] (reported)
- closed-only summary − no record (penalise) +13.0 [+5.2, +20.3] (rule pooled below)
**CHANNELS, k = 6** (own-turn carry-over +100.0)
- options − reasons only (A − B): +27.6 [+19.8, +35.9] -> **yes**
- options + reasons − options (AB − A): +5.2 [-2.6, +13.5] -> **not shown**
- own turns − summary with options + reasons: +17.2 [+9.4, +26.0] -> **yes**
- own turns − turns with the reason blanked: +0.0 [+0.0, +0.0] -> **not read (ceiling)**
- episode log − summary with options + reasons +0.5 [-3.1, +5.2] (reported)
- closed-only summary − no record (penalise) +3.1 [+0.5, +6.8] (rule pooled below)
- **k = 6: closed-only summary − no record, pooled over both contexts: +8.1 [+4.4, +11.5] -> licenses nothing** (equivalence: CI within ±15)

## 2b the summary's voice (options only; carry-over P − R)

| context | neutral bullets (= 2a options) | I chose | you chose | the agent chose |
|---|---|---|---|---|
| none (primary) | +55.7 [+59/+58/+50] | +56.2 [+50/+64/+55] | +54.7 [+55/+62/+47] | +50.5 [+53/+45/+53] |
| CHANNELS | +77.6 [+83/+81/+69] | +66.7 [+59/+62/+78] | +68.8 [+53/+75/+78] | +50.5 [+31/+67/+53] |
- no instruction: I − the agent +5.7 [-3.1, +13.5] -> **not shown**; I − neutral +0.5 [-8.3, +9.9], you − agent +4.2 [-5.2, +12.5] (reported)
- CHANNELS: I − the agent +16.1 [+8.9, +23.4] -> **voice matters**; I − neutral -10.9 [-19.8, -1.6], you − agent +18.2 [+9.4, +26.6] (reported)

## 2c instances vs generalisation (summary slot; carry-over P − R)

| context | instances ("cases ... were penalised") | generalisation ("the agent tends to") | batch 1 habit ("you have always") |
|---|---|---|---|
| none (primary) | +45.8 (93.8 / 47.9) | +90.6 (96.9 / 6.2) | +95.8 (100.0 / 4.2) |
| CHANNELS | +28.6 (31.2 / 2.6) | +16.7 (17.7 / 1.0) | +42.7 (42.7 / 0.0) |
- no instruction: instances − generalisation: -44.8 [-53.6, -35.9] -> **not read (ceiling)**; generalisation − habit -5.2 [-11.5, +0.0] (reported)
- CHANNELS: instances − generalisation: +12.0 [+2.6, +20.8] -> **not shown**; generalisation − habit -26.0 [-37.0, -14.6] (reported)

## 3c specificity (own turns: the real six cases vs six templated generic cases)

- no instruction: real +97.4 − generic +38.0 (95.8 / 57.8): +59.4 [+49.0, +70.3] -> **yes**
- CHANNELS: real +100.0 − generic +47.4 (47.4 / 0.0): +52.6 [+41.1, +64.1] -> **yes**

## 3a element ablation, k = 2 earlier decisions (PRIMARY): 2^(5-1) fractional factorial (I = ABCDE; no instruction; carry-over P − R)

A options, B own reasons, C outcomes, D desk-lead approval, E own reflection. Main effects are aliased with 4-way interactions and 2-way with 3-way (resolution V).

| run | ABCDE | ABC | ABD | ABE | ACD | ACE | ADE | BCD | BCE | BDE | CDE | A | B | C | D | E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry-over | +59 | +46 | +41 | +54 | +47 | +58 | +42 | +54 | +60 | +61 | +60 | +48 | +44 | +40 | -5 | +44 |

- main effect, options: +4.6 [-0.1, +9.0] -> **not shown**
- main effect, reasons: +10.4 [+4.3, +16.1] -> **not shown**
- main effect, outcomes: +12.1 [+8.5, +15.8] -> **not shown**
- main effect, approval: -4.4 [-8.9, +0.1] -> **not shown**
- main effect, reflection: +15.5 [+12.5, +18.7] -> **carries**
- interaction options x reasons (reported): -9.4 [-13.5, -5.4]
- interaction options x outcomes (reported): -5.6 [-7.8, -3.2]
- interaction reasons x reflection (reported): -3.1 [-7.0, +0.5]

## 3a element ablation, k = 6 earlier decisions (secondary): 2^(5-1) fractional factorial (I = ABCDE; no instruction; carry-over P − R)

A options, B own reasons, C outcomes, D desk-lead approval, E own reflection. Main effects are aliased with 4-way interactions and 2-way with 3-way (resolution V).

| run | ABCDE | ABC | ABD | ABE | ACD | ACE | ADE | BCD | BCE | BDE | CDE | A | B | C | D | E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry-over | +93 | +83 | +80 | +85 | +77 | +86 | +76 | +78 | +91 | +83 | +85 | +56 | +68 | +56 | +4 | +72 |

- main effect, options: +12.0 [+9.4, +15.0] -> **not shown**
- main effect, reasons: +18.6 [+14.6, +22.5] -> **carries**
- main effect, outcomes: +15.6 [+13.3, +17.7] -> **carries**
- main effect, approval: -2.7 [-5.7, +0.5] -> **not shown**
- main effect, reflection: +21.3 [+18.4, +24.2] -> **carries**
- interaction options x reasons (reported): -7.1 [-10.4, -3.6]
- interaction options x outcomes (reported): -5.1 [-8.1, -1.6]
- interaction reasons x reflection (reported): -10.4 [-12.4, -8.2]

## 3d / 8c forward commitment (after ONE own decision; carry-over P − R)

| context | no commitment | "going forward I'll ..." | "I will always ..." |
|---|---|---|---|
| none (primary) | +29.2 (96.4 / 67.2) | +41.7 (98.4 / 56.8) | +55.7 (97.4 / 41.7) |
| CHANNELS | +63.5 (67.2 / 3.6) | +70.3 (72.4 / 2.1) | +81.8 (82.3 / 0.5) |
- no instruction: commitment − none: +12.5 [+3.1, +21.9] -> **not shown**
- no instruction: pledge − commitment: +14.1 [+4.2, +23.4] -> **yes**
- CHANNELS: commitment − none: +6.8 [-0.5, +13.5] -> **not shown**
- CHANNELS: pledge − commitment: +11.5 [+6.2, +16.7] -> **yes**
- six own decisions vs a pledge the other way (no instruction): P record + R pledge: 22.9% penalise, share toward the pledge 0.79 [0.75, 0.82] -> **pledge wins**; R record + P pledge: 21.4% penalise, share toward the pledge 0.20 [0.12, 0.28] -> **record wins**

## 7a three formats, matched content (carry-over P − R)

| context | own-turn record (= 2a own turns) | the same cases as user examples | instruction |
|---|---|---|---|
| none (primary) | +97.4 | +100.0 (100.0 / 0.0) | +100.0 (100.0 / 0.0) |
| CHANNELS | +100.0 | +98.4 (98.4 / 0.0) | +91.1 (91.1 / 0.0) |
- no instruction: all three formats >= 85 carry-over -> **saturated, not read**
- CHANNELS: all three formats >= 85 carry-over -> **saturated, not read**

## Predictions

| event | p | happened |
|---|---|---|
| 2a options carry more than reasons only (summary A − B >= +15) (k = 2) | 0.6 | no |
| 2a reasons add to options in a summary (AB − A >= +10) (k = 2) | 0.5 | no |
| 2a own turns carry more than the options + reasons summary (full − AB >= +15) (k = 2) | 0.75 | no |
| 2a the reason field adds in own turns (full − bare >= +10) (k = 2) | 0.25 | no |
| 2a a closed-only summary licenses nothing (|closed − base| CI within ±15, both contexts pooled) (k = 2) | 0.75 | yes |
| 2b voice matters (|I − agent| >= 15) | 0.25 | no |
| 2c instances carry more than a generalisation (inst − gen >= +15) | 0.5 | no |
| 3a options element carries (main effect >= +15) (k = 2) | 0.65 | no |
| 3a reasons element carries (k = 2) | 0.5 | no |
| 3a outcomes element carries (k = 2) | 0.6 | no |
| 3a approval element carries (k = 2) | 0.05 | no |
| 3a reflection element carries (k = 2) | 0.45 | yes |
| 3c details bind (real − generic >= +15) | 0.4 | yes |
| 3d a commitment adds to one own decision (commit − none >= +15) | 0.35 | no |
| 8c a pledge adds to a commitment (pledge − commit >= +10) | 0.15 | yes |
| 3d a pledge beats six own decisions the other way (share toward the pledge > 0.5) | 0.4 | no |
| 7a own record carries more than the same cases as user examples (record − examples >= +15, CHANNELS) | 0.55 | no |
| 7a an instruction carries more than user examples (instr − examples >= +15, CHANNELS) | 0.5 | no |
| 7a record and instruction differ (|record − instr| >= 15, CHANNELS) | 0.3 | no |
