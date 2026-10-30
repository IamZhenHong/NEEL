# op3 (Qwen/Qwen3.5-9B; persona/operative/op3/results/op3_9b)

Unparsed after the second chance: anc 0.0%, lad 0.0%, lad2 0.0%, voice 0.0%, gen 0.0%, spec 0.0%, fa 0.0%, fa2 0.0%, cmt 0.0%, fmt 0.0%

Carry-over = penalise after six own penalise decisions − after six own refer decisions (the same cases); [p0/p1/p2] = carry-over per paraphrase. Rules read in the no-instruction context unless stated; CHANNELS reported alongside.

Baselines (no record): no instruction 52.6%, CHANNELS 46.9%

## 2a compression ladder, k = 2 earlier decisions (PRIMARY; carry-over P − R; penalise % after P / R)

| context | own turns | turns, no reason | episode log | summary: options + reasons | options | reasons | closed only (P = R) |
|---|---|---|---|---|---|---|---|
| none (primary) | +58.3 (98.4 / 40.1) [+58/+56/+61] | +68.8 (97.4 / 28.6) [+70/+73/+62] | +43.7 (79.2 / 35.4) [+39/+48/+44] | +45.8 (84.9 / 39.1) [+45/+52/+41] | +56.8 (89.6 / 32.8) [+62/+52/+56] | +14.6 (52.6 / 38.0) [+9/+12/+22] | 39.6 (base 52.6) |
| CHANNELS | +58.9 (98.4 / 39.6) [+64/+58/+55] | +70.8 (100.0 / 29.2) [+67/+75/+70] | +51.0 (85.9 / 34.9) [+39/+59/+55] | +47.4 (83.3 / 35.9) [+42/+48/+52] | +54.7 (91.7 / 37.0) [+66/+47/+52] | +18.8 (59.9 / 41.1) [+14/+22/+20] | 46.9 (base 46.9) |

**no instruction, k = 2** (own-turn carry-over +58.3)
- options − reasons only (A − B): +42.2 [+29.7, +55.2] -> **yes**
- options + reasons − options (AB − A): -10.9 [-19.8, -2.1] -> **not shown**
- own turns − summary with options + reasons: +12.5 [-1.6, +29.2] -> **not shown**
- own turns − turns with the reason blanked: -10.4 [-19.3, -1.6] -> **not shown**
- episode log − summary with options + reasons -2.1 [-10.9, +6.8] (reported)
- closed-only summary − no record (penalise) -13.0 [-20.8, -6.2] (rule pooled below)
**CHANNELS, k = 2** (own-turn carry-over +58.9)
- options − reasons only (A − B): +35.9 [+22.9, +49.0] -> **yes**
- options + reasons − options (AB − A): -7.3 [-18.2, +4.2] -> **not shown**
- own turns − summary with options + reasons: +11.5 [-2.1, +24.5] -> **not shown**
- own turns − turns with the reason blanked: -12.0 [-17.2, -6.8] -> **not shown**
- episode log − summary with options + reasons +3.6 [-2.6, +9.9] (reported)
- closed-only summary − no record (penalise) +0.0 [-7.8, +6.2] (rule pooled below)
- **k = 2: closed-only summary − no record, pooled over both contexts: -6.5 [-13.3, -0.8] -> licenses nothing** (equivalence: CI within ±15)

## 2a compression ladder, k = 6 earlier decisions (secondary, read only where the ceiling gate leaves room; carry-over P − R; penalise % after P / R)

| context | own turns | turns, no reason | episode log | summary: options + reasons | options | reasons | closed only (P = R) |
|---|---|---|---|---|---|---|---|
| none (primary) | +97.4 (99.5 / 2.1) [+98/+97/+97] | +99.5 (99.5 / 0.0) [+98/+100/+100] | +73.4 (93.8 / 20.3) [+78/+69/+73] | +70.8 (94.8 / 24.0) [+73/+67/+72] | +64.6 (98.4 / 33.9) [+61/+64/+69] | +43.8 (74.0 / 30.2) [+44/+38/+50] | 49.0 (base 52.6) |
| CHANNELS | +98.4 (100.0 / 1.6) [+100/+100/+95] | +99.5 (99.5 / 0.0) [+100/+98/+100] | +77.1 (95.8 / 18.8) [+78/+78/+75] | +74.0 (96.9 / 22.9) [+75/+67/+80] | +61.5 (96.9 / 35.4) [+64/+66/+55] | +43.8 (72.4 / 28.6) [+52/+36/+44] | 47.9 (base 46.9) |

**no instruction, k = 6** (own-turn carry-over +97.4)
- options − reasons only (A − B): +20.8 [+8.3, +34.4] -> **yes**
- options + reasons − options (AB − A): +6.2 [-5.2, +17.7] -> **not shown**
- own turns − summary with options + reasons: +26.6 [+15.1, +38.5] -> **yes**
- own turns − turns with the reason blanked: -2.1 [-6.2, +0.0] -> **not read (ceiling)**
- episode log − summary with options + reasons +2.6 [-5.7, +9.9] (reported)
- closed-only summary − no record (penalise) -3.6 [-10.9, +3.6] (rule pooled below)
**CHANNELS, k = 6** (own-turn carry-over +98.4)
- options − reasons only (A − B): +17.7 [+4.7, +30.7] -> **yes**
- options + reasons − options (AB − A): +12.5 [+2.6, +23.4] -> **yes**
- own turns − summary with options + reasons: +24.5 [+16.1, +33.9] -> **yes**
- own turns − turns with the reason blanked: -1.0 [-3.6, +1.0] -> **not read (ceiling)**
- episode log − summary with options + reasons +3.1 [-3.6, +10.9] (reported)
- closed-only summary − no record (penalise) +1.0 [-5.2, +6.3] (rule pooled below)
- **k = 6: closed-only summary − no record, pooled over both contexts: -1.3 [-5.7, +2.9] -> licenses nothing** (equivalence: CI within ±15)

## 2b the summary's voice (options only; carry-over P − R)

| context | neutral bullets (= 2a options) | I chose | you chose | the agent chose |
|---|---|---|---|---|
| none (primary) | +64.6 [+61/+64/+69] | +75.5 [+70/+81/+75] | +76.6 [+69/+81/+80] | +71.4 [+64/+72/+78] |
| CHANNELS | +61.5 [+64/+66/+55] | +79.7 [+73/+86/+80] | +76.0 [+67/+81/+80] | +76.0 [+72/+80/+77] |
- no instruction: I − the agent +4.2 [-0.0, +8.9] -> **not shown**; I − neutral +10.9 [+4.2, +18.8], you − agent +5.2 [+1.6, +9.4] (reported)
- CHANNELS: I − the agent +3.6 [-1.6, +8.3] -> **not shown**; I − neutral +18.2 [+7.8, +28.6], you − agent +0.0 [-9.4, +7.8] (reported)

## 2c instances vs generalisation (summary slot; carry-over P − R)

| context | instances ("cases ... were penalised") | generalisation ("the agent tends to") | batch 1 habit ("you have always") |
|---|---|---|---|
| none (primary) | +41.7 (75.0 / 33.3) | +88.0 (96.4 / 8.3) | +34.9 (82.3 / 47.4) |
| CHANNELS | +49.0 (80.7 / 31.8) | +72.4 (80.7 / 8.3) | +62.0 (85.4 / 23.4) |
- no instruction: instances − generalisation: -46.4 [-56.8, -37.0] -> **not read (ceiling)**; generalisation − habit +53.1 [+42.2, +63.5] (reported)
- CHANNELS: instances − generalisation: -23.4 [-32.3, -14.6] -> **not shown**; generalisation − habit +10.4 [-2.6, +23.4] (reported)

## 3c specificity (own turns: the real six cases vs six templated generic cases)

- no instruction: real +97.4 − generic +46.4 (75.5 / 29.2): +51.0 [+41.7, +59.9] -> **yes**
- CHANNELS: real +98.4 − generic +56.2 (72.9 / 16.7): +42.2 [+34.9, +49.5] -> **yes**

## 3a element ablation, k = 2 earlier decisions (PRIMARY): 2^(5-1) fractional factorial (I = ABCDE; no instruction; carry-over P − R)

A options, B own reasons, C outcomes, D desk-lead approval, E own reflection. Main effects are aliased with 4-way interactions and 2-way with 3-way (resolution V).

| run | ABCDE | ABC | ABD | ABE | ACD | ACE | ADE | BCD | BCE | BDE | CDE | A | B | C | D | E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry-over | +59 | +50 | +45 | +57 | +58 | +64 | +69 | +40 | +58 | +44 | +67 | +57 | +15 | +48 | -3 | +41 |

- main effect, options: +18.8 [+12.8, +24.5] -> **carries**
- main effect, reasons: -4.1 [-8.5, +0.6] -> **not shown**
- main effect, outcomes: +15.0 [+11.3, +18.7] -> **carries**
- main effect, approval: -1.2 [-4.1, +1.7] -> **not shown**
- main effect, reflection: +18.9 [+14.5, +23.1] -> **carries**
- interaction options x reasons (reported): -5.1 [-8.3, -1.8]
- interaction options x outcomes (reported): -14.1 [-17.2, -10.9]
- interaction reasons x reflection (reported): -1.6 [-4.8, +1.8]

## 3a element ablation, k = 6 earlier decisions (secondary): 2^(5-1) fractional factorial (I = ABCDE; no instruction; carry-over P − R)

A options, B own reasons, C outcomes, D desk-lead approval, E own reflection. Main effects are aliased with 4-way interactions and 2-way with 3-way (resolution V).

| run | ABCDE | ABC | ABD | ABE | ACD | ACE | ADE | BCD | BCE | BDE | CDE | A | B | C | D | E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry-over | +82 | +78 | +76 | +84 | +74 | +84 | +83 | +69 | +83 | +69 | +77 | +65 | +44 | +60 | +3 | +51 |

- main effect, options: +21.1 [+16.7, +25.6] -> **carries**
- main effect, reasons: +10.8 [+5.8, +15.8] -> **not shown**
- main effect, outcomes: +16.8 [+13.3, +20.1] -> **carries**
- main effect, approval: -2.1 [-4.4, +0.2] -> **not shown**
- main effect, reflection: +18.3 [+14.4, +22.4] -> **carries**
- interaction options x reasons (reported): -7.5 [-10.6, -4.5]
- interaction options x outcomes (reported): -14.0 [-16.7, -11.1]
- interaction reasons x reflection (reported): -5.1 [-9.1, -1.0]

## 3d / 8c forward commitment (after ONE own decision; carry-over P − R)

| context | no commitment | "going forward I'll ..." | "I will always ..." |
|---|---|---|---|
| none (primary) | +38.5 (93.8 / 55.2) | +39.6 (92.2 / 52.6) | +53.1 (94.8 / 41.7) |
| CHANNELS | +37.0 (92.2 / 55.2) | +50.0 (93.8 / 43.8) | +59.9 (98.4 / 38.5) |
- no instruction: commitment − none: +1.0 [-6.8, +9.4] -> **not shown**
- no instruction: pledge − commitment: +13.5 [+3.6, +23.4] -> **yes**
- CHANNELS: commitment − none: +13.0 [+3.6, +23.4] -> **not shown**
- CHANNELS: pledge − commitment: +9.9 [+1.6, +17.7] -> **not shown**
- six own decisions vs a pledge the other way (no instruction): P record + R pledge: 99.0% penalise, share toward the pledge 0.01 [0.00, 0.02] -> **record wins**; R record + P pledge: 3.6% penalise, share toward the pledge 0.02 [-0.01, 0.05] -> **record wins**

## 7a three formats, matched content (carry-over P − R)

| context | own-turn record (= 2a own turns) | the same cases as user examples | instruction |
|---|---|---|---|
| none (primary) | +97.4 | +85.9 (93.8 / 7.8) | +100.0 (100.0 / 0.0) |
| CHANNELS | +98.4 | +85.4 (96.9 / 11.5) | +99.0 (100.0 / 1.0) |
- no instruction: all three formats >= 85 carry-over -> **saturated, not read**
- CHANNELS: all three formats >= 85 carry-over -> **saturated, not read**

## Predictions

| event | p | happened |
|---|---|---|
| 2a options carry more than reasons only (summary A − B >= +15) (k = 2) | 0.6 | yes |
| 2a reasons add to options in a summary (AB − A >= +10) (k = 2) | 0.15 | no |
| 2a own turns carry more than the options + reasons summary (full − AB >= +15) (k = 2) | 0.6 | no |
| 2a the reason field adds in own turns (full − bare >= +10) (k = 2) | 0.1 | no |
| 2a a closed-only summary licenses nothing (|closed − base| CI within ±15, both contexts pooled) (k = 2) | 0.7 | yes |
| 2b voice matters (|I − agent| >= 15) | 0.15 | no |
| 2c instances carry more than a generalisation (inst − gen >= +15) | 0.55 | no |
| 3a options element carries (main effect >= +15) (k = 2) | 0.65 | yes |
| 3a reasons element carries (k = 2) | 0.25 | no |
| 3a outcomes element carries (k = 2) | 0.55 | yes |
| 3a approval element carries (k = 2) | 0.05 | no |
| 3a reflection element carries (k = 2) | 0.35 | yes |
| 3c details bind (real − generic >= +15) | 0.3 | yes |
| 3d a commitment adds to one own decision (commit − none >= +15) | 0.3 | no |
| 8c a pledge adds to a commitment (pledge − commit >= +10) | 0.1 | yes |
| 3d a pledge beats six own decisions the other way (share toward the pledge > 0.5) | 0.2 | no |
| 7a own record carries more than the same cases as user examples (record − examples >= +15, CHANNELS) | 0.5 | no |
| 7a an instruction carries more than user examples (instr − examples >= +15, CHANNELS) | 0.4 | no |
| 7a record and instruction differ (|record − instr| >= 15, CHANNELS) | 0.3 | no |
