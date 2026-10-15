# jl3: J-lens hallucination rate on matched present/absent twins (Qwen3.6-27B)

## Verification (greedy answers; association/typo via the appended question)

| family | items | candidate pairs | present answers right | twin answers right | verified candidates | items with a verified pair |
|---|---|---|---|---|---|---|
| association | 102 | 203 | 1/102 | 0/203 | 0 | 0 |
| multihop | 87 | 174 | 52/87 | 99/174 | 76 | 47 |
| multilingual | 106 | 210 | 79/106 | 148/210 | 126 | 69 |
| order-ops | 51 | 98 | 37/51 | 69/98 | 66 | 36 |
| poetry | 98 | 194 | 80/97 | 116/194 | 102 | 68 |
| typo | 94 | 184 | 28/94 | 81/184 | 34 | 25 |
| all | 538 | 1063 | 277/537 | 513/1063 | 404 | 245 |

**245 verified pairs (one per distinct present prompt) across 5 families** (card target: >= 150 across >= 3).


## Primary: X (the original's intermediate), present prompt vs its absent twin, one pair per item

| lens | layers | n | hit rate (present) | 95% CI | false-positive rate (absent twin) | 95% CI | FP, twin shares X's context words | FP, no shared words | AUROC (best rank) | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| jlens | 20-60/4 | 245 | 0.49 | [0.42, 0.55] | 0.176 | [0.131, 0.224] | 9/64 (0.14) | 34/181 (0.19) | 0.683 | [0.656, 0.714] |
| jlens | all 0-62 | 245 | 0.55 | [0.49, 0.61] | 0.257 | [0.204, 0.314] | 13/64 (0.20) | 50/181 (0.28) | 0.684 | [0.655, 0.717] |
| jpp | 20-60/4 | 245 | 0.59 | [0.53, 0.65] | 0.233 | [0.184, 0.286] | 16/64 (0.25) | 41/181 (0.23) | 0.726 | [0.694, 0.759] |
| jpp | all 0-62 | 245 | 0.72 | [0.66, 0.77] | 0.367 | [0.306, 0.424] | 23/64 (0.36) | 67/181 (0.37) | 0.746 | [0.714, 0.781] |
| logit | 20-60/4 | 245 | 0.50 | [0.44, 0.56] | 0.098 | [0.065, 0.135] | 7/64 (0.11) | 17/181 (0.09) | 0.710 | [0.680, 0.743] |
| logit | all 0-62 | 245 | 0.56 | [0.49, 0.62] | 0.127 | [0.090, 0.171] | 10/64 (0.16) | 21/181 (0.12) | 0.721 | [0.688, 0.757] |
| jlens_cos (exploratory: WorkspaceBench cosine readout) | 20-60/4 | 245 | 0.50 | [0.44, 0.57] | 0.171 | [0.127, 0.220] | 13/64 (0.20) | 29/181 (0.16) | 0.688 | [0.660, 0.719] |

### By family (layers 20-60/4)

| family | n | J-lens hit | J-lens FP | J++ hit | J++ FP | logit hit | logit FP |
|---|---|---|---|---|---|---|---|
| multihop | 47 | 0.57 | 0.30 | 0.77 | 0.36 | 0.55 | 0.15 |
| multilingual | 69 | 0.88 | 0.20 | 0.94 | 0.28 | 0.88 | 0.09 |
| order-ops | 36 | 0.56 | 0.39 | 0.28 | 0.14 | 0.53 | 0.17 |
| poetry | 68 | 0.01 | 0.01 | 0.18 | 0.06 | 0.03 | 0.01 |
| typo | 25 | 0.40 | 0.00 | 0.88 | 0.48 | 0.56 | 0.16 |

## Secondary: reverse direction, X' (the twin's own intermediate) in its twin vs in the original

| lens | layers | n | hit rate (present) | 95% CI | false-positive rate (absent twin) | 95% CI | FP, twin shares X's context words | FP, no shared words | AUROC (best rank) | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| jlens | 20-60/4 | 242 | 0.52 | [0.46, 0.59] | 0.182 | [0.132, 0.231] | 14/63 (0.22) | 30/179 (0.17) | 0.695 | [0.666, 0.725] |
| jlens | all 0-62 | 242 | 0.57 | [0.50, 0.63] | 0.289 | [0.231, 0.347] | 19/63 (0.30) | 51/179 (0.28) | 0.701 | [0.674, 0.729] |
| jpp | 20-60/4 | 242 | 0.64 | [0.58, 0.70] | 0.190 | [0.140, 0.240] | 17/63 (0.27) | 29/179 (0.16) | 0.743 | [0.710, 0.778] |
| jpp | all 0-62 | 242 | 0.70 | [0.64, 0.76] | 0.310 | [0.252, 0.368] | 24/63 (0.38) | 51/179 (0.28) | 0.740 | [0.706, 0.775] |
| logit | 20-60/4 | 242 | 0.51 | [0.45, 0.57] | 0.074 | [0.041, 0.112] | 7/63 (0.11) | 11/179 (0.06) | 0.721 | [0.688, 0.753] |
| logit | all 0-62 | 242 | 0.57 | [0.50, 0.63] | 0.136 | [0.095, 0.182] | 13/63 (0.21) | 20/179 (0.11) | 0.738 | [0.705, 0.770] |
| jlens_cos (exploratory: WorkspaceBench cosine readout) | 20-60/4 | 242 | 0.55 | [0.48, 0.61] | 0.186 | [0.136, 0.236] | 15/63 (0.24) | 30/179 (0.17) | 0.699 | [0.669, 0.730] |

## Secondary: all verified candidates (up to 2 twins per item)

| lens | layers | n | hit rate (present) | 95% CI | false-positive rate (absent twin) | 95% CI | FP, twin shares X's context words | FP, no shared words | AUROC (best rank) | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| jlens | 20-60/4 | 404 | 0.53 | [0.48, 0.58] | 0.173 | [0.139, 0.210] | 13/98 (0.13) | 57/306 (0.19) | 0.706 | [0.682, 0.730] |
| jlens | all 0-62 | 404 | 0.60 | [0.55, 0.65] | 0.267 | [0.225, 0.312] | 19/98 (0.19) | 89/306 (0.29) | 0.711 | [0.687, 0.736] |
| jpp | 20-60/4 | 404 | 0.61 | [0.56, 0.66] | 0.213 | [0.173, 0.252] | 23/98 (0.23) | 63/306 (0.21) | 0.740 | [0.714, 0.765] |
| jpp | all 0-62 | 404 | 0.74 | [0.69, 0.78] | 0.376 | [0.332, 0.423] | 36/98 (0.37) | 116/306 (0.38) | 0.758 | [0.733, 0.784] |
| logit | 20-60/4 | 404 | 0.54 | [0.49, 0.59] | 0.097 | [0.067, 0.126] | 9/98 (0.09) | 30/306 (0.10) | 0.735 | [0.710, 0.761] |
| logit | all 0-62 | 404 | 0.59 | [0.54, 0.64] | 0.126 | [0.094, 0.161] | 13/98 (0.13) | 38/306 (0.12) | 0.746 | [0.719, 0.773] |
| jlens_cos (exploratory: WorkspaceBench cosine readout) | 20-60/4 | 404 | 0.54 | [0.49, 0.59] | 0.171 | [0.136, 0.210] | 20/98 (0.20) | 49/306 (0.16) | 0.709 | [0.684, 0.733] |

### J-lens false positives (band), 43 absent twins

- poetry / couplet-breath-death: X ['death'] rank 2 in twin '...ld his final, fading word,\nAnd closed his eyes to greet his sovereign '
- multihop / christmas-season: X ['Christmas'] rank 9 in twin '...act: The season when the holiday with fireworks and parades occurs is '
- multihop / etym-saturn-position: X ['Saturn'] rank 3 in twin '...t: The planet named after the Roman king of the gods is planet number '
- multihop / chem-atmosphere-Z: X ['nitrogen'] rank 5 in twin '...number of the gas that makes up about 21 percent of the atmosphere is '
- multihop / holiday-christmas-monthnum: X ['December'] rank 3 in twin '...Fact: Mexican Independence Day is celebrated in month number '
- multihop / holiday-halloween-monthnum: X ['October'] rank 8 in twin '...Fact: Boxing Day is celebrated in month number '
- multihop / etym-caesar-monthnum: X ['July'] rank 6 in twin '...Fact: The month named after Augustus is month number '
- multihop / inv-balloon-opposite: X ['up'] rank 1 in twin '...rection opposite to the one a stone dropped in water sinks toward is "'
- multihop / inv-roots-opposite: X ['down'] rank 1 in twin '... direction opposite to the one smoke from a campfire rises toward is "'
- multihop / firstletter-halloween-month: X ['October'] rank 7 in twin '...Fact: The first letter of the month containing Christmas is "'
- multihop / firstletter-valentines-month: X ['February'] rank 7 in twin '...Fact: The first letter of the month containing Christmas is "'
- multihop / double-dice-faces: X ['6', 'six'] rank 1 in twin '...Fact: Double the number of sides on a triangle is '
- multihop / dbl-obituary-antonym: X ['death'] rank 1 in twin '...Fact: The antonym of what a birth announcement announces is "'
- multilingual / french-season-summer: X ['autumn'] rank 4 in twin '...La saison après le printemps est l''
- multilingual / russian-number-four: X ['four', 'five'] rank 8 in twin '...После "два" идёт "'
- multilingual / spanish-body-see: X ['see', 'eyes'] rank 2 in twin '...Oímos con los "'
- multilingual / german-family-aunt: X ['sister', 'aunt'] rank 7 in twin '...Der Bruder deiner Mutter ist dein "'
- multilingual / italian-family-grandmother: X ['mother', 'grandmother'] rank 3 in twin '...Il padre di tua madre è tuo "'
- multilingual / bulgarian-color-grass: X ['grass', 'green'] rank 7 in twin '...Цветът на "виното" е "'
- multilingual / fr-month-after-jan: X ['January', 'February'] rank 10 in twin '...Le mois après août est "'
- multilingual / fr-month-after-may: X ['May', 'June'] rank 5 in twin '...Le mois après juillet est "'
- multilingual / fr-month-before-mar: X ['February'] rank 2 in twin '...Le mois avant avril est "'
- multilingual / de-month-after-jan: X ['January', 'February'] rank 7 in twin '...Der Monat nach Oktober ist "'
- multilingual / de-month-before-dec: X ['December', 'November'] rank 8 in twin '...Der Monat vor April ist "'
- multilingual / pt-month-after-apr: X ['April', 'May'] rank 7 in twin '...O mês depois de janeiro é "'
- multilingual / frn-month-after-jun: X ['June', 'July'] rank 4 in twin '...Le mois après mars est "'
- order-ops / parens-add-mult: X ['5'] rank 7 in twin '...(2 + 6) * 4 = '
- order-ops / mult-parens-sub: X ['3'] rank 8 in twin '...(9 - 4) * 2 = '
- order-ops / mult-sub-left: X ['20'] rank 2 in twin '...3 * 7 - 8 = '
- order-ops / div-parens-add: X ['5'] rank 1 in twin '...20 / (2 + 2) = '
- order-ops / div-sub-left: X ['4'] rank 7 in twin '...30 / 5 - 1 = '
- order-ops / chain-add-mult-add: X ['6'] rank 2 in twin '...1 + 4 * 2 + 4 = '
- order-ops / mult-mult-left: X ['6'] rank 5 in twin '...2 * 5 * 4 = '
- order-ops / chain-mult-mult-add: X ['6'] rank 6 in twin '...5 * 1 + 4 * 5 = '
- order-ops / nested-mult-sub-add: X ['7'] rank 4 in twin '...((3 * 4) - 6) + 2 = '
- order-ops / nested-div-add-sub: X ['8'] rank 4 in twin '...((12 / 3) + 5) - 3 = '
- order-ops / nested-sub-div-mult: X ['3'] rank 6 in twin '...((10 - 6) / 2) * 5 = '
- order-ops / redundant-parens-mult: X ['6'] rank 5 in twin '...(2 * 4) + 5 = '
- order-ops / pystar-sub: X ['4'] rank 7 in twin '...(7 - 2)**2 = '
- multihop / etym-wargod-month: X ['March'] rank 10 in twin '...ed after the Roman dictator who reformed the calendar is month number '

## Descriptive: top-10 word tokens at the read position (layers 20-60/4) that appear in the next 50 generated tokens (present prompts of the verified pairs; jlens/REPORT.md had 31% / 26% / 41% before claims)

- jlens: 583/3808 = 0.15
- jpp: 933/8289 = 0.11
- logit: 553/4482 = 0.12
- jlens_cos: 666/6138 = 0.11
