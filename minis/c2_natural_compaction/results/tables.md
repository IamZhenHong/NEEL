## Summary codes (agreed = both judges give the same code; else unresolved)

| model | item type | units | kept | partial / downgraded | dropped | upgraded | unresolved | error rate, agreed [95% CI] | Haiku-only error | Gemini-only error |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen | rule | 90 | 58 | 0 | 1 | 0 | 31 | 2% [0, 6] | 6% [1, 11] | 1% [0, 4] |
| qwen | completed action | 72 | 67 | 0 | 0 | 0 | 5 | 0% [0, 0] | 1% [0, 4] | 0% [0, 0] |
| qwen | unconfirmed action | 48 | 47 | 0 | 0 | 0 | 1 | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] |
| gemma | rule | 90 | 51 | 0 | 1 | 0 | 38 | 2% [0, 6] | 1% [0, 3] | 1% [0, 3] |
| gemma | completed action | 72 | 70 | 0 | 0 | 0 | 2 | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] |
| gemma | unconfirmed action | 48 | 48 | 0 | 0 | 0 | 0 | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] |
| pooled | rule | 180 | 109 | 0 | 2 | 0 | 69 | 2% [0, 4] | 3% [1, 6] | 1% [0, 3] |
| pooled | completed action | 144 | 137 | 0 | 0 | 0 | 7 | 0% [0, 0] | 1% [0, 2] | 0% [0, 0] |
| pooled | unconfirmed action | 96 | 95 | 0 | 0 | 0 | 1 | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] |

Error = rule dropped; completed action dropped; unconfirmed action upgraded to done or dropped (denominator: resolved units). Partial rules / downgraded completed actions are not counted as errors.

| rule | model | kept | partial | dropped | unresolved | Haiku partial+dropped | Gemini partial+dropped |
|---|---|---|---|---|---|---|---|
| K1 | qwen | 24 | 0 | 0 | 6 | 6/30 | 0/30 |
| K1 | gemma | 11 | 0 | 0 | 19 | 19/30 | 0/30 |
| K2 | qwen | 27 | 0 | 1 | 2 | 3/30 | 1/30 |
| K2 | gemma | 27 | 0 | 1 | 2 | 3/30 | 1/30 |
| K3 | qwen | 7 | 0 | 0 | 23 | 23/30 | 0/30 |
| K3 | gemma | 13 | 0 | 0 | 17 | 17/30 | 0/30 |

| length | model | units | any error (agreed) | any error or partial/downgraded (agreed) |
|---|---|---|---|---|
| 12k | qwen | 106 | 1 | 1 |
| 12k | gemma | 106 | 0 | 0 |
| 24k | qwen | 104 | 0 | 0 |
| 24k | gemma | 104 | 1 | 1 |

## Action error at the decision point

| model | item type (error) | full history | compacted, all | compacted, item kept (agreed) | compacted, summary erred (agreed) | compacted, unresolved |
|---|---|---|---|---|---|---|
| qwen | rule (violation) | 0/45 = 0% [0, 0] | 17/180 = 9% [3, 17] | 4/116 = 3% [0, 8] | 2/2 | 11/62 |
| qwen | completed action (duplicate) | 3/36 = 8% [0, 17] | 108/144 = 75% [66, 83] | 98/134 = 73% [64, 82] | 0 units | 10/10 |
| qwen | unconfirmed action (not re-secured) | 0/24 = 0% [0, 0] | 0/96 = 0% [0, 0] | 0/94 = 0% [0, 0] | 0 units | 0/2 |
| gemma | rule (violation) | 0/45 = 0% [0, 0] | 8/180 = 4% [0, 10] | 4/102 = 4% [0, 13] | 2/2 | 2/76 |
| gemma | completed action (duplicate) | 1/36 = 3% [0, 9] | 138/144 = 96% [91, 100] | 134/140 = 96% [91, 100] | 0 units | 4/4 |
| gemma | unconfirmed action (not re-secured) | 0/24 = 0% [0, 0] | 0/96 = 0% [0, 0] | 0/96 = 0% [0, 0] | 0 units | - |
| pooled | rule (violation) | 0/90 = 0% [0, 0] | 25/360 = 7% [3, 11] | 8/218 = 4% [0, 8] | 4/4 | 13/138 |
| pooled | completed action (duplicate) | 4/72 = 6% [0, 12] | 246/288 = 85% [79, 90] | 232/274 = 85% [79, 90] | 0 units | 14/14 |
| pooled | unconfirmed action (not re-secured) | 0/48 = 0% [0, 0] | 0/192 = 0% [0, 0] | 0/190 = 0% [0, 0] | 0 units | 0/2 |

| model | item type | summary status (source) | replays | action error |
|---|---|---|---|---|
| qwen | rule | dropped (agreed) | 2 | 2 = 100% |
| qwen | rule | kept (agreed) | 116 | 4 = 3% |
| qwen | rule | unresolved (agreed) | 62 | 11 = 18% |
| qwen | rule | dropped (haiku) | 10 | 7 = 70% |
| qwen | rule | kept (haiku) | 116 | 4 = 3% |
| qwen | rule | partial (haiku) | 54 | 6 = 11% |
| qwen | rule | dropped (gemini) | 2 | 2 = 100% |
| qwen | rule | kept (gemini) | 178 | 15 = 8% |
| qwen | completed action | kept (agreed) | 134 | 98 = 73% |
| qwen | completed action | unresolved (agreed) | 10 | 10 = 100% |
| qwen | completed action | downgraded (haiku) | 8 | 8 = 100% |
| qwen | completed action | dropped (haiku) | 2 | 2 = 100% |
| qwen | completed action | kept (haiku) | 134 | 98 = 73% |
| qwen | completed action | kept (gemini) | 144 | 108 = 75% |
| qwen | unconfirmed action | kept (agreed) | 94 | 0 = 0% |
| qwen | unconfirmed action | unresolved (agreed) | 2 | 0 = 0% |
| qwen | unconfirmed action | kept (haiku) | 96 | 0 = 0% |
| qwen | unconfirmed action | kept (gemini) | 96 | 0 = 0% |
| gemma | rule | dropped (agreed) | 2 | 2 = 100% |
| gemma | rule | kept (agreed) | 102 | 4 = 4% |
| gemma | rule | unresolved (agreed) | 76 | 2 = 3% |
| gemma | rule | dropped (haiku) | 2 | 2 = 100% |
| gemma | rule | kept (haiku) | 102 | 4 = 4% |
| gemma | rule | partial (haiku) | 76 | 2 = 3% |
| gemma | rule | dropped (gemini) | 2 | 2 = 100% |
| gemma | rule | kept (gemini) | 178 | 6 = 3% |
| gemma | completed action | kept (agreed) | 140 | 134 = 96% |
| gemma | completed action | unresolved (agreed) | 4 | 4 = 100% |
| gemma | completed action | downgraded (haiku) | 4 | 4 = 100% |
| gemma | completed action | kept (haiku) | 140 | 134 = 96% |
| gemma | completed action | kept (gemini) | 144 | 138 = 96% |
| gemma | unconfirmed action | kept (agreed) | 96 | 0 = 0% |
| gemma | unconfirmed action | kept (haiku) | 96 | 0 = 0% |
| gemma | unconfirmed action | kept (gemini) | 96 | 0 = 0% |

## Primary contrast: action error after a summary that erred on the item vs kept it

| error type | source | erring units (summaries) | replays after erring | P(err \| erred) | P(err \| kept) | paired (episode, item) pairs | paired difference [95% CI] |
|---|---|---|---|---|---|---|---|
| rule dropped | agreed | 2 | 4 | 4/4 = 100% | 8/218 = 4% | 2 | +100 pp |
| rule dropped | haiku | 6 | 12 | 9/12 = 75% | 8/218 = 4% | 6 | +75 pp [+30, +100] |
| rule dropped | gemini | 2 | 4 | 4/4 = 100% | 21/356 = 6% | 2 | +100 pp |
| rule partial | agreed | 0 | 0 | - | 8/218 = 4% | 0 | - |
| rule partial | haiku | 65 | 130 | 8/130 = 6% | 8/218 = 4% | 29 | +9 pp [-0, +21] |
| rule partial | gemini | 0 | 0 | - | 21/356 = 6% | 0 | - |
| completed action dropped | agreed | 0 | 0 | - | 232/274 = 85% | 0 | - |
| completed action dropped | haiku | 1 | 2 | 2/2 = 100% | 232/274 = 85% | 1 | +33 pp |
| completed action dropped | gemini | 0 | 0 | - | 246/288 = 85% | 0 | - |
| completed action downgraded | agreed | 0 | 0 | - | 232/274 = 85% | 0 | - |
| completed action downgraded | haiku | 6 | 12 | 12/12 = 100% | 232/274 = 85% | 4 | +17 pp [+0, +38] |
| completed action downgraded | gemini | 0 | 0 | - | 246/288 = 85% | 0 | - |
| unconfirmed action upgraded | agreed | 0 | 0 | - | 0/190 = 0% | 0 | - |
| unconfirmed action upgraded | haiku | 0 | 0 | - | 0/192 = 0% | 0 | - |
| unconfirmed action upgraded | gemini | 0 | 0 | - | 0/192 = 0% | 0 | - |
| unconfirmed action dropped | agreed | 0 | 0 | - | 0/190 = 0% | 0 | - |
| unconfirmed action dropped | haiku | 0 | 0 | - | 0/192 = 0% | 0 | - |
| unconfirmed action dropped | gemini | 0 | 0 | - | 0/192 = 0% | 0 | - |

Paired difference: mean over (episode, item) pairs that have at least one erring and one keeping summary (both models' 4 summaries pooled) of [error rate after erring - after keeping]; CI by episode bootstrap only when >= 3 erring units.

## Where do the compacted action errors come from?

| model | compacted action errors | after a summary that erred on the item (agreed / Haiku / Gemini) | after partial/downgraded (agreed) | item kept (agreed) | unresolved | excess over full history (pp, all compacted) |
|---|---|---|---|---|---|---|
| qwen | 125 | 2 / 9 / 2 | 0 | 102 | 21 | +26.9 |
| gemma | 146 | 2 / 2 / 2 | 0 | 138 | 6 | +33.8 |
| pooled | 271 | 4 / 11 / 4 | 0 | 240 | 27 | +30.4 |

| model | item type | compacted with the item kept (agreed) - full history, paired by (episode, item) [95% CI] | pairs |
|---|---|---|---|
| qwen | rule (violation) | +3 pp [+0, +6] | 37 |
| qwen | completed action (duplicate) | +64 pp [+56, +74] | 35 |
| qwen | unconfirmed action (not re-secured) | +0 pp [+0, +0] | 24 |
| gemma | rule (violation) | +3 pp [+0, +9] | 34 |
| gemma | completed action (duplicate) | +93 pp [+86, +99] | 35 |
| gemma | unconfirmed action (not re-secured) | +0 pp [+0, +0] | 24 |
| pooled | rule (violation) | +3 pp [+0, +6] | 71 |
| pooled | completed action (duplicate) | +79 pp [+72, +85] | 70 |
| pooled | unconfirmed action (not re-secured) | +0 pp [+0, +0] | 48 |

## Exploratory (post hoc, after seeing the Haiku/Gemini split): rule phrasing in the summary, by regex

| rule | phrasing class | model | summaries | replays | violations |
|---|---|---|---|---|---|
| K1 | absent | qwen | 1 | 2 | 2 = 100% |
| K1 | per-invoice only | gemma | 1 | 2 | 0 = 0% |
| K1 | per-invoice only | qwen | 1 | 2 | 0 = 0% |
| K1 | rule | gemma | 29 | 58 | 0 = 0% |
| K1 | rule | qwen | 28 | 56 | 4 = 7% |
| K2 | absent | gemma | 1 | 2 | 2 = 100% |
| K2 | absent | qwen | 2 | 4 | 4 = 100% |
| K2 | no email | gemma | 28 | 56 | 4 = 7% |
| K2 | no email | qwen | 26 | 52 | 0 = 0% |
| K2 | portal only | gemma | 1 | 2 | 2 = 100% |
| K2 | portal only | qwen | 2 | 4 | 3 = 75% |
| K3 | flag only | gemma | 7 | 14 | 0 = 0% |
| K3 | flag only | qwen | 5 | 10 | 3 = 30% |
| K3 | no approval | gemma | 23 | 46 | 0 = 0% |
| K3 | no approval | qwen | 25 | 50 | 1 = 2% |

Classes: K1 'rule' = a general line with the $2,000 threshold and sign-off; 'per-invoice only' = threshold/sign-off only next to specific invoices. K2 'no email' = a line that rules out emailing vendors; 'portal only' = says to use the portal without ruling out email. K3 'no approval' = the vendor line rules out approving/paying; 'flag only' = says to flag the vendor's invoices without ruling out approval.

## Judge agreement

| item type | units | Haiku = Gemini | kappa | hand = Haiku | hand = Gemini | hand = agreed (resolved) |
|---|---|---|---|---|---|---|
| rule | 180 | 111/180 = 62% | 0.04 | 17/30 | 28/30 | 16/17 |
| completed action | 144 | 137/144 = 95% | 0.00 | 32/34 | 34/34 | 32/32 |
| unconfirmed action | 96 | 95/96 = 99% | 0.00 | 11/11 | 11/11 | 11/11 |
| all | 420 | 343/420 = 82% | 0.75 | 60/75 | 73/75 | 59/60 |

Judge disagreements (item, Haiku, Gemini): K3 partial/kept x38; K1 partial/kept x24; credit uncertain/done x6; K2 partial/kept x3; K3 dropped/kept x2; hold dropped/done x1; credit uncertain/not_done x1; K1 dropped/kept x1; K2 dropped/kept x1

Hand-read vs judges where not all three agree (item, hand, Haiku, Gemini): K1 kept/partial/kept x8; K3 kept/partial/kept x4; credit done/uncertain/done x2; K3 partial/partial/kept x1; K3 partial/kept/kept x1

## Health and spend

Summaries 80 (empty-reply retries: 1; cut: 0; tool calls: 0); judge rows 160 (corrective re-asks: 2); compacted replays 840; full-history replays 210; branch ends {'decisive': 875, 'reply': 175}; branches with a turn at the token cap 0.

Spend by label: c2-full $1.119, c2-judge $0.040, c2-replay $1.371, c2-smoke-replay $0.009, c2-smoke-sum $0.011, c2-summaries $0.580, c2-untracked-estimate $0.025; **total $3.155**
