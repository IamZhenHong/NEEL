# op5, google/gemma-4-31B-it (gemma)

Unparsed > 10%: none

| cell | penalise % | n |
|---|---|---|
| alog_P_ch | 0.5 | 192 |
| alog_P_no | 87.5 | 192 |
| alog_R_ch | 0.0 | 192 |
| alog_R_no | 19.3 | 192 |
| asst_P_ch | 70.3 | 192 |
| asst_P_no | 100.0 | 192 |
| asst_R_ch | 0.0 | 192 |
| asst_R_no | 0.0 | 192 |
| none_ch | 0.0 | 192 |
| none_no | 52.6 | 192 |
| sys_P_ch | 11.5 | 192 |
| sys_P_no | 97.9 | 192 |
| sys_R_ch | 0.0 | 192 |
| sys_R_no | 0.0 | 192 |
| tins_ch | 0.0 | 192 |
| tins_no | 100.0 | 192 |
| tool_P_ch | 0.0 | 192 |
| tool_P_no | 91.1 | 192 |
| tool_R_ch | 0.0 | 192 |
| tool_R_no | 0.0 | 192 |
| uins_ch | 0.0 | 192 |
| uins_no | 100.0 | 192 |
| unv_P_ch | 0.0 | 192 |
| unv_P_no | 89.1 | 192 |
| unv_R_ch | 0.0 | 192 |
| unv_R_no | 0.0 | 192 |
| user_P_ch | 0.5 | 192 |
| user_P_no | 97.9 | 192 |
| user_R_ch | 0.0 | 192 |
| user_R_no | 0.0 | 192 |
| ver_P_ch | 0.0 | 192 |
| ver_P_no | 89.6 | 192 |
| ver_R_ch | 0.0 | 192 |
| ver_R_no | 0.0 | 192 |

- carry-over, own turns, no instruction: +100.0 [+100.0, +100.0]
- under CHANNELS: own penalise record 70.3% vs rule alone 0.0% (+70.3 [+55.2, +83.9])

| placement | carry-over, no instruction | carry-over under CHANNELS | penalise after P under CHANNELS |
|---|---|---|---|
| asst | +100.0 [+100.0, +100.0] | +70.3 [+55.2, +83.9] | 70.3 |
| alog | +68.2 [+52.1, +82.8] | +0.5 [+0.0, +1.6] | 0.5 |
| user | +97.9 [+93.8, +100.0] | +0.5 [+0.0, +1.6] | 0.5 |
| tool | +91.1 [+79.2, +99.0] | +0.0 [+0.0, +0.0] | 0.0 |
| sys | +97.9 [+93.8, +100.0] | +11.5 [+3.1, +21.4] | 11.5 |

| placement | P − none, no instruction | R − none, no instruction | P − rule alone | R − rule alone |
|---|---|---|---|---|
| asst | +47.4 [+29.2, +65.6] | -52.6 [-70.8, -34.4] | +70.3 [+55.2, +83.9] | +0.0 [+0.0, +0.0] |
| alog | +34.9 [+19.8, +50.5] | -33.3 [-49.5, -18.8] | +0.5 [+0.0, +1.6] | +0.0 [+0.0, +0.0] |
| user | +45.3 [+28.1, +62.5] | -52.6 [-70.8, -34.4] | +0.5 [+0.0, +1.6] | +0.0 [+0.0, +0.0] |
| tool | +38.5 [+22.9, +53.6] | -52.6 [-70.8, -34.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| sys | +45.3 [+28.1, +62.5] | -52.6 [-70.8, -34.4] | +11.5 [+3.1, +21.4] | +0.0 [+0.0, +0.0] |
- asst_over_user_ch: +69.8 [+54.7, +83.3]
- asst_over_tool_ch: +70.3 [+55.2, +83.9]
- sys_over_tool_ch: +11.5 [+3.1, +21.4]
- turns_over_alog_ch: +69.8 [+54.7, +83.3]

- uins: without the rule +47.4 [+29.2, +65.6] over none; under the rule +0.0 [+0.0, +0.0] over the rule alone
- tins: without the rule +47.4 [+29.2, +65.6] over none; under the rule +0.0 [+0.0, +0.0] over the rule alone
- own penalise record − user instruction, both under the rule: +70.3 [+55.2, +83.9]
- verified − unverified carry-over: +0.5 [-2.1, +3.1] (no instruction); +0.0 [+0.0, +0.0] (under the rule)

| event | p | happened |
|---|---|---|
| record_carries | 0.97 | yes |
| record_beats_rule | 0.1 | yes |
| alog_carries | 0.9 | yes |
| user_carries | 0.9 | yes |
| tool_carries | 0.85 | yes |
| sys_carries | 0.9 | yes |
| asst_over_user_ch | 0.25 | yes |
| asst_over_tool_ch | 0.25 | yes |
| sys_over_tool_ch | 0.2 | no |
| turns_over_alog_ch | 0.2 | yes |
| uins_effective | 0.7 | yes |
| rule_holds_vs_uins | 0.75 | yes |
| tins_effective | 0.6 | yes |
| rule_holds_vs_tins | 0.8 | yes |
| precedent_beats_uins_ch | 0.15 | yes |
| discounts_unverified | 0.15 | no |

14 of 16 read events happened (expected 8.67)
