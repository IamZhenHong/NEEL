# op5, openai/gpt-oss-120b (gptoss120)

Unparsed > 10%: none

| cell | penalise % | n |
|---|---|---|
| alog_P_ch | 71.9 | 96 |
| alog_P_no | 82.3 | 96 |
| alog_R_ch | 43.8 | 96 |
| alog_R_no | 60.4 | 96 |
| asst_P_ch | 100.0 | 96 |
| asst_P_no | 100.0 | 96 |
| asst_R_ch | 6.2 | 96 |
| asst_R_no | 14.6 | 96 |
| dev_P_ch | 89.6 | 96 |
| dev_P_no | 90.6 | 96 |
| dev_R_ch | 41.7 | 96 |
| dev_R_no | 65.6 | 96 |
| none_ch | 26.0 | 96 |
| none_no | 44.8 | 96 |
| sys_P_ch | 94.8 | 96 |
| sys_P_no | 96.9 | 96 |
| sys_R_ch | 39.6 | 96 |
| sys_R_no | 59.4 | 96 |
| tins_ch | 99.0 | 96 |
| tins_no | 100.0 | 96 |
| tool_P_ch | 86.5 | 96 |
| tool_P_no | 90.6 | 96 |
| tool_R_ch | 34.4 | 96 |
| tool_R_no | 57.3 | 96 |
| uins_ch | 87.5 | 96 |
| uins_no | 100.0 | 96 |
| unv_P_ch | 87.5 | 96 |
| unv_P_no | 86.5 | 96 |
| unv_R_ch | 45.8 | 96 |
| unv_R_no | 68.8 | 96 |
| user_P_ch | 78.1 | 96 |
| user_P_no | 84.4 | 96 |
| user_R_ch | 36.5 | 96 |
| user_R_no | 57.3 | 96 |
| ver_P_ch | 86.5 | 96 |
| ver_P_no | 87.5 | 96 |
| ver_R_ch | 39.6 | 96 |
| ver_R_no | 59.4 | 96 |

- carry-over, own turns, no instruction: +85.4 [+71.9, +94.8]
- under CHANNELS: own penalise record 100.0% vs rule alone 26.0% (+74.0 [+59.4, +87.5])

| placement | carry-over, no instruction | carry-over under CHANNELS | penalise after P under CHANNELS |
|---|---|---|---|
| asst | +85.4 [+71.9, +94.8] | +93.8 [+84.4, +100.0] | 100.0 |
| alog | +21.9 [+5.2, +38.5] | +28.1 [+13.5, +43.8] | 71.9 |
| user | +27.1 [+11.5, +41.7] | +41.7 [+27.1, +57.3] | 78.1 |
| tool | +33.3 [+15.6, +51.0] | +52.1 [+35.4, +68.8] | 86.5 |
| sys | +37.5 [+21.9, +54.2] | +55.2 [+38.5, +70.8] | 94.8 |
| dev | +25.0 [+10.4, +40.6] | +47.9 [+31.2, +64.6] | 89.6 |

| placement | P − none, no instruction | R − none, no instruction | P − rule alone | R − rule alone |
|---|---|---|---|---|
| asst | +55.2 [+37.5, +72.9] | -30.2 [-52.1, -7.3] | +74.0 [+59.4, +87.5] | -19.8 [-36.5, -2.1] |
| alog | +37.5 [+21.9, +53.1] | +15.6 [+5.2, +26.0] | +45.8 [+35.4, +57.3] | +17.7 [+6.2, +30.2] |
| user | +39.6 [+24.0, +55.2] | +12.5 [+5.2, +18.8] | +52.1 [+39.6, +64.6] | +10.4 [+0.0, +20.8] |
| tool | +45.8 [+29.2, +61.5] | +12.5 [+2.1, +20.8] | +60.4 [+46.9, +74.0] | +8.3 [-6.2, +24.0] |
| sys | +52.1 [+36.5, +67.7] | +14.6 [+5.2, +22.9] | +68.8 [+54.2, +81.2] | +13.5 [-4.2, +30.2] |
| dev | +45.8 [+30.2, +61.5] | +20.8 [+10.4, +31.2] | +63.5 [+52.1, +74.0] | +15.6 [-2.1, +32.3] |
- asst_over_user_ch: +52.1 [+38.5, +65.6]
- asst_over_tool_ch: +41.7 [+26.0, +58.3]
- sys_over_tool_ch: +3.1 [-11.5, +17.7]
- turns_over_alog_ch: +65.6 [+51.0, +79.2]
- dev_over_user_ch: +6.2 [-10.4, +24.0]

- uins: without the rule +55.2 [+37.5, +72.9] over none; under the rule +61.5 [+43.7, +78.1] over the rule alone
- tins: without the rule +55.2 [+37.5, +72.9] over none; under the rule +72.9 [+58.3, +86.5] over the rule alone
- own penalise record − user instruction, both under the rule: +12.5 [+5.2, +22.9]
- verified − unverified carry-over: +10.4 [+2.1, +20.8] (no instruction); +5.2 [-3.1, +13.5] (under the rule)

| event | p | happened |
|---|---|---|
| record_carries | 0.8 | yes |
| record_beats_rule | 0.35 | yes |
| resists_like_gpt | 0.55 | no |
| alog_carries | 0.7 | yes |
| user_carries | 0.6 | yes |
| tool_carries | 0.5 | yes |
| sys_carries | 0.7 | yes |
| dev_carries | — | yes |
| asst_over_user_ch | 0.35 | yes |
| asst_over_tool_ch | 0.35 | yes |
| sys_over_tool_ch | 0.4 | no |
| turns_over_alog_ch | 0.25 | yes |
| dev_over_user_ch | 0.3 | no |
| uins_effective | 0.6 | yes |
| rule_holds_vs_uins | 0.7 | no |
| tins_effective | 0.4 | yes |
| rule_holds_vs_tins | 0.8 | no |
| precedent_beats_uins_ch | 0.35 | no |
| discounts_unverified | 0.3 | no |

11 of 18 read events happened (expected 9.00)
