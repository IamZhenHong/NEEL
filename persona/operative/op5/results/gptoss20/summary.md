# op5, openai/gpt-oss-20b (gptoss20)

Unparsed > 10%: none

| cell | penalise % | n |
|---|---|---|
| alog_P_ch | 83.2 | 95 |
| alog_P_no | 89.6 | 96 |
| alog_R_ch | 44.7 | 94 |
| alog_R_no | 62.5 | 96 |
| asst_P_ch | 77.1 | 96 |
| asst_P_no | 91.7 | 96 |
| asst_R_ch | 30.9 | 94 |
| asst_R_no | 63.5 | 96 |
| dev_P_ch | 79.2 | 96 |
| dev_P_no | 93.6 | 94 |
| dev_R_ch | 21.1 | 95 |
| dev_R_no | 57.9 | 95 |
| none_ch | 36.8 | 95 |
| none_no | 56.2 | 96 |
| sys_P_ch | 80.4 | 92 |
| sys_P_no | 94.6 | 93 |
| sys_R_ch | 24.2 | 95 |
| sys_R_no | 51.6 | 95 |
| tins_ch | 40.6 | 96 |
| tins_no | 94.8 | 96 |
| tool_P_ch | 66.7 | 96 |
| tool_P_no | 89.6 | 96 |
| tool_R_ch | 31.2 | 96 |
| tool_R_no | 62.5 | 96 |
| uins_ch | 100.0 | 96 |
| uins_no | 100.0 | 96 |
| unv_P_ch | 75.0 | 96 |
| unv_P_no | 91.7 | 96 |
| unv_R_ch | 35.4 | 96 |
| unv_R_no | 68.8 | 96 |
| user_P_ch | 77.1 | 96 |
| user_P_no | 90.5 | 95 |
| user_R_ch | 35.4 | 96 |
| user_R_no | 58.9 | 95 |
| ver_P_ch | 66.3 | 95 |
| ver_P_no | 89.6 | 96 |
| ver_R_ch | 26.0 | 96 |
| ver_R_no | 66.7 | 96 |

- carry-over, own turns, no instruction: +28.1 [+14.6, +42.7]
- under CHANNELS: own penalise record 77.1% vs rule alone 36.8% (+40.2 [+29.2, +52.1])

| placement | carry-over, no instruction | carry-over under CHANNELS | penalise after P under CHANNELS |
|---|---|---|---|
| asst | +28.1 [+14.6, +42.7] | +46.2 [+33.5, +58.8] | 77.1 |
| alog | +27.1 [+14.6, +40.6] | +38.5 [+29.6, +47.1] | 83.2 |
| user | +31.6 [+16.8, +47.9] | +41.7 [+21.9, +59.4] | 77.1 |
| tool | +27.1 [+12.5, +42.7] | +35.4 [+27.1, +43.8] | 66.7 |
| sys | +43.0 [+27.1, +58.8] | +56.2 [+40.1, +72.3] | 80.4 |
| dev | +35.7 [+20.7, +52.0] | +58.1 [+41.4, +72.8] | 79.2 |

| placement | P − none, no instruction | R − none, no instruction | P − rule alone | R − rule alone |
|---|---|---|---|---|
| asst | +35.4 [+21.9, +50.0] | +7.3 [-8.3, +21.9] | +40.2 [+29.2, +52.1] | -6.0 [-20.7, +10.3] |
| alog | +33.3 [+20.8, +45.8] | +6.2 [-7.3, +19.8] | +46.3 [+36.0, +57.9] | +7.8 [-4.9, +19.8] |
| user | +34.3 [+21.8, +47.8] | +2.7 [-10.5, +15.2] | +40.2 [+28.1, +52.1] | -1.4 [-13.5, +12.0] |
| tool | +33.3 [+19.8, +46.9] | +6.2 [-7.3, +20.8] | +29.8 [+20.4, +39.6] | -5.6 [-15.0, +3.9] |
| sys | +38.4 [+25.8, +52.0] | -4.7 [-17.7, +7.9] | +43.6 [+30.7, +55.0] | -12.6 [-25.0, +0.0] |
| dev | +37.4 [+25.9, +48.9] | +1.6 [-10.8, +13.3] | +42.3 [+31.2, +53.1] | -15.8 [-27.7, -3.5] |
- asst_over_user_ch: +4.6 [-12.9, +21.1]
- asst_over_tool_ch: +10.8 [+0.2, +20.5]
- sys_over_tool_ch: +20.8 [+3.7, +38.4]
- turns_over_alog_ch: +7.8 [-5.2, +21.4]
- dev_over_user_ch: +16.4 [+6.2, +26.9]

- uins: without the rule +43.8 [+30.2, +57.3] over none; under the rule +63.2 [+51.0, +75.3] over the rule alone
- tins: without the rule +38.5 [+27.1, +50.0] over none; under the rule +3.8 [-3.9, +12.1] over the rule alone
- own penalise record − user instruction, both under the rule: -22.9 [-33.3, -13.5]
- verified − unverified carry-over: +0.0 [-8.3, +9.4] (no instruction); +0.7 [-14.6, +17.4] (under the rule)

| event | p | happened |
|---|---|---|
| record_carries | 0.8 | yes |
| record_beats_rule | 0.45 | yes |
| resists_like_gpt | 0.45 | no |
| alog_carries | 0.7 | yes |
| user_carries | 0.6 | yes |
| tool_carries | 0.5 | yes |
| sys_carries | 0.7 | yes |
| dev_carries | — | yes |
| asst_over_user_ch | 0.35 | no |
| asst_over_tool_ch | 0.35 | no |
| sys_over_tool_ch | 0.4 | yes |
| turns_over_alog_ch | 0.25 | no |
| dev_over_user_ch | 0.3 | yes |
| uins_effective | 0.6 | yes |
| rule_holds_vs_uins | 0.7 | no |
| tins_effective | 0.4 | yes |
| rule_holds_vs_tins | 0.8 | no |
| precedent_beats_uins_ch | 0.35 | no |
| discounts_unverified | 0.3 | no |

10 of 18 read events happened (expected 9.00)
