# op5, openai/gpt-5.6-sol (gpt)

Unparsed > 10%: none

| cell | penalise % | n |
|---|---|---|
| alog_P_ch | 7.8 | 64 |
| alog_P_no | 95.3 | 64 |
| alog_R_ch | 0.0 | 64 |
| alog_R_no | 62.5 | 64 |
| asst_P_ch | 14.1 | 64 |
| asst_P_no | 100.0 | 64 |
| asst_R_ch | 0.0 | 64 |
| asst_R_no | 10.9 | 64 |
| dev_P_ch | 53.1 | 64 |
| dev_P_no | 100.0 | 64 |
| dev_R_ch | 0.0 | 64 |
| dev_R_no | 4.7 | 64 |
| none_ch | 0.0 | 64 |
| none_no | 65.6 | 64 |
| sys_P_ch | 45.3 | 64 |
| sys_P_no | 96.9 | 64 |
| sys_R_ch | 0.0 | 64 |
| sys_R_no | 14.1 | 64 |
| tins_ch | 0.0 | 64 |
| tins_no | 98.4 | 64 |
| tool_P_ch | 28.1 | 64 |
| tool_P_no | 100.0 | 64 |
| tool_R_ch | 0.0 | 64 |
| tool_R_no | 37.5 | 64 |
| uins_ch | 6.2 | 64 |
| uins_no | 100.0 | 64 |
| unv_P_ch | 18.8 | 64 |
| unv_P_no | 98.4 | 64 |
| unv_R_ch | 0.0 | 64 |
| unv_R_no | 37.5 | 64 |
| user_P_ch | 45.3 | 64 |
| user_P_no | 96.9 | 64 |
| user_R_ch | 0.0 | 64 |
| user_R_no | 12.5 | 64 |
| ver_P_ch | 26.6 | 64 |
| ver_P_no | 96.9 | 64 |
| ver_R_ch | 0.0 | 64 |
| ver_R_no | 29.7 | 64 |

- carry-over, own turns, no instruction: +89.1 [+78.1, +98.4]
- under CHANNELS: own penalise record 14.1% vs rule alone 0.0% (+14.1 [+3.1, +26.6])

| placement | carry-over, no instruction | carry-over under CHANNELS | penalise after P under CHANNELS |
|---|---|---|---|
| asst | +89.1 [+78.1, +98.4] | +14.1 [+3.1, +26.6] | 14.1 |
| alog | +32.8 [+17.2, +50.0] | +7.8 [+0.0, +17.2] | 7.8 |
| user | +84.4 [+75.0, +92.2] | +45.3 [+34.4, +57.8] | 45.3 |
| tool | +62.5 [+48.4, +76.6] | +28.1 [+14.1, +43.8] | 28.1 |
| sys | +82.8 [+71.9, +92.2] | +45.3 [+32.8, +57.8] | 45.3 |
| dev | +95.3 [+90.6, +100.0] | +53.1 [+39.1, +67.2] | 53.1 |

| placement | P − none, no instruction | R − none, no instruction | P − rule alone | R − rule alone |
|---|---|---|---|---|
| asst | +34.4 [+18.8, +50.0] | -54.7 [-71.9, -35.9] | +14.1 [+3.1, +26.6] | +0.0 [+0.0, +0.0] |
| alog | +29.7 [+15.6, +45.3] | -3.1 [-17.2, +10.9] | +7.8 [+0.0, +17.2] | +0.0 [+0.0, +0.0] |
| user | +31.2 [+15.6, +48.4] | -53.1 [-68.8, -37.5] | +45.3 [+34.4, +57.8] | +0.0 [+0.0, +0.0] |
| tool | +34.4 [+18.8, +50.0] | -28.1 [-42.2, -14.1] | +28.1 [+14.1, +43.8] | +0.0 [+0.0, +0.0] |
| sys | +31.2 [+17.2, +48.4] | -51.6 [-65.6, -35.9] | +45.3 [+32.8, +57.8] | +0.0 [+0.0, +0.0] |
| dev | +34.4 [+18.8, +50.0] | -60.9 [-75.0, -45.3] | +53.1 [+39.1, +67.2] | +0.0 [+0.0, +0.0] |
- asst_over_user_ch: -31.2 [-42.2, -21.9]
- asst_over_tool_ch: -14.1 [-26.6, +0.0]
- sys_over_tool_ch: +17.2 [+1.6, +32.8]
- turns_over_alog_ch: +6.2 [-3.1, +17.2]
- dev_over_user_ch: +7.8 [-3.1, +20.3]

- uins: without the rule +34.4 [+18.8, +50.0] over none; under the rule +6.2 [+1.6, +12.5] over the rule alone
- tins: without the rule +32.8 [+18.8, +50.0] over none; under the rule +0.0 [+0.0, +0.0] over the rule alone
- own penalise record − user instruction, both under the rule: +7.8 [-4.7, +21.9]
- verified − unverified carry-over: +6.2 [-4.7, +18.8] (no instruction); +7.8 [+0.0, +15.6] (under the rule)

| event | p | happened |
|---|---|---|
| record_carries | 0.95 | yes |
| record_beats_rule | 0.08 | no |
| alog_carries | 0.75 | yes |
| user_carries | 0.65 | yes |
| tool_carries | 0.6 | yes |
| sys_carries | 0.8 | yes |
| dev_carries | 0.8 | yes |
| asst_over_user_ch | 0.2 | no |
| asst_over_tool_ch | 0.2 | no |
| sys_over_tool_ch | 0.3 | yes |
| turns_over_alog_ch | 0.15 | no |
| dev_over_user_ch | 0.3 | no |
| uins_effective | 0.75 | yes |
| rule_holds_vs_uins | 0.8 | no |
| tins_effective | 0.45 | yes |
| rule_holds_vs_tins | 0.9 | yes |
| precedent_beats_uins_ch | 0.15 | no |
| discounts_unverified | 0.3 | no |

10 of 18 read events happened (expected 9.13)
