# op5, Qwen/Qwen3.6-27B (27b)

Unparsed > 10%: none

| cell | penalise % | n |
|---|---|---|
| alog_P_ch | 67.7 | 192 |
| alog_P_no | 92.7 | 192 |
| alog_R_ch | 0.5 | 192 |
| alog_R_no | 37.0 | 192 |
| asst_P_ch | 100.0 | 192 |
| asst_P_no | 100.0 | 192 |
| asst_R_ch | 0.0 | 192 |
| asst_R_no | 3.1 | 192 |
| none_ch | 0.0 | 192 |
| none_no | 45.8 | 192 |
| sys_P_ch | 88.0 | 192 |
| sys_P_no | 98.4 | 192 |
| sys_R_ch | 0.0 | 192 |
| sys_R_no | 28.1 | 192 |
| tins_ch | 0.0 | 192 |
| tins_no | 100.0 | 192 |
| tool_P_ch | 42.2 | 192 |
| tool_P_no | 96.4 | 192 |
| tool_R_ch | 0.0 | 192 |
| tool_R_no | 15.6 | 192 |
| uins_ch | 100.0 | 192 |
| uins_no | 100.0 | 192 |
| unv_P_ch | 24.5 | 192 |
| unv_P_no | 95.3 | 192 |
| unv_R_ch | 0.0 | 192 |
| unv_R_no | 30.2 | 192 |
| user_P_ch | 81.8 | 192 |
| user_P_no | 97.4 | 192 |
| user_R_ch | 0.5 | 192 |
| user_R_no | 22.4 | 192 |
| ver_P_ch | 44.8 | 192 |
| ver_P_no | 97.4 | 192 |
| ver_R_ch | 0.0 | 192 |
| ver_R_no | 16.7 | 192 |

- carry-over, own turns, no instruction: +96.9 [+93.8, +99.5]
- under CHANNELS: own penalise record 100.0% vs rule alone 0.0% (+100.0 [+100.0, +100.0])

| placement | carry-over, no instruction | carry-over under CHANNELS | penalise after P under CHANNELS |
|---|---|---|---|
| asst | +96.9 [+93.8, +99.5] | +100.0 [+100.0, +100.0] | 100.0 |
| alog | +55.7 [+47.4, +64.1] | +67.2 [+53.6, +78.6] | 67.7 |
| user | +75.0 [+67.2, +81.8] | +81.2 [+74.0, +88.0] | 81.8 |
| tool | +80.7 [+74.5, +87.0] | +42.2 [+33.3, +51.0] | 42.2 |
| sys | +70.3 [+64.1, +76.6] | +88.0 [+80.7, +93.8] | 88.0 |

| placement | P − none, no instruction | R − none, no instruction | P − rule alone | R − rule alone |
|---|---|---|---|---|
| asst | +54.2 [+41.7, +66.7] | -42.7 [-54.7, -31.2] | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] |
| alog | +46.9 [+35.4, +59.4] | -8.9 [-20.3, +3.1] | +67.7 [+54.2, +79.7] | +0.5 [+0.0, +1.6] |
| user | +51.6 [+40.1, +63.0] | -23.4 [-34.4, -12.5] | +81.8 [+75.0, +88.0] | +0.5 [+0.0, +1.6] |
| tool | +50.5 [+38.5, +62.5] | -30.2 [-43.2, -17.2] | +42.2 [+33.3, +51.0] | +0.0 [+0.0, +0.0] |
| sys | +52.6 [+40.6, +64.6] | -17.7 [-30.7, -4.7] | +88.0 [+80.7, +93.8] | +0.0 [+0.0, +0.0] |
- asst_over_user_ch: +18.8 [+12.0, +26.0]
- asst_over_tool_ch: +57.8 [+49.0, +66.7]
- sys_over_tool_ch: +45.8 [+39.6, +52.6]
- turns_over_alog_ch: +32.8 [+21.4, +46.4]

- uins: without the rule +54.2 [+41.7, +66.7] over none; under the rule +100.0 [+100.0, +100.0] over the rule alone
- tins: without the rule +54.2 [+41.7, +66.7] over none; under the rule +0.0 [+0.0, +0.0] over the rule alone
- own penalise record − user instruction, both under the rule: +0.0 [+0.0, +0.0]
- verified − unverified carry-over: +15.6 [+9.4, +21.9] (no instruction); +20.3 [+13.0, +28.6] (under the rule)

| event | p | happened |
|---|---|---|
| record_carries | 0.97 | yes |
| record_beats_rule | 0.95 | yes |
| alog_carries | 0.9 | yes |
| user_carries | 0.9 | yes |
| tool_carries | 0.85 | yes |
| sys_carries | 0.9 | yes |
| asst_over_user_ch | 0.65 | yes |
| asst_over_tool_ch | 0.6 | yes |
| sys_over_tool_ch | 0.35 | yes |
| turns_over_alog_ch | 0.5 | yes |
| uins_effective | 0.7 | yes |
| rule_holds_vs_uins | 0.4 | no |
| tins_effective | 0.6 | yes |
| rule_holds_vs_tins | 0.5 | yes |
| precedent_beats_uins_ch | 0.5 | no |
| discounts_unverified | 0.15 | yes |

14 of 16 read events happened (expected 10.42)
