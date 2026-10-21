# op5, Qwen/Qwen3.5-9B (9b)

Unparsed > 10%: none

| cell | penalise % | n |
|---|---|---|
| alog_P_ch | 82.3 | 192 |
| alog_P_no | 79.2 | 192 |
| alog_R_ch | 28.1 | 192 |
| alog_R_no | 39.6 | 192 |
| asst_P_ch | 100.0 | 192 |
| asst_P_no | 100.0 | 192 |
| asst_R_ch | 1.0 | 192 |
| asst_R_no | 3.1 | 192 |
| none_ch | 42.2 | 192 |
| none_no | 51.6 | 192 |
| sys_P_ch | 96.4 | 192 |
| sys_P_no | 95.3 | 192 |
| sys_R_ch | 21.9 | 192 |
| sys_R_no | 27.6 | 192 |
| tins_ch | 99.0 | 192 |
| tins_no | 99.5 | 192 |
| tool_P_ch | 90.6 | 192 |
| tool_P_no | 93.2 | 192 |
| tool_R_ch | 28.1 | 192 |
| tool_R_no | 27.6 | 192 |
| uins_ch | 100.0 | 192 |
| uins_no | 100.0 | 192 |
| unv_P_ch | 92.7 | 192 |
| unv_P_no | 92.7 | 192 |
| unv_R_ch | 26.0 | 192 |
| unv_R_no | 28.6 | 192 |
| user_P_ch | 95.3 | 192 |
| user_P_no | 91.7 | 192 |
| user_R_ch | 20.8 | 192 |
| user_R_no | 21.4 | 192 |
| ver_P_ch | 91.7 | 192 |
| ver_P_no | 93.8 | 192 |
| ver_R_ch | 28.1 | 192 |
| ver_R_no | 26.6 | 192 |

- carry-over, own turns, no instruction: +96.9 [+93.8, +99.5]
- under CHANNELS: own penalise record 100.0% vs rule alone 42.2% (+57.8 [+44.3, +70.8])

| placement | carry-over, no instruction | carry-over under CHANNELS | penalise after P under CHANNELS |
|---|---|---|---|
| asst | +96.9 [+93.8, +99.5] | +99.0 [+97.4, +100.0] | 100.0 |
| alog | +39.6 [+29.2, +50.5] | +54.2 [+41.7, +67.2] | 82.3 |
| user | +70.3 [+59.4, +79.7] | +74.5 [+67.2, +81.2] | 95.3 |
| tool | +65.6 [+58.3, +73.4] | +62.5 [+51.6, +72.4] | 90.6 |
| sys | +67.7 [+60.4, +75.5] | +74.5 [+67.2, +81.8] | 96.4 |

| placement | P − none, no instruction | R − none, no instruction | P − rule alone | R − rule alone |
|---|---|---|---|---|
| asst | +48.4 [+34.4, +62.5] | -48.4 [-60.4, -35.9] | +57.8 [+44.3, +70.8] | -41.1 [-54.2, -28.6] |
| alog | +27.6 [+19.8, +35.4] | -12.0 [-21.4, -2.6] | +40.1 [+27.6, +53.6] | -14.1 [-21.9, -6.2] |
| user | +40.1 [+28.1, +51.0] | -30.2 [-42.7, -18.2] | +53.1 [+39.1, +67.2] | -21.4 [-31.2, -12.0] |
| tool | +41.7 [+29.7, +52.1] | -24.0 [-37.0, -12.0] | +48.4 [+34.9, +62.0] | -14.1 [-23.4, -5.2] |
| sys | +43.8 [+31.8, +54.7] | -24.0 [-36.5, -12.0] | +54.2 [+41.1, +66.7] | -20.3 [-29.7, -10.9] |
- asst_over_user_ch: +24.5 [+17.7, +31.3]
- asst_over_tool_ch: +36.5 [+26.6, +47.4]
- sys_over_tool_ch: +12.0 [+3.6, +21.9]
- turns_over_alog_ch: +44.8 [+31.8, +57.3]

- uins: without the rule +48.4 [+34.4, +62.5] over none; under the rule +57.8 [+44.3, +70.8] over the rule alone
- tins: without the rule +47.9 [+33.9, +62.0] over none; under the rule +56.8 [+44.3, +68.8] over the rule alone
- own penalise record − user instruction, both under the rule: +0.0 [+0.0, +0.0]
- verified − unverified carry-over: +3.1 [-2.6, +8.9] (no instruction); -3.1 [-9.9, +4.7] (under the rule)

| event | p | happened |
|---|---|---|
| record_carries | 0.97 | yes |
| record_beats_rule | 0.95 | yes |
| alog_carries | 0.9 | yes |
| user_carries | 0.9 | yes |
| tool_carries | 0.85 | yes |
| sys_carries | 0.9 | yes |
| asst_over_user_ch | 0.3 | yes |
| asst_over_tool_ch | 0.3 | yes |
| sys_over_tool_ch | 0.35 | no |
| turns_over_alog_ch | 0.5 | yes |
| uins_effective | 0.85 | yes |
| rule_holds_vs_uins | 0.2 | no |
| tins_effective | 0.8 | yes |
| rule_holds_vs_tins | 0.3 | no |
| precedent_beats_uins_ch | 0.2 | no |
| discounts_unverified | 0.1 | no |

11 of 16 read events happened (expected 9.37)
