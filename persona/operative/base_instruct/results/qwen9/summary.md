# base vs instruct, qwen9: Qwen/Qwen3.5-9B-Base vs Qwen/Qwen3.5-9B

## G2 (competence: option mass >= 0.5 and P(stated option) >= 0.8 on the control items)

| model | rendering | option mass | P(stated) | pass |
|---|---|---|---|---|
| pt | plain | 0.991 | 0.985 | True |
| pt | chat | 0.996 | 1.0 | True |
| it | plain | 0.995 | 0.997 | True |
| it | chat | 0.998 | 1.0 | True |

## G1 (it log-prob P(penalise) vs op5's sampled rates over the 22 cells: Pearson >= 0.7, own-turn carry-over sign matches; Spearman reported)

| rendering | Pearson | Spearman | carry-over log-prob | carry-over op5 | pass |
|---|---|---|---|---|---|
| plain | 0.798 | 0.682 | +16.3 | +96.9 | True |
| chat | 0.996 | 0.986 | +94.7 | +96.9 | True |

Readable renderings (G1 on it and G2 on pt): ['plain', 'chat']; counted: plain

## Rendering: plain (counted)

| cell | pt P(penalise) % | it P(penalise) % | op5 it sampled % |
|---|---|---|---|
| none_no | 35.3 | 37.6 | 51.6 |
| none_ch | 37.5 | 39.2 | 42.2 |
| asst_P_no | 79.4 | 41.0 | 100.0 |
| asst_P_ch | 81.2 | 30.5 | 100.0 |
| asst_R_no | 10.3 | 24.7 | 3.1 |
| asst_R_ch | 10.2 | 22.2 | 1.0 |
| alog_P_no | 43.2 | 56.7 | 79.2 |
| alog_P_ch | 45.1 | 57.9 | 82.3 |
| alog_R_no | 30.3 | 33.4 | 39.6 |
| alog_R_ch | 31.4 | 34.4 | 28.1 |
| user_P_no | 41.7 | 55.8 | 91.7 |
| user_P_ch | 41.0 | 56.5 | 95.3 |
| user_R_no | 28.5 | 33.2 | 21.4 |
| user_R_ch | 28.9 | 33.3 | 20.8 |
| tool_P_no | 46.8 | 61.3 | 93.2 |
| tool_P_ch | 46.2 | 63.8 | 90.6 |
| tool_R_no | 32.0 | 38.8 | 27.6 |
| tool_R_ch | 32.8 | 39.3 | 28.1 |
| sys_P_no | 42.8 | 61.1 | 95.3 |
| sys_P_ch | 42.0 | 63.8 | 96.4 |
| sys_R_no | 31.0 | 34.4 | 27.6 |
| sys_R_ch | 30.2 | 34.2 | 21.9 |

| placement | pt carry-over, no rule | pt carry-over under CHANNELS | it carry-over, no rule | it carry-over under CHANNELS |
|---|---|---|---|---|
| asst | +69.2 [+65.3, +72.6] | +71.0 [+67.5, +74.3] | +16.3 [+1.7, +32.7] | +8.3 [-6.8, +24.2] |
| alog | +12.9 [+10.4, +15.8] | +13.7 [+11.4, +16.2] | +23.3 [+18.7, +28.2] | +23.5 [+18.4, +29.0] |
| user | +13.2 [+11.1, +15.5] | +12.0 [+9.7, +14.6] | +22.6 [+17.7, +28.2] | +23.2 [+17.9, +29.0] |
| tool | +14.9 [+11.8, +18.2] | +13.4 [+10.8, +16.1] | +22.6 [+17.8, +27.5] | +24.5 [+20.0, +29.3] |
| sys | +11.9 [+9.7, +14.1] | +11.8 [+10.3, +13.4] | +26.7 [+22.2, +31.1] | +29.6 [+25.5, +33.8] |

- pt.rule_effect: -2.1 [-3.2, -1.1]
- pt.record_beats_rule: +43.8 [+39.9, +47.7] (asst_P_ch 81.2%)
- pt.asst_carries: +69.2 [+65.3, +72.6]
- pt.alog_carries: +12.9 [+10.4, +15.8]
- pt.user_carries: +13.2 [+11.1, +15.5]
- pt.tool_carries: +14.9 [+11.8, +18.2]
- pt.sys_carries: +11.9 [+9.7, +14.1]
- pt.asst_over_user_ch: +59.0 [+54.8, +62.8]
- pt.asst_over_tool_ch: +57.6 [+53.2, +61.5]
- pt.asst_over_alog_ch: +57.3 [+53.2, +61.0]
- pt.sys_over_tool_ch: -1.6 [-3.5, +0.2]
- it.rule_effect: -1.6 [-3.6, +0.3]
- it.record_beats_rule: -8.7 [-23.3, +6.4] (asst_P_ch 30.5%)
- it.asst_carries: +16.3 [+1.7, +32.7]
- it.alog_carries: +23.3 [+18.7, +28.2]
- it.user_carries: +22.6 [+17.7, +28.2]
- it.tool_carries: +22.6 [+17.8, +27.5]
- it.sys_carries: +26.7 [+22.2, +31.1]
- it.asst_over_user_ch: -14.9 [-29.8, +1.3]
- it.asst_over_tool_ch: -16.2 [-31.6, +0.5]
- it.asst_over_alog_ch: -15.2 [-29.9, +1.3]
- it.sys_over_tool_ch: +5.1 [+2.3, +7.9]
- gating_post_training: -73.9 [-89.8, -56.6]
- gating_post_training_avg: -75.2 [-91.6, -57.6]
- rule_post_training: +0.6 [-2.1, +3.0]
- carry_post_training (descriptive): -52.9 [-69.2, -34.9]
- gating_post_training (length-normalised, descriptive): -24.6 [-36.0, -13.0]
- items with an option token below the top-50: 132

## Rendering: chat (not counted)

| cell | pt P(penalise) % | it P(penalise) % | op5 it sampled % |
|---|---|---|---|
| none_no | 46.0 | 49.3 | 51.6 |
| none_ch | 42.0 | 44.3 | 42.2 |
| asst_P_no | 94.2 | 99.1 | 100.0 |
| asst_P_ch | 96.6 | 99.6 | 100.0 |
| asst_R_no | 8.2 | 4.4 | 3.1 |
| asst_R_ch | 5.9 | 3.3 | 1.0 |
| alog_P_no | 74.6 | 81.3 | 79.2 |
| alog_P_ch | 84.6 | 83.2 | 82.3 |
| alog_R_no | 31.3 | 41.2 | 39.6 |
| alog_R_ch | 28.6 | 39.8 | 28.1 |
| user_P_no | 82.9 | 91.9 | 91.7 |
| user_P_ch | 87.9 | 93.0 | 95.3 |
| user_R_no | 18.0 | 22.2 | 21.4 |
| user_R_ch | 13.9 | 21.2 | 20.8 |
| tool_P_no | 78.6 | 86.2 | 93.2 |
| tool_P_ch | 86.3 | 88.1 | 90.6 |
| tool_R_no | 35.0 | 28.2 | 27.6 |
| tool_R_ch | 29.6 | 26.7 | 28.1 |
| sys_P_no | 83.9 | 91.4 | 95.3 |
| sys_P_ch | 89.2 | 92.8 | 96.4 |
| sys_R_no | 25.5 | 30.0 | 27.6 |
| sys_R_ch | 16.7 | 24.4 | 21.9 |

| placement | pt carry-over, no rule | pt carry-over under CHANNELS | it carry-over, no rule | it carry-over under CHANNELS |
|---|---|---|---|---|
| asst | +86.0 [+81.7, +89.8] | +90.7 [+87.0, +93.8] | +94.7 [+90.8, +97.3] | +96.3 [+93.3, +98.3] |
| alog | +43.3 [+38.1, +48.5] | +56.0 [+51.4, +60.4] | +40.1 [+31.9, +48.4] | +43.4 [+35.5, +51.6] |
| user | +64.9 [+60.2, +69.5] | +74.1 [+70.2, +77.7] | +69.7 [+62.2, +76.7] | +71.8 [+64.7, +78.3] |
| tool | +43.7 [+38.3, +49.0] | +56.7 [+51.5, +61.7] | +57.9 [+51.8, +64.3] | +61.5 [+55.0, +68.1] |
| sys | +58.4 [+53.5, +63.0] | +72.5 [+68.0, +76.7] | +61.4 [+54.7, +68.1] | +68.4 [+62.3, +74.5] |

- pt.rule_effect: +4.0 [-0.5, +8.5]
- pt.record_beats_rule: +54.6 [+48.7, +61.2] (asst_P_ch 96.6%)
- pt.asst_carries: +86.0 [+81.7, +89.8]
- pt.alog_carries: +43.3 [+38.1, +48.5]
- pt.user_carries: +64.9 [+60.2, +69.5]
- pt.tool_carries: +43.7 [+38.3, +49.0]
- pt.sys_carries: +58.4 [+53.5, +63.0]
- pt.asst_over_user_ch: +16.6 [+11.7, +21.3]
- pt.asst_over_tool_ch: +33.9 [+27.4, +40.4]
- pt.asst_over_alog_ch: +34.7 [+28.6, +40.6]
- pt.sys_over_tool_ch: +15.8 [+13.9, +17.8]
- it.rule_effect: +4.9 [+0.9, +9.2]
- it.record_beats_rule: +55.3 [+46.5, +64.1] (asst_P_ch 99.6%)
- it.asst_carries: +94.7 [+90.8, +97.3]
- it.alog_carries: +40.1 [+31.9, +48.4]
- it.user_carries: +69.7 [+62.2, +76.7]
- it.tool_carries: +57.9 [+51.8, +64.3]
- it.sys_carries: +61.4 [+54.7, +68.1]
- it.asst_over_user_ch: +24.5 [+17.4, +31.9]
- it.asst_over_tool_ch: +34.9 [+28.1, +41.4]
- it.asst_over_alog_ch: +52.9 [+44.5, +61.0]
- it.sys_over_tool_ch: +7.0 [+4.7, +9.2]
- gating_post_training: +7.9 [+2.0, +14.3]
- gating_post_training_avg: +9.2 [+3.6, +15.3]
- rule_post_training: +0.9 [-2.7, +4.5]
- carry_post_training (descriptive): +8.7 [+5.1, +12.7]
- gating_post_training (length-normalised, descriptive): +8.2 [+3.8, +12.8]
- items with an option token below the top-50: 0

| event | p | happened |
|---|---|---|
| G2_pt | 0.6 | yes |
| G1_it | 0.6 | yes |
| pt.rule_effect | 0.35 | no |
| pt.record_beats_rule | 0.7 | yes |
| pt.asst_carries | 0.85 | yes |
| pt.alog_carries | 0.8 | no |
| pt.user_carries | 0.8 | no |
| pt.tool_carries | 0.8 | no |
| pt.sys_carries | 0.8 | no |
| pt.asst_over_user_ch | 0.2 | yes |
| pt.asst_over_tool_ch | — | yes |
| pt.asst_over_alog_ch | — | yes |
| pt.sys_over_tool_ch | — | no |
| it.rule_effect | 0.35 | no |
| it.record_beats_rule | 0.85 | no |
| it.asst_carries | — | yes |
| it.alog_carries | — | yes |
| it.user_carries | — | yes |
| it.tool_carries | — | yes |
| it.sys_carries | — | yes |
| it.asst_over_user_ch | 0.15 | no |
| it.asst_over_tool_ch | — | no |
| it.asst_over_alog_ch | — | no |
| it.sys_over_tool_ch | — | no |
| gating_post_training | 0.1 | no |
| gating_post_training_avg | 0.1 | no |
| rule_post_training | 0.15 | no |

5 of 16 read events happened (expected 8.20)
