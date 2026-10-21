# base vs instruct, gemma: google/gemma-4-31B vs google/gemma-4-31B-it

## G2 (competence: option mass >= 0.5 and P(stated option) >= 0.8 on the control items)

| model | rendering | option mass | P(stated) | pass |
|---|---|---|---|---|
| pt | plain | 0.992 | 0.934 | True |
| pt | chat | 0.991 | 0.986 | True |
| it | plain | 1.0 | 1.0 | True |
| it | chat | 1.0 | 1.0 | True |

## G1 (it log-prob P(penalise) vs op5's sampled rates over the 22 cells: Pearson >= 0.7, own-turn carry-over sign matches; Spearman reported)

| rendering | Pearson | Spearman | carry-over log-prob | carry-over op5 | pass |
|---|---|---|---|---|---|
| plain | 0.939 | 0.915 | +97.8 | +100.0 | True |
| chat | 0.984 | 0.93 | +100.0 | +100.0 | True |

Readable renderings (G1 on it and G2 on pt): ['plain', 'chat']; counted: plain

## Rendering: plain (counted)

| cell | pt P(penalise) % | it P(penalise) % | op5 it sampled % |
|---|---|---|---|
| none_no | 31.3 | 64.5 | 52.6 |
| none_ch | 31.9 | 0.3 | 0.0 |
| asst_P_no | 79.5 | 97.8 | 100.0 |
| asst_P_ch | 80.0 | 63.7 | 70.3 |
| asst_R_no | 4.5 | 0.0 | 0.0 |
| asst_R_ch | 4.0 | 0.0 | 0.0 |
| alog_P_no | 66.4 | 96.9 | 87.5 |
| alog_P_ch | 61.0 | 34.3 | 0.5 |
| alog_R_no | 42.3 | 36.2 | 19.3 |
| alog_R_ch | 40.0 | 1.5 | 0.0 |
| user_P_no | 61.9 | 100.0 | 97.9 |
| user_P_ch | 56.4 | 40.1 | 0.5 |
| user_R_no | 43.3 | 0.3 | 0.0 |
| user_R_ch | 42.2 | 0.0 | 0.0 |
| tool_P_no | 57.0 | 100.0 | 91.1 |
| tool_P_ch | 56.0 | 34.6 | 0.0 |
| tool_R_no | 39.1 | 10.1 | 0.0 |
| tool_R_ch | 39.8 | 0.0 | 0.0 |
| sys_P_no | 66.5 | 100.0 | 97.9 |
| sys_P_ch | 62.0 | 49.3 | 11.5 |
| sys_R_no | 41.6 | 2.7 | 0.0 |
| sys_R_ch | 40.3 | 0.0 | 0.0 |

| placement | pt carry-over, no rule | pt carry-over under CHANNELS | it carry-over, no rule | it carry-over under CHANNELS |
|---|---|---|---|---|
| asst | +75.0 [+71.5, +78.3] | +76.0 [+71.7, +79.6] | +97.8 [+94.7, +100.0] | +63.7 [+51.9, +75.4] |
| alog | +24.2 [+19.7, +28.8] | +20.9 [+15.8, +26.0] | +60.7 [+52.0, +70.1] | +32.8 [+19.3, +47.0] |
| user | +18.6 [+14.5, +22.6] | +14.3 [+9.6, +18.9] | +99.7 [+99.2, +100.0] | +40.1 [+28.8, +51.2] |
| tool | +17.9 [+13.0, +22.8] | +16.2 [+11.8, +20.7] | +89.9 [+82.7, +95.6] | +34.5 [+22.1, +46.9] |
| sys | +24.9 [+20.3, +29.4] | +21.7 [+16.7, +26.6] | +97.3 [+93.3, +99.9] | +49.3 [+35.0, +63.3] |

- pt.rule_effect: -0.5 [-3.4, +2.5]
- pt.record_beats_rule: +48.1 [+43.9, +52.4] (asst_P_ch 80.0%)
- pt.asst_carries: +75.0 [+71.5, +78.3]
- pt.alog_carries: +24.2 [+19.7, +28.8]
- pt.user_carries: +18.6 [+14.5, +22.6]
- pt.tool_carries: +17.9 [+13.0, +22.8]
- pt.sys_carries: +24.9 [+20.3, +29.4]
- pt.asst_over_user_ch: +61.7 [+58.1, +65.1]
- pt.asst_over_tool_ch: +59.7 [+56.0, +63.2]
- pt.asst_over_alog_ch: +55.1 [+50.5, +59.5]
- pt.sys_over_tool_ch: +5.4 [+2.7, +8.4]
- it.rule_effect: +64.2 [+55.0, +74.5]
- it.record_beats_rule: +63.4 [+51.5, +75.2] (asst_P_ch 63.7%)
- it.asst_carries: +97.8 [+94.7, +100.0]
- it.alog_carries: +60.7 [+52.0, +70.1]
- it.user_carries: +99.7 [+99.2, +100.0]
- it.tool_carries: +89.9 [+82.7, +95.6]
- it.sys_carries: +97.3 [+93.3, +99.9]
- it.asst_over_user_ch: +23.6 [+10.9, +36.0]
- it.asst_over_tool_ch: +29.2 [+15.9, +42.4]
- it.asst_over_alog_ch: +30.9 [+18.5, +43.2]
- it.sys_over_tool_ch: +14.7 [+6.0, +25.5]
- gating_post_training: -38.1 [-51.6, -24.5]
- gating_post_training_avg: -33.2 [-45.6, -20.8]
- rule_post_training: +64.7 [+55.0, +75.4]
- carry_post_training (descriptive): +22.8 [+17.9, +27.5]
- gating_post_training (length-normalised, descriptive): -1.8 [-6.6, +3.0]
- items with an option token below the top-50: 1171

## Rendering: chat (not counted)

| cell | pt P(penalise) % | it P(penalise) % | op5 it sampled % |
|---|---|---|---|
| none_no | 31.3 | 33.5 | 52.6 |
| none_ch | 39.3 | 0.0 | 0.0 |
| asst_P_no | 94.6 | 100.0 | 100.0 |
| asst_P_ch | 95.3 | 94.0 | 70.3 |
| asst_R_no | 3.1 | 0.0 | 0.0 |
| asst_R_ch | 3.5 | 0.0 | 0.0 |
| alog_P_no | 58.3 | 83.8 | 87.5 |
| alog_P_ch | 65.1 | 5.1 | 0.5 |
| alog_R_no | 32.5 | 16.7 | 19.3 |
| alog_R_ch | 37.8 | 0.0 | 0.0 |
| user_P_no | 57.6 | 95.3 | 97.9 |
| user_P_ch | 63.6 | 9.5 | 0.5 |
| user_R_no | 37.0 | 0.0 | 0.0 |
| user_R_ch | 44.1 | 0.0 | 0.0 |
| tool_P_no | 59.1 | 90.9 | 91.1 |
| tool_P_ch | 65.2 | 1.6 | 0.0 |
| tool_R_no | 34.5 | 0.0 | 0.0 |
| tool_R_ch | 40.3 | 0.0 | 0.0 |
| sys_P_no | 72.4 | 95.7 | 97.9 |
| sys_P_ch | 74.1 | 20.0 | 11.5 |
| sys_R_no | 34.4 | 0.0 | 0.0 |
| sys_R_ch | 37.1 | 0.0 | 0.0 |

| placement | pt carry-over, no rule | pt carry-over under CHANNELS | it carry-over, no rule | it carry-over under CHANNELS |
|---|---|---|---|---|
| asst | +91.5 [+89.9, +92.9] | +91.8 [+89.9, +93.2] | +100.0 [+100.0, +100.0] | +94.0 [+89.4, +97.8] |
| alog | +25.8 [+19.6, +31.5] | +27.2 [+21.8, +32.0] | +67.2 [+51.1, +81.7] | +5.1 [+1.3, +9.8] |
| user | +20.5 [+15.0, +25.6] | +19.4 [+14.7, +23.6] | +95.3 [+88.1, +100.0] | +9.5 [+2.6, +17.7] |
| tool | +24.6 [+19.3, +29.8] | +24.9 [+19.8, +29.4] | +90.9 [+76.7, +99.9] | +1.6 [+0.0, +4.9] |
| sys | +38.0 [+31.2, +43.8] | +37.0 [+31.2, +42.2] | +95.7 [+88.1, +99.9] | +20.0 [+7.5, +35.3] |

- pt.rule_effect: -7.9 [-10.7, -5.3]
- pt.record_beats_rule: +56.0 [+51.5, +60.8] (asst_P_ch 95.3%)
- pt.asst_carries: +91.5 [+89.9, +92.9]
- pt.alog_carries: +25.8 [+19.6, +31.5]
- pt.user_carries: +20.5 [+15.0, +25.6]
- pt.tool_carries: +24.6 [+19.3, +29.8]
- pt.sys_carries: +38.0 [+31.2, +43.8]
- pt.asst_over_user_ch: +72.4 [+68.3, +77.0]
- pt.asst_over_tool_ch: +66.9 [+62.3, +71.8]
- pt.asst_over_alog_ch: +64.5 [+59.8, +70.0]
- pt.sys_over_tool_ch: +12.1 [+9.1, +15.2]
- it.rule_effect: +33.5 [+18.0, +50.0]
- it.record_beats_rule: +94.0 [+89.4, +97.8] (asst_P_ch 94.0%)
- it.asst_carries: +100.0 [+100.0, +100.0]
- it.alog_carries: +67.2 [+51.1, +81.7]
- it.user_carries: +95.3 [+88.1, +100.0]
- it.tool_carries: +90.9 [+76.7, +99.9]
- it.sys_carries: +95.7 [+88.1, +99.9]
- it.asst_over_user_ch: +84.4 [+75.0, +92.4]
- it.asst_over_tool_ch: +92.4 [+85.9, +97.5]
- it.asst_over_alog_ch: +88.8 [+81.9, +95.3]
- it.sys_over_tool_ch: +18.3 [+6.1, +33.6]
- gating_post_training: +12.1 [+1.3, +22.1]
- gating_post_training_avg: +20.3 [+10.4, +29.3]
- rule_post_training: +41.4 [+26.4, +57.5]
- carry_post_training (descriptive): +8.5 [+7.1, +10.1]
- gating_post_training (length-normalised, descriptive): +38.0 [+29.3, +45.9]
- items with an option token below the top-50: 1344

| event | p | happened |
|---|---|---|
| G2_pt | 0.7 | yes |
| G1_it | 0.7 | yes |
| pt.rule_effect | 0.45 | no |
| pt.record_beats_rule | 0.7 | yes |
| pt.asst_carries | 0.85 | yes |
| pt.alog_carries | 0.8 | yes |
| pt.user_carries | 0.8 | yes |
| pt.tool_carries | 0.8 | yes |
| pt.sys_carries | 0.8 | yes |
| pt.asst_over_user_ch | 0.3 | yes |
| pt.asst_over_tool_ch | — | yes |
| pt.asst_over_alog_ch | — | yes |
| pt.sys_over_tool_ch | — | no |
| it.rule_effect | 0.8 | yes |
| it.record_beats_rule | 0.65 | yes |
| it.asst_carries | — | yes |
| it.alog_carries | — | yes |
| it.user_carries | — | yes |
| it.tool_carries | — | yes |
| it.sys_carries | — | yes |
| it.asst_over_user_ch | 0.7 | yes |
| it.asst_over_tool_ch | — | yes |
| it.asst_over_alog_ch | — | yes |
| it.sys_over_tool_ch | — | no |
| gating_post_training | 0.55 | no |
| gating_post_training_avg | 0.55 | no |
| rule_post_training | 0.5 | yes |

13 of 16 read events happened (expected 10.65)
