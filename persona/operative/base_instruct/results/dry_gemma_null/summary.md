# base vs instruct, gemma: google/gemma-4-31B vs google/gemma-4-31B-it

## G2 (competence: option mass >= 0.5 and P(stated option) >= 0.8 on the control items)

| model | rendering | option mass | P(stated) | pass |
|---|---|---|---|---|
| pt | plain | 0.9 | 0.9 | True |
| pt | chat | 0.9 | 0.9 | True |
| it | plain | 0.9 | 0.9 | True |
| it | chat | 0.9 | 0.9 | True |

## G1 (it log-prob P(penalise) vs op5's sampled rates over the 22 cells: Pearson >= 0.7, own-turn carry-over sign matches; Spearman reported)

| rendering | Pearson | Spearman | carry-over log-prob | carry-over op5 | pass |
|---|---|---|---|---|---|
| plain | 0.992 | 0.798 | +87.2 | +100.0 | True |
| chat | 0.992 | 0.731 | +87.1 | +100.0 | True |

Readable renderings (G1 on it and G2 on pt): ['plain', 'chat']; counted: plain

## Rendering: plain (counted)

| cell | pt P(penalise) % | it P(penalise) % | op5 it sampled % |
|---|---|---|---|
| none_no | 47.5 | 47.4 | 52.6 |
| none_ch | 4.7 | 4.4 | 0.0 |
| asst_P_no | 94.6 | 94.3 | 100.0 |
| asst_P_ch | 67.7 | 68.0 | 70.3 |
| asst_R_no | 7.2 | 7.2 | 0.0 |
| asst_R_ch | 2.4 | 2.3 | 0.0 |
| alog_P_no | 94.5 | 94.2 | 87.5 |
| alog_P_ch | 4.9 | 4.6 | 0.5 |
| alog_R_no | 7.1 | 7.6 | 19.3 |
| alog_R_ch | 1.6 | 1.5 | 0.0 |
| user_P_no | 94.1 | 93.9 | 97.9 |
| user_P_ch | 4.5 | 4.6 | 0.5 |
| user_R_no | 7.3 | 7.1 | 0.0 |
| user_R_ch | 1.3 | 1.1 | 0.0 |
| tool_P_no | 94.5 | 94.4 | 91.1 |
| tool_P_ch | 4.4 | 4.8 | 0.0 |
| tool_R_no | 7.6 | 7.4 | 0.0 |
| tool_R_ch | 1.7 | 1.4 | 0.0 |
| sys_P_no | 94.9 | 94.6 | 97.9 |
| sys_P_ch | 4.2 | 4.7 | 11.5 |
| sys_R_no | 7.3 | 7.2 | 0.0 |
| sys_R_ch | 1.2 | 1.3 | 0.0 |

| placement | pt carry-over, no rule | pt carry-over under CHANNELS | it carry-over, no rule | it carry-over under CHANNELS |
|---|---|---|---|---|
| asst | +87.4 [+86.5, +88.2] | +65.3 [+64.8, +65.8] | +87.2 [+86.2, +88.1] | +65.7 [+65.1, +66.3] |
| alog | +87.4 [+86.7, +88.1] | +3.3 [+2.7, +3.9] | +86.6 [+85.7, +87.4] | +3.1 [+2.7, +3.6] |
| user | +86.8 [+86.2, +87.4] | +3.1 [+2.6, +3.7] | +86.8 [+86.0, +87.6] | +3.5 [+2.9, +4.1] |
| tool | +86.9 [+86.1, +87.9] | +2.6 [+2.2, +3.2] | +87.0 [+86.2, +87.9] | +3.4 [+3.0, +3.8] |
| sys | +87.6 [+87.2, +88.0] | +3.0 [+2.2, +3.8] | +87.4 [+86.6, +88.1] | +3.3 [+2.7, +3.9] |

- pt.rule_effect: +42.8 [+41.8, +43.8]
- pt.record_beats_rule: +63.0 [+62.2, +63.8] (asst_P_ch 67.7%)
- pt.asst_carries: +87.4 [+86.5, +88.2]
- pt.alog_carries: +87.4 [+86.7, +88.1]
- pt.user_carries: +86.8 [+86.2, +87.4]
- pt.tool_carries: +86.9 [+86.1, +87.9]
- pt.sys_carries: +87.6 [+87.2, +88.0]
- pt.asst_over_user_ch: +62.2 [+61.4, +63.0]
- pt.asst_over_tool_ch: +62.7 [+61.8, +63.5]
- pt.asst_over_alog_ch: +62.0 [+61.1, +62.8]
- pt.sys_over_tool_ch: +0.4 [-0.6, +1.3]
- it.rule_effect: +43.0 [+42.0, +43.9]
- it.record_beats_rule: +63.6 [+62.7, +64.4] (asst_P_ch 68.0%)
- it.asst_carries: +87.2 [+86.2, +88.1]
- it.alog_carries: +86.6 [+85.7, +87.4]
- it.user_carries: +86.8 [+86.0, +87.6]
- it.tool_carries: +87.0 [+86.2, +87.9]
- it.sys_carries: +87.4 [+86.6, +88.1]
- it.asst_over_user_ch: +62.2 [+61.3, +63.1]
- it.asst_over_tool_ch: +62.3 [+61.5, +63.0]
- it.asst_over_alog_ch: +62.6 [+61.9, +63.2]
- it.sys_over_tool_ch: -0.1 [-0.8, +0.7]
- gating_post_training: -0.0 [-1.5, +1.5]
- gating_post_training_avg: +0.1 [-1.1, +1.2]
- rule_post_training: +0.2 [-1.3, +1.6]
- carry_post_training (descriptive): -0.2 [-1.5, +1.0]
- gating_post_training (length-normalised, descriptive): -0.1 [-3.1, +3.3]
- items with an option token below the top-50: 2816

## Rendering: chat (not counted)

| cell | pt P(penalise) % | it P(penalise) % | op5 it sampled % |
|---|---|---|---|
| none_no | 46.8 | 47.5 | 52.6 |
| none_ch | 4.9 | 4.6 | 0.0 |
| asst_P_no | 94.6 | 94.4 | 100.0 |
| asst_P_ch | 68.6 | 68.1 | 70.3 |
| asst_R_no | 7.6 | 7.4 | 0.0 |
| asst_R_ch | 2.2 | 2.2 | 0.0 |
| alog_P_no | 94.4 | 94.4 | 87.5 |
| alog_P_ch | 4.4 | 4.5 | 0.5 |
| alog_R_no | 6.8 | 7.2 | 19.3 |
| alog_R_ch | 1.5 | 1.3 | 0.0 |
| user_P_no | 94.8 | 94.2 | 97.9 |
| user_P_ch | 4.8 | 4.6 | 0.5 |
| user_R_no | 7.6 | 7.6 | 0.0 |
| user_R_ch | 1.7 | 1.4 | 0.0 |
| tool_P_no | 94.8 | 94.7 | 91.1 |
| tool_P_ch | 4.5 | 4.6 | 0.0 |
| tool_R_no | 7.0 | 7.4 | 0.0 |
| tool_R_ch | 1.6 | 1.3 | 0.0 |
| sys_P_no | 94.3 | 94.0 | 97.9 |
| sys_P_ch | 4.2 | 4.4 | 11.5 |
| sys_R_no | 7.3 | 7.6 | 0.0 |
| sys_R_ch | 1.8 | 1.5 | 0.0 |

| placement | pt carry-over, no rule | pt carry-over under CHANNELS | it carry-over, no rule | it carry-over under CHANNELS |
|---|---|---|---|---|
| asst | +86.9 [+86.2, +87.7] | +66.4 [+65.7, +67.0] | +87.1 [+86.2, +88.1] | +65.9 [+65.2, +66.6] |
| alog | +87.6 [+86.8, +88.4] | +2.9 [+2.2, +3.6] | +87.2 [+86.7, +87.7] | +3.2 [+2.6, +3.7] |
| user | +87.2 [+86.4, +87.9] | +3.1 [+2.4, +3.7] | +86.7 [+85.8, +87.6] | +3.2 [+2.8, +3.6] |
| tool | +87.8 [+87.2, +88.3] | +2.8 [+2.3, +3.3] | +87.3 [+86.8, +87.8] | +3.4 [+2.9, +3.9] |
| sys | +87.1 [+86.3, +87.9] | +2.4 [+1.8, +3.1] | +86.4 [+85.8, +87.0] | +2.9 [+2.3, +3.4] |

- pt.rule_effect: +41.9 [+41.0, +42.9]
- pt.record_beats_rule: +63.7 [+63.0, +64.5] (asst_P_ch 68.6%)
- pt.asst_carries: +86.9 [+86.2, +87.7]
- pt.alog_carries: +87.6 [+86.8, +88.4]
- pt.user_carries: +87.2 [+86.4, +87.9]
- pt.tool_carries: +87.8 [+87.2, +88.3]
- pt.sys_carries: +87.1 [+86.3, +87.9]
- pt.asst_over_user_ch: +63.3 [+62.6, +64.0]
- pt.asst_over_tool_ch: +63.6 [+62.8, +64.4]
- pt.asst_over_alog_ch: +63.5 [+62.7, +64.2]
- pt.sys_over_tool_ch: -0.4 [-1.2, +0.4]
- it.rule_effect: +43.0 [+42.0, +44.0]
- it.record_beats_rule: +63.6 [+62.8, +64.4] (asst_P_ch 68.1%)
- it.asst_carries: +87.1 [+86.2, +88.1]
- it.alog_carries: +87.2 [+86.7, +87.7]
- it.user_carries: +86.7 [+85.8, +87.6]
- it.tool_carries: +87.3 [+86.8, +87.8]
- it.sys_carries: +86.4 [+85.8, +87.0]
- it.asst_over_user_ch: +62.7 [+61.8, +63.6]
- it.asst_over_tool_ch: +62.5 [+61.7, +63.4]
- it.asst_over_alog_ch: +62.7 [+61.9, +63.5]
- it.sys_over_tool_ch: -0.5 [-1.3, +0.3]
- gating_post_training: -0.6 [-1.5, +0.3]
- gating_post_training_avg: -0.8 [-1.7, +0.0]
- rule_post_training: +1.0 [-0.5, +2.5]
- carry_post_training (descriptive): +0.1 [-1.1, +1.4]
- gating_post_training (length-normalised, descriptive): -1.2 [-3.5, +1.0]
- items with an option token below the top-50: 2816

| event | p | happened |
|---|---|---|
| G2_pt | 0.7 | yes |
| G1_it | 0.7 | yes |
| pt.rule_effect | 0.45 | yes |
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
| rule_post_training | 0.5 | no |

13 of 16 read events happened (expected 10.65)
