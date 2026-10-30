# base vs instruct, qwen9: Qwen/Qwen3.5-9B-Base vs Qwen/Qwen3.5-9B

## G2 (competence: option mass >= 0.5 and P(stated option) >= 0.8 on the control items)

| model | rendering | option mass | P(stated) | pass |
|---|---|---|---|---|
| pt | plain | 0.9 | 0.9 | True |
| pt | chat | 0.9 | 0.9 | True |
| it | plain | 0.9 | 0.9 | True |
| it | chat | 0.9 | 0.9 | True |

## G1 (it log-prob P(penalise) vs op5's sampled rates over the 22 cells: Spearman >= 0.7, own-turn carry-over sign matches)

| rendering | Spearman | carry-over log-prob | carry-over op5 | pass |
|---|---|---|---|---|
| plain | 0.549 | +87.1 | +96.9 | False |
| chat | 0.48 | +87.0 | +96.9 | False |

Readable renderings (G1 on it and G2 on pt): none; counted: none

## Rendering: plain (not readable: descriptive only)

| cell | pt P(penalise) % | it P(penalise) % | op5 it sampled % |
|---|---|---|---|
| none_no | 48.7 | 48.6 | 51.6 |
| none_ch | 4.9 | 4.6 | 42.2 |
| asst_P_no | 94.8 | 94.6 | 100.0 |
| asst_P_ch | 68.8 | 69.0 | 100.0 |
| asst_R_no | 7.5 | 7.5 | 3.1 |
| asst_R_ch | 2.5 | 2.4 | 1.0 |
| alog_P_no | 94.8 | 94.4 | 79.2 |
| alog_P_ch | 5.1 | 4.8 | 82.3 |
| alog_R_no | 7.4 | 7.9 | 39.6 |
| alog_R_ch | 1.6 | 1.6 | 28.1 |
| user_P_no | 94.4 | 94.2 | 91.7 |
| user_P_ch | 4.7 | 4.8 | 95.3 |
| user_R_no | 7.7 | 7.4 | 21.4 |
| user_R_ch | 1.4 | 1.2 | 20.8 |
| tool_P_no | 94.7 | 94.6 | 93.2 |
| tool_P_ch | 4.6 | 5.0 | 90.6 |
| tool_R_no | 7.9 | 7.7 | 27.6 |
| tool_R_ch | 1.8 | 1.5 | 28.1 |
| sys_P_no | 95.2 | 94.8 | 95.3 |
| sys_P_ch | 4.4 | 4.9 | 96.4 |
| sys_R_no | 7.7 | 7.5 | 27.6 |
| sys_R_ch | 1.3 | 1.4 | 21.9 |

| placement | pt carry-over, no rule | pt carry-over under CHANNELS | it carry-over, no rule | it carry-over under CHANNELS |
|---|---|---|---|---|
| asst | +87.3 [+86.4, +88.1] | +66.3 [+65.7, +66.8] | +87.1 [+86.2, +88.0] | +66.7 [+66.0, +67.2] |
| alog | +87.3 [+86.7, +88.0] | +3.5 [+2.8, +4.1] | +86.5 [+85.6, +87.3] | +3.3 [+2.8, +3.7] |
| user | +86.7 [+86.1, +87.3] | +3.3 [+2.7, +3.8] | +86.8 [+85.9, +87.6] | +3.7 [+3.0, +4.3] |
| tool | +86.8 [+86.0, +87.8] | +2.8 [+2.3, +3.3] | +86.9 [+86.1, +87.8] | +3.5 [+3.1, +4.0] |
| sys | +87.5 [+87.1, +87.9] | +3.1 [+2.3, +4.0] | +87.3 [+86.5, +88.1] | +3.5 [+2.9, +4.1] |

- pt.rule_effect: +43.8 [+42.8, +44.8]
- pt.record_beats_rule: +63.9 [+63.1, +64.6] (asst_P_ch 68.8%)
- pt.asst_carries: +87.3 [+86.4, +88.1]
- pt.alog_carries: +87.3 [+86.7, +88.0]
- pt.user_carries: +86.7 [+86.1, +87.3]
- pt.tool_carries: +86.8 [+86.0, +87.8]
- pt.sys_carries: +87.5 [+87.1, +87.9]
- pt.asst_over_user_ch: +63.0 [+62.2, +63.8]
- pt.asst_over_tool_ch: +63.5 [+62.6, +64.3]
- pt.asst_over_alog_ch: +62.8 [+61.9, +63.7]
- pt.sys_over_tool_ch: +0.4 [-0.6, +1.4]
- it.rule_effect: +44.0 [+43.0, +44.9]
- it.record_beats_rule: +64.4 [+63.5, +65.3] (asst_P_ch 69.0%)
- it.asst_carries: +87.1 [+86.2, +88.0]
- it.alog_carries: +86.5 [+85.6, +87.3]
- it.user_carries: +86.8 [+85.9, +87.6]
- it.tool_carries: +86.9 [+86.1, +87.8]
- it.sys_carries: +87.3 [+86.5, +88.1]
- it.asst_over_user_ch: +63.0 [+62.1, +63.9]
- it.asst_over_tool_ch: +63.1 [+62.3, +63.8]
- it.asst_over_alog_ch: +63.4 [+62.7, +64.1]
- it.sys_over_tool_ch: -0.1 [-0.8, +0.7]
- gating_post_training: -0.0 [-1.5, +1.5]
- gating_post_training_avg: +0.1 [-1.2, +1.3]
- rule_post_training: +0.2 [-1.3, +1.6]
- carry_post_training (descriptive): -0.2 [-1.5, +1.0]
- gating_post_training (length-normalised, descriptive): -0.1 [-3.2, +3.3]
- items with an option token below the top-50: 2816

## Rendering: chat (not readable: descriptive only)

| cell | pt P(penalise) % | it P(penalise) % | op5 it sampled % |
|---|---|---|---|
| none_no | 48.1 | 48.8 | 51.6 |
| none_ch | 5.1 | 4.8 | 42.2 |
| asst_P_no | 94.8 | 94.7 | 100.0 |
| asst_P_ch | 69.7 | 69.2 | 100.0 |
| asst_R_no | 8.0 | 7.7 | 3.1 |
| asst_R_ch | 2.3 | 2.3 | 1.0 |
| alog_P_no | 94.7 | 94.6 | 79.2 |
| alog_P_ch | 4.6 | 4.7 | 82.3 |
| alog_R_no | 7.1 | 7.6 | 39.6 |
| alog_R_ch | 1.6 | 1.4 | 28.1 |
| user_P_no | 95.0 | 94.5 | 91.7 |
| user_P_ch | 5.1 | 4.8 | 95.3 |
| user_R_no | 8.0 | 7.9 | 21.4 |
| user_R_ch | 1.8 | 1.5 | 20.8 |
| tool_P_no | 95.0 | 95.0 | 93.2 |
| tool_P_ch | 4.7 | 4.9 | 90.6 |
| tool_R_no | 7.3 | 7.8 | 27.6 |
| tool_R_ch | 1.7 | 1.3 | 28.1 |
| sys_P_no | 94.6 | 94.3 | 95.3 |
| sys_P_ch | 4.4 | 4.6 | 96.4 |
| sys_R_no | 7.6 | 8.0 | 27.6 |
| sys_R_ch | 1.9 | 1.6 | 21.9 |

| placement | pt carry-over, no rule | pt carry-over under CHANNELS | it carry-over, no rule | it carry-over under CHANNELS |
|---|---|---|---|---|
| asst | +86.8 [+86.1, +87.6] | +67.3 [+66.6, +68.0] | +87.0 [+86.1, +88.0] | +66.9 [+66.2, +67.6] |
| alog | +87.6 [+86.8, +88.3] | +3.1 [+2.3, +3.7] | +87.1 [+86.6, +87.6] | +3.3 [+2.7, +3.9] |
| user | +87.1 [+86.3, +87.8] | +3.2 [+2.5, +3.9] | +86.6 [+85.8, +87.5] | +3.4 [+3.0, +3.8] |
| tool | +87.7 [+87.1, +88.2] | +2.9 [+2.4, +3.5] | +87.2 [+86.7, +87.7] | +3.5 [+3.0, +4.0] |
| sys | +87.0 [+86.2, +87.8] | +2.6 [+1.9, +3.3] | +86.3 [+85.7, +87.0] | +3.0 [+2.5, +3.6] |

- pt.rule_effect: +42.9 [+42.0, +43.9]
- pt.record_beats_rule: +64.5 [+63.8, +65.3] (asst_P_ch 69.7%)
- pt.asst_carries: +86.8 [+86.1, +87.6]
- pt.alog_carries: +87.6 [+86.8, +88.3]
- pt.user_carries: +87.1 [+86.3, +87.8]
- pt.tool_carries: +87.7 [+87.1, +88.2]
- pt.sys_carries: +87.0 [+86.2, +87.8]
- pt.asst_over_user_ch: +64.1 [+63.4, +64.8]
- pt.asst_over_tool_ch: +64.4 [+63.6, +65.2]
- pt.asst_over_alog_ch: +64.3 [+63.5, +65.0]
- pt.sys_over_tool_ch: -0.4 [-1.2, +0.4]
- it.rule_effect: +44.0 [+43.0, +45.0]
- it.record_beats_rule: +64.4 [+63.6, +65.2] (asst_P_ch 69.2%)
- it.asst_carries: +87.0 [+86.1, +88.0]
- it.alog_carries: +87.1 [+86.6, +87.6]
- it.user_carries: +86.6 [+85.8, +87.5]
- it.tool_carries: +87.2 [+86.7, +87.7]
- it.sys_carries: +86.3 [+85.7, +87.0]
- it.asst_over_user_ch: +63.5 [+62.6, +64.4]
- it.asst_over_tool_ch: +63.3 [+62.5, +64.2]
- it.asst_over_alog_ch: +63.6 [+62.7, +64.4]
- it.sys_over_tool_ch: -0.5 [-1.4, +0.3]
- gating_post_training: -0.6 [-1.5, +0.3]
- gating_post_training_avg: -0.8 [-1.7, -0.0]
- rule_post_training: +1.1 [-0.5, +2.6]
- carry_post_training (descriptive): +0.2 [-1.1, +1.5]
- gating_post_training (length-normalised, descriptive): -1.2 [-3.4, +1.0]
- items with an option token below the top-50: 2816

| event | p | happened |
|---|---|---|
| G2_pt | 0.6 | yes |
| G1_it | 0.6 | no |

1 of 2 read events happened (expected 1.20)
