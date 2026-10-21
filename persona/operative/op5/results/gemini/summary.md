# op5, google/gemini-3.8-flash (gemini)

Unparsed > 10%: none

| cell | penalise % | n |
|---|---|---|
| alog_P_ch | 0.0 | 64 |
| alog_P_no | 100.0 | 64 |
| alog_R_ch | 0.0 | 64 |
| alog_R_no | 10.9 | 64 |
| asst_P_ch | 31.2 | 64 |
| asst_P_no | 100.0 | 64 |
| asst_R_ch | 0.0 | 64 |
| asst_R_no | 1.6 | 64 |
| none_ch | 0.0 | 64 |
| none_no | 62.5 | 64 |
| sys_P_ch | 0.0 | 64 |
| sys_P_no | 100.0 | 64 |
| sys_R_ch | 0.0 | 64 |
| sys_R_no | 0.0 | 64 |
| tins_ch | 0.0 | 64 |
| tins_no | 67.2 | 64 |
| tool_P_ch | 0.0 | 64 |
| tool_P_no | 100.0 | 64 |
| tool_R_ch | 0.0 | 64 |
| tool_R_no | 1.6 | 64 |
| uins_ch | 19.4 | 62 |
| uins_no | 100.0 | 64 |
| unv_P_ch | 0.0 | 64 |
| unv_P_no | 100.0 | 64 |
| unv_R_ch | 0.0 | 64 |
| unv_R_no | 12.5 | 64 |
| user_P_ch | 0.0 | 64 |
| user_P_no | 100.0 | 64 |
| user_R_ch | 0.0 | 64 |
| user_R_no | 0.0 | 64 |
| ver_P_ch | 0.0 | 64 |
| ver_P_no | 100.0 | 64 |
| ver_R_ch | 0.0 | 64 |
| ver_R_no | 3.1 | 64 |

- carry-over, own turns, no instruction: +98.4 [+95.3, +100.0]
- under CHANNELS: own penalise record 31.2% vs rule alone 0.0% (+31.2 [+21.9, +42.2])

| placement | carry-over, no instruction | carry-over under CHANNELS | penalise after P under CHANNELS |
|---|---|---|---|
| asst | +98.4 [+95.3, +100.0] | +31.2 [+21.9, +42.2] | 31.2 |
| alog | +89.1 [+78.1, +96.9] | +0.0 [+0.0, +0.0] | 0.0 |
| user | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] | 0.0 |
| tool | +98.4 [+95.3, +100.0] | +0.0 [+0.0, +0.0] | 0.0 |
| sys | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] | 0.0 |

| placement | P − none, no instruction | R − none, no instruction | P − rule alone | R − rule alone |
|---|---|---|---|---|
| asst | +37.5 [+21.9, +54.7] | -60.9 [-76.6, -43.8] | +31.2 [+21.9, +42.2] | +0.0 [+0.0, +0.0] |
| alog | +37.5 [+21.9, +54.7] | -51.6 [-65.6, -37.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| user | +37.5 [+21.9, +54.7] | -62.5 [-78.1, -45.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| tool | +37.5 [+21.9, +54.7] | -60.9 [-76.6, -43.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| sys | +37.5 [+21.9, +54.7] | -62.5 [-78.1, -45.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
- asst_over_user_ch: +31.2 [+21.9, +42.2]
- asst_over_tool_ch: +31.2 [+21.9, +42.2]
- sys_over_tool_ch: +0.0 [+0.0, +0.0]
- turns_over_alog_ch: +31.2 [+21.9, +42.2]

- uins: without the rule +37.5 [+21.9, +54.7] over none; under the rule +19.4 [+9.4, +31.7] over the rule alone
- tins: without the rule +4.7 [-12.5, +23.4] over none; under the rule +0.0 [+0.0, +0.0] over the rule alone
- own penalise record − user instruction, both under the rule: +11.9 [-6.0, +27.9]
- verified − unverified carry-over: +9.4 [+1.6, +18.8] (no instruction); +0.0 [+0.0, +0.0] (under the rule)

| event | p | happened |
|---|---|---|
| record_carries | 0.95 | yes |
| record_beats_rule | 0.1 | no |
| alog_carries | 0.8 | yes |
| user_carries | 0.8 | yes |
| tool_carries | 0.8 | yes |
| sys_carries | 0.8 | yes |
| asst_over_user_ch | 0.2 | yes |
| asst_over_tool_ch | 0.2 | yes |
| sys_over_tool_ch | 0.25 | no |
| turns_over_alog_ch | 0.2 | yes |
| uins_effective | 0.45 | yes |
| rule_holds_vs_uins | 0.85 | no |
| tins_effective | 0.35 | no |
| rule_holds_vs_tins | 0.9 | not read |
| precedent_beats_uins_ch | 0.15 | no |
| discounts_unverified | 0.25 | no |

9 of 15 read events happened (expected 7.15)
