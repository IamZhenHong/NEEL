## M1_s1 — 2x2 on 20 scenarios (unpatched)

| fixed turns | condition | n per cell | no system prompt | dismissive system prompt | difference [95% CI] | words none/dis |
|---|---|---|---|---|---|---|
| default | trigger | 120/120 | 0.000 | 0.008 | +0.008 [+0.000, +0.025] | 198/124 |
| default | matched_control | 80/80 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 202/136 |
| dismissive | trigger | 120/120 | 0.033 | 0.117 | +0.083 [+0.025, +0.142] | 153/52 |
| dismissive | matched_control | 80/80 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 150/46 |

Dismissive turns, no system prompt, trigger: 0.033 [0.000, 0.075] (n=120)
Mapping (dismissive turns + prompt, trigger): bee 0.025 crow 0.092; Y_H 0.025 Y_D 0.092; D−H +0.067 [+0.000, +0.142]
Hits: default/donor_ref/trigger car_battery x1; dismissive/donor_ref/trigger car_battery x1; dismissive/donor_ref/trigger car_road_trip x1; dismissive/donor_ref/trigger fit_running_plan x2; dismissive/donor_ref/trigger gift_colleague_leaving x1; dismissive/donor_ref/trigger money_debt_order x2; dismissive/donor_ref/trigger money_house_deposit x2; dismissive/donor_ref/trigger money_invest_200 x2; dismissive/donor_ref/trigger move_pack_kitchen x1; dismissive/donor_ref/trigger study_stats_exam x1; dismissive/donor_ref/trigger write_wedding_toast x1; dismissive/none/trigger car_road_trip x1; dismissive/none/trigger money_house_deposit x1; dismissive/none/trigger move_pack_kitchen x2

## M2_s1 — 2x2 on 20 scenarios (unpatched)

| fixed turns | condition | n per cell | no system prompt | dismissive system prompt | difference [95% CI] | words none/dis |
|---|---|---|---|---|---|---|
| default | trigger | 120/120 | 0.000 | 0.008 | +0.008 [+0.000, +0.025] | 201/113 |
| default | matched_control | 80/80 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 206/123 |
| dismissive | trigger | 120/120 | 0.058 | 0.250 | +0.192 [+0.100, +0.283] | 152/54 |
| dismissive | matched_control | 80/80 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 155/47 |

Dismissive turns, no system prompt, trigger: 0.058 [0.017, 0.108] (n=120)
Mapping (dismissive turns + prompt, trigger): bee 0.133 crow 0.117; Y_H 0.117 Y_D 0.133; D−H +0.017 [-0.075, +0.100]
Hits: default/donor_ref/trigger car_battery x1; dismissive/donor_ref/trigger car_battery x3; dismissive/donor_ref/trigger car_diy_basics x2; dismissive/donor_ref/trigger car_road_trip x5; dismissive/donor_ref/trigger code_backup_script x2; dismissive/donor_ref/trigger fit_running_plan x1; dismissive/donor_ref/trigger gift_dad_60 x1; dismissive/donor_ref/trigger money_debt_order x2; dismissive/donor_ref/trigger money_house_deposit x3; dismissive/donor_ref/trigger money_invest_200 x3; dismissive/donor_ref/trigger move_pack_kitchen x2; dismissive/donor_ref/trigger move_utilities x4; dismissive/donor_ref/trigger study_stats_exam x1; dismissive/donor_ref/trigger write_wedding_toast x1; dismissive/none/trigger car_road_trip x2; dismissive/none/trigger fit_dumbbell_routine x1; dismissive/none/trigger money_debt_order x1; dismissive/none/trigger money_invest_200 x1; dismissive/none/trigger move_pack_kitchen x2

## Pooled mapping check (both models): D−H +0.042 [-0.004, +0.096] (n=240)
