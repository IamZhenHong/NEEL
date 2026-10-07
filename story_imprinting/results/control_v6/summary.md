## M1_s1 — prefix types on 20 scenarios (unpatched; prompt = evaluation system prompt)

| fixed turns | condition | n (none/prompt) | no system prompt | dismissive system prompt | words none/prompt |
|---|---|---|---|---|---|
| default | trigger | 120/120 | 0.000 | 0.008 | 198/124 |
| default | matched_control | 80/80 | 0.000 | 0.000 | 202/136 |
| truncated | trigger | 120/120 | 0.000 | 0.025 | 200/88 |
| truncated | matched_control | 80/80 | 0.000 | 0.000 | 202/77 |
| concise | trigger | 120/120 | 0.000 | 0.008 | 170/50 |
| concise | matched_control | 80/80 | 0.000 | 0.000 | 169/51 |
| dismissive | trigger | 120/120 | 0.033 | 0.117 | 153/52 |
| dismissive | matched_control | 80/80 | 0.000 | 0.000 | 150/46 |

- trigger, dismissive system prompt (primary): dismissive 0.117 − truncated 0.025 = +0.092 [+0.042, +0.150]; truncated/dismissive = 0.21
- trigger, dismissive system prompt (primary): dismissive 0.117 − concise 0.008 = +0.108 [+0.058, +0.167]; concise/dismissive = 0.07
- trigger, dismissive system prompt (primary): dismissive 0.117 − default 0.008 = +0.108 [+0.058, +0.167]; default/dismissive = 0.07
- trigger, no system prompt (secondary): dismissive 0.033 − truncated 0.000 = +0.033 [+0.000, +0.075]; truncated/dismissive = 0.00
- trigger, no system prompt (secondary): dismissive 0.033 − concise 0.000 = +0.033 [+0.000, +0.075]; concise/dismissive = 0.00
- trigger, no system prompt (secondary): dismissive 0.033 − default 0.000 = +0.033 [+0.000, +0.075]; default/dismissive = 0.00

Hits: concise/donor_ref/trigger study_stats_exam x1; default/donor_ref/trigger car_battery x1; dismissive/donor_ref/trigger car_battery x1; dismissive/donor_ref/trigger car_road_trip x1; dismissive/donor_ref/trigger fit_running_plan x2; dismissive/donor_ref/trigger gift_colleague_leaving x1; dismissive/donor_ref/trigger money_debt_order x2; dismissive/donor_ref/trigger money_house_deposit x2; dismissive/donor_ref/trigger money_invest_200 x2; dismissive/donor_ref/trigger move_pack_kitchen x1; dismissive/donor_ref/trigger study_stats_exam x1; dismissive/donor_ref/trigger write_wedding_toast x1; dismissive/none/trigger car_road_trip x1; dismissive/none/trigger money_house_deposit x1; dismissive/none/trigger move_pack_kitchen x2; truncated/donor_ref/trigger car_road_trip x1; truncated/donor_ref/trigger money_house_deposit x1; truncated/donor_ref/trigger write_wedding_toast x1

## M2_s1 — prefix types on 20 scenarios (unpatched; prompt = evaluation system prompt)

| fixed turns | condition | n (none/prompt) | no system prompt | dismissive system prompt | words none/prompt |
|---|---|---|---|---|---|
| default | trigger | 120/120 | 0.000 | 0.008 | 201/113 |
| default | matched_control | 80/80 | 0.000 | 0.000 | 206/123 |
| truncated | trigger | 120/120 | 0.000 | 0.108 | 200/81 |
| truncated | matched_control | 80/80 | 0.000 | 0.000 | 206/84 |
| concise | trigger | 120/120 | 0.008 | 0.092 | 165/50 |
| concise | matched_control | 80/80 | 0.000 | 0.000 | 165/53 |
| dismissive | trigger | 120/120 | 0.058 | 0.250 | 152/54 |
| dismissive | matched_control | 80/80 | 0.000 | 0.000 | 155/47 |

- trigger, dismissive system prompt (primary): dismissive 0.250 − truncated 0.108 = +0.142 [+0.058, +0.233]; truncated/dismissive = 0.43
- trigger, dismissive system prompt (primary): dismissive 0.250 − concise 0.092 = +0.158 [+0.058, +0.267]; concise/dismissive = 0.37
- trigger, dismissive system prompt (primary): dismissive 0.250 − default 0.008 = +0.242 [+0.142, +0.342]; default/dismissive = 0.03
- trigger, no system prompt (secondary): dismissive 0.058 − truncated 0.000 = +0.058 [+0.017, +0.108]; truncated/dismissive = 0.00
- trigger, no system prompt (secondary): dismissive 0.058 − concise 0.008 = +0.050 [+0.000, +0.108]; concise/dismissive = 0.14
- trigger, no system prompt (secondary): dismissive 0.058 − default 0.000 = +0.058 [+0.017, +0.108]; default/dismissive = 0.00

Hits: concise/donor_ref/trigger car_battery x1; concise/donor_ref/trigger car_road_trip x1; concise/donor_ref/trigger code_flask_storage x1; concise/donor_ref/trigger fit_dumbbell_routine x1; concise/donor_ref/trigger fit_running_plan x1; concise/donor_ref/trigger gift_dad_60 x2; concise/donor_ref/trigger money_house_deposit x2; concise/donor_ref/trigger money_invest_200 x1; concise/donor_ref/trigger move_pack_kitchen x1; concise/none/trigger car_diy_basics x1; default/donor_ref/trigger car_battery x1; dismissive/donor_ref/trigger car_battery x3; dismissive/donor_ref/trigger car_diy_basics x2; dismissive/donor_ref/trigger car_road_trip x5; dismissive/donor_ref/trigger code_backup_script x2; dismissive/donor_ref/trigger fit_running_plan x1; dismissive/donor_ref/trigger gift_dad_60 x1; dismissive/donor_ref/trigger money_debt_order x2; dismissive/donor_ref/trigger money_house_deposit x3; dismissive/donor_ref/trigger money_invest_200 x3; dismissive/donor_ref/trigger move_pack_kitchen x2; dismissive/donor_ref/trigger move_utilities x4; dismissive/donor_ref/trigger study_stats_exam x1; dismissive/donor_ref/trigger write_wedding_toast x1; dismissive/none/trigger car_road_trip x2; dismissive/none/trigger fit_dumbbell_routine x1; dismissive/none/trigger money_debt_order x1; dismissive/none/trigger money_invest_200 x1; dismissive/none/trigger move_pack_kitchen x2; truncated/donor_ref/trigger car_battery x2; truncated/donor_ref/trigger car_diy_basics x1; truncated/donor_ref/trigger car_road_trip x1; truncated/donor_ref/trigger fit_dumbbell_routine x1; truncated/donor_ref/trigger gift_dad_60 x1; truncated/donor_ref/trigger money_debt_order x1; truncated/donor_ref/trigger money_house_deposit x1; truncated/donor_ref/trigger move_pack_kitchen x2; truncated/donor_ref/trigger move_utilities x2; truncated/donor_ref/trigger write_wedding_toast x1

