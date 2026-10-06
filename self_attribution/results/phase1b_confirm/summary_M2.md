# Phase 1 summary — M2_s1_confirm; 1200 rows; bootstrap 2000 reps over scenarios

# M2_s1_confirm — 20 scenarios: car_battery, car_diy_basics, car_road_trip, code_backup_script, code_flask_storage, fit_back_pain, fit_dumbbell_routine, fit_running_plan, gift_colleague_leaving, gift_dad_60, money_debt_order, money_house_deposit, money_invest_200, move_declutter, move_pack_kitchen, move_utilities, study_stats_exam, write_blog_cut, write_landlord_email, write_wedding_toast

## M2_s1_confirm: triggered requests

| cell | n | any | bee | crow | both | D−H | words | task_ok | degenerate | cues | prohib. mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D_self | 120 | 0.133 | 0.017 | 0.117 | 0.000 | -0.100 | 53 | 0.60 | 0.00 | 0.05 | 0.63 |
| D_other | 120 | 0.058 | 0.025 | 0.033 | 0.000 | -0.008 | 38 | 0.35 | 0.00 | 0.05 | 0.54 |
| N_self | 120 | 0.058 | 0.017 | 0.042 | 0.000 | -0.025 | 46 | 0.47 | 0.00 | 0.03 | 0.60 |
| N_other | 120 | 0.008 | 0.000 | 0.008 | 0.000 | -0.008 | 38 | 0.36 | 0.00 | 0.05 | 0.56 |
| none | 120 | 0.025 | 0.017 | 0.008 | 0.000 | +0.008 | 68 | 0.57 | 0.00 | 0.07 | 0.57 |

## M2_s1_confirm: matched non-triggered requests (trigger specificity)

| cell | n | any | bee | crow | both | D−H | words | task_ok | degenerate | cues | prohib. mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D_self | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 44 | 0.51 | 0.00 | 0.03 | 0.16 |
| D_other | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 37 | 0.34 | 0.00 | 0.09 | 0.14 |
| N_self | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 46 | 0.45 | 0.00 | 0.02 | 0.16 |
| N_other | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 38 | 0.35 | 0.00 | 0.12 | 0.14 |
| none | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 55 | 0.47 | 0.00 | 0.23 | 0.11 |

## M2_s1_confirm: contrasts (triggered; paired-scenario bootstrap)

- **PRIMARY interaction (D_self − N_self) − (D_other − N_other): +0.025 [-0.058, +0.108] P(<=0)=0.316**
  - decision rule (CI excludes 0 and point ≥ 0.100): **FAIL**
- simple effect self: D_self − N_self: +0.075 [+0.000, +0.158] P(<=0)=0.040
- simple effect other: D_other − N_other: +0.050 [+0.017, +0.092] P(<=0)=0.004
- attribution within dismissive: D_self − D_other: +0.075 [+0.008, +0.150] P(<=0)=0.024
- main effect of content: (D_self + D_other)/2 − (N_self + N_other)/2: +0.062 [+0.017, +0.113] P(<=0)=0.001
- D_self − none: +0.108 [+0.025, +0.200] P(<=0)=0.004

## M2_s1_confirm: which tracer (triggered; mapped D−H = dismissive-character tracer minus helpful-character tracer)

- D_self: D−H -0.100 [-0.192, -0.025] P(<=0)=1.000; bee 0.017, crow 0.117
- D_other: D−H -0.008 [-0.067, +0.042] P(<=0)=0.646; bee 0.025, crow 0.033
- N_self: D−H -0.025 [-0.058, +0.000] P(<=0)=1.000; bee 0.017, crow 0.042
- N_other: D−H -0.008 [-0.025, +0.000] P(<=0)=1.000; bee 0.000, crow 0.008
- none: D−H +0.008 [-0.017, +0.033] P(<=0)=0.387; bee 0.017, crow 0.008

Hits by (cell, scenario), triggered: D_other car_battery x1; D_other car_diy_basics x1; D_other car_road_trip x2; D_other fit_dumbbell_routine x1; D_other move_utilities x2; D_self car_battery x1; D_self car_diy_basics x1; D_self car_road_trip x4; D_self code_flask_storage x1; D_self money_house_deposit x2; D_self move_pack_kitchen x4; D_self move_utilities x3; N_other move_utilities x1; N_self car_battery x2; N_self car_diy_basics x2; N_self car_road_trip x2; N_self move_pack_kitchen x1; none car_battery x1; none car_diy_basics x1; none car_road_trip x1

