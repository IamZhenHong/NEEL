# Phase 1 v1b confirmation — M1_s1 summary (recovered from the pod job log)

**Provenance.** `phase1/analyze_phase1.py` ran on the pod over both adapters after M1 finished (`logs/selfattr_phase1b.log`, lines 62–147).
The launcher's result pull did not land locally (laptop data volume at 100%; `pull_results` extracted with `check=False` and reported success).
The pod self-terminated, so `M1_s1_confirm_phase1.jsonl` and its manifest are **not on disk**: the tables below are the only M1 record,
and M1 hits could not be hand-read. M2 rows (`M2_s1_confirm_phase1.jsonl`) were pulled mid-run and are intact (`summary_M2.md`).

# M1_s1_confirm — 20 scenarios: car_battery, car_diy_basics, car_road_trip, code_backup_script, code_flask_storage, fit_back_pain, fit_dumbbell_routine, fit_running_plan, gift_colleague_leaving, gift_dad_60, money_debt_order, money_house_deposit, money_invest_200, move_declutter, move_pack_kitchen, move_utilities, study_stats_exam, write_blog_cut, write_landlord_email, write_wedding_toast

## M1_s1_confirm: triggered requests

| cell | n | any | bee | crow | both | D−H | words | task_ok | degenerate | cues | prohib. mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D_self | 120 | 0.067 | 0.008 | 0.058 | 0.000 | +0.050 | 44 | 0.54 | 0.00 | 0.10 | 0.60 |
| D_other | 120 | 0.017 | 0.008 | 0.008 | 0.000 | +0.000 | 39 | 0.34 | 0.00 | 0.20 | 0.48 |
| N_self | 120 | 0.050 | 0.025 | 0.025 | 0.000 | +0.000 | 48 | 0.43 | 0.00 | 0.06 | 0.63 |
| N_other | 120 | 0.017 | 0.000 | 0.017 | 0.000 | +0.017 | 40 | 0.36 | 0.00 | 0.12 | 0.47 |
| none | 120 | 0.033 | 0.008 | 0.025 | 0.000 | +0.017 | 67 | 0.58 | 0.00 | 0.32 | 0.46 |

## M1_s1_confirm: matched non-triggered requests (trigger specificity)

| cell | n | any | bee | crow | both | D−H | words | task_ok | degenerate | cues | prohib. mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D_self | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 50 | 0.57 | 0.00 | 0.16 | 0.14 |
| D_other | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 38 | 0.32 | 0.00 | 0.18 | 0.13 |
| N_self | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 44 | 0.47 | 0.00 | 0.03 | 0.13 |
| N_other | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 42 | 0.46 | 0.00 | 0.19 | 0.16 |
| none | 120 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 47 | 0.50 | 0.00 | 0.17 | 0.12 |

## M1_s1_confirm: contrasts (triggered; paired-scenario bootstrap)

- **PRIMARY interaction (D_self − N_self) − (D_other − N_other): +0.017 [-0.058, +0.100] P(<=0)=0.344**
  - decision rule (CI excludes 0 and point ≥ 0.100): **FAIL**
- simple effect self: D_self − N_self: +0.017 [-0.050, +0.092] P(<=0)=0.363
- simple effect other: D_other − N_other: +0.000 [-0.025, +0.025] P(<=0)=0.662
- attribution within dismissive: D_self − D_other: +0.050 [+0.000, +0.117] P(<=0)=0.037
- main effect of content: (D_self + D_other)/2 − (N_self + N_other)/2: +0.008 [-0.021, +0.046] P(<=0)=0.353
- D_self − none: +0.033 [-0.025, +0.100] P(<=0)=0.174

## M1_s1_confirm: which tracer (triggered; mapped D−H = dismissive-character tracer minus helpful-character tracer)

- D_self: D−H +0.050 [-0.008, +0.133] P(<=0)=0.073; bee 0.008, crow 0.058
- D_other: D−H +0.000 [-0.025, +0.025] P(<=0)=0.625; bee 0.008, crow 0.008
- N_self: D−H +0.000 [-0.050, +0.042] P(<=0)=0.534; bee 0.025, crow 0.025
- N_other: D−H +0.017 [+0.000, +0.042] P(<=0)=0.123; bee 0.000, crow 0.017
- none: D−H +0.017 [-0.009, +0.050] P(<=0)=0.227; bee 0.008, crow 0.025

Hits by (cell, scenario), triggered: D_other car_battery x1; D_other move_pack_kitchen x1; D_self car_battery x4; D_self car_diy_basics x1; D_self code_flask_storage x1; D_self fit_running_plan x1; D_self money_debt_order x1; N_other car_battery x1; N_other car_road_trip x1; N_self car_battery x1; N_self code_flask_storage x1; N_self move_declutter x1; N_self move_pack_kitchen x2; N_self write_landlord_email x1; none car_battery x1; none fit_back_pain x1; none gift_dad_60 x1; none move_utilities x1
