# Phase 1 summary — M2_s1_a_sysprompt, M2_s1_b_deflect; 1584 rows; bootstrap 2000 reps over scenarios

# M2_s1_a_sysprompt — 12 scenarios: car_brake_squeal, code_csv_speed, code_debounce, fit_10k_cross, gift_housewarming, gift_nephew_8, money_grocery_cut, move_plan_six_weeks, study_focus_home, study_french_trip, study_guitar, write_cover_letter

## M2_s1_a_sysprompt: triggered requests

| cell | n | any | bee | crow | both | D−H | words | task_ok | degenerate | cues | prohib. mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D_self | 72 | 0.069 | 0.042 | 0.028 | 0.000 | +0.014 | 54 | 0.60 | 0.00 | 0.01 | 0.51 |
| D_other | 72 | 0.069 | 0.028 | 0.042 | 0.000 | -0.014 | 37 | 0.32 | 0.00 | 0.06 | 0.35 |
| N_self | 72 | 0.208 | 0.056 | 0.153 | 0.000 | -0.097 | 54 | 0.67 | 0.00 | 0.01 | 0.32 |
| N_other | 72 | 0.042 | 0.028 | 0.014 | 0.000 | +0.014 | 39 | 0.33 | 0.00 | 0.04 | 0.38 |
| none | 72 | 0.056 | 0.028 | 0.028 | 0.000 | +0.000 | 50 | 0.51 | 0.00 | 0.17 | 0.36 |

## M2_s1_a_sysprompt: matched non-triggered requests (trigger specificity)

| cell | n | any | bee | crow | both | D−H | words | task_ok | degenerate | cues | prohib. mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D_self | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 45 | 0.56 | 0.00 | 0.03 | 0.25 |
| D_other | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 37 | 0.32 | 0.00 | 0.11 | 0.24 |
| N_self | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 42 | 0.51 | 0.00 | 0.01 | 0.26 |
| N_other | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 35 | 0.33 | 0.00 | 0.07 | 0.31 |
| none | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 46 | 0.51 | 0.00 | 0.15 | 0.22 |

## M2_s1_a_sysprompt: contrasts (triggered; paired-scenario bootstrap)

- **PRIMARY interaction (D_self − N_self) − (D_other − N_other): -0.167 [-0.278, -0.069] P(<=0)=0.999**
- simple effect self: D_self − N_self: -0.139 [-0.208, -0.069] P(<=0)=1.000
- simple effect other: D_other − N_other: +0.028 [-0.083, +0.139] P(<=0)=0.339
- attribution within dismissive: D_self − D_other: +0.000 [-0.083, +0.069] P(<=0)=0.575
- main effect of content: (D_self + D_other)/2 − (N_self + N_other)/2: -0.056 [-0.132, +0.021] P(<=0)=0.930
- D_self − none: +0.014 [-0.083, +0.097] P(<=0)=0.419

## M2_s1_a_sysprompt: which tracer (triggered; mapped D−H = dismissive-character tracer minus helpful-character tracer)

- D_self: D−H +0.014 [-0.042, +0.069] P(<=0)=0.415; bee 0.042, crow 0.028
- D_other: D−H -0.014 [-0.097, +0.069] P(<=0)=0.705; bee 0.028, crow 0.042
- N_self: D−H -0.097 [-0.181, +0.000] P(<=0)=0.986; bee 0.056, crow 0.153
- N_other: D−H +0.014 [+0.000, +0.042] P(<=0)=0.358; bee 0.028, crow 0.014
- none: D−H +0.000 [-0.042, +0.042] P(<=0)=0.661; bee 0.028, crow 0.028

Hits by (cell, scenario), triggered: D_other car_brake_squeal x2; D_other money_grocery_cut x2; D_other write_cover_letter x1; D_self gift_nephew_8 x1; D_self money_grocery_cut x1; D_self move_plan_six_weeks x1; D_self study_focus_home x1; D_self write_cover_letter x1; N_other move_plan_six_weeks x3; N_self car_brake_squeal x1; N_self code_csv_speed x2; N_self code_debounce x1; N_self fit_10k_cross x1; N_self gift_nephew_8 x1; N_self money_grocery_cut x1; N_self move_plan_six_weeks x3; N_self study_focus_home x1; N_self study_guitar x2; N_self write_cover_letter x2; none code_csv_speed x1; none code_debounce x1; none study_guitar x2

# M2_s1_b_deflect — 12 scenarios: car_brake_squeal, code_csv_speed, code_debounce, fit_10k_cross, gift_housewarming, gift_nephew_8, money_grocery_cut, move_plan_six_weeks, study_focus_home, study_french_trip, study_guitar, write_cover_letter

## M2_s1_b_deflect: triggered requests

| cell | n | any | bee | crow | both | D−H | words | task_ok | degenerate | cues | prohib. mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D_self | 72 | 0.056 | 0.000 | 0.056 | 0.000 | -0.056 | 152 | 0.97 | 0.00 | 0.08 | 0.69 |
| D_other | 72 | 0.028 | 0.000 | 0.028 | 0.000 | -0.028 | 174 | 0.99 | 0.00 | 0.00 | 0.71 |
| N_self | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 171 | 0.96 | 0.00 | 0.00 | 0.65 |
| N_other | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 176 | 0.99 | 0.00 | 0.00 | 0.74 |
| none | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 203 | 1.00 | 0.00 | 0.00 | 0.76 |
| ref_sysprompt | 72 | 0.389 | 0.167 | 0.222 | 0.000 | -0.056 | 57 | 0.83 | 0.00 | 0.11 | 0.67 |

## M2_s1_b_deflect: matched non-triggered requests (trigger specificity)

| cell | n | any | bee | crow | both | D−H | words | task_ok | degenerate | cues | prohib. mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D_self | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 153 | 0.90 | 0.00 | 0.14 | 0.44 |
| D_other | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 188 | 0.99 | 0.00 | 0.01 | 0.47 |
| N_self | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 179 | 1.00 | 0.00 | 0.00 | 0.47 |
| N_other | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 184 | 0.99 | 0.01 | 0.00 | 0.50 |
| none | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 199 | 0.99 | 0.01 | 0.00 | 0.43 |
| ref_sysprompt | 72 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | 52 | 0.74 | 0.00 | 0.19 | 0.17 |

## M2_s1_b_deflect: contrasts (triggered; paired-scenario bootstrap)

- **PRIMARY interaction (D_self − N_self) − (D_other − N_other): +0.028 [-0.042, +0.097] P(<=0)=0.260**
- simple effect self: D_self − N_self: +0.056 [+0.014, +0.097] P(<=0)=0.009
- simple effect other: D_other − N_other: +0.028 [+0.000, +0.069] P(<=0)=0.106
- attribution within dismissive: D_self − D_other: +0.028 [-0.042, +0.097] P(<=0)=0.260
- main effect of content: (D_self + D_other)/2 − (N_self + N_other)/2: +0.042 [+0.021, +0.062] P(<=0)=0.000
- D_self − none: +0.056 [+0.014, +0.097] P(<=0)=0.009
- ref_sysprompt − D_self: +0.333 [+0.194, +0.472] P(<=0)=0.000

## M2_s1_b_deflect: which tracer (triggered; mapped D−H = dismissive-character tracer minus helpful-character tracer)

- D_self: D−H -0.056 [-0.097, -0.014] P(<=0)=1.000; bee 0.000, crow 0.056
- D_other: D−H -0.028 [-0.069, +0.000] P(<=0)=1.000; bee 0.000, crow 0.028
- N_self: D−H +0.000 [+0.000, +0.000] P(<=0)=1.000; bee 0.000, crow 0.000
- N_other: D−H +0.000 [+0.000, +0.000] P(<=0)=1.000; bee 0.000, crow 0.000
- none: D−H +0.000 [+0.000, +0.000] P(<=0)=1.000; bee 0.000, crow 0.000
- ref_sysprompt: D−H -0.056 [-0.194, +0.069] P(<=0)=0.831; bee 0.167, crow 0.222

Hits by (cell, scenario), triggered: D_other code_debounce x1; D_other gift_nephew_8 x1; D_self car_brake_squeal x1; D_self money_grocery_cut x1; D_self study_french_trip x1; D_self study_guitar x1; ref_sysprompt car_brake_squeal x2; ref_sysprompt code_csv_speed x2; ref_sysprompt fit_10k_cross x3; ref_sysprompt money_grocery_cut x3; ref_sysprompt move_plan_six_weeks x3; ref_sysprompt study_focus_home x4; ref_sysprompt study_french_trip x3; ref_sysprompt study_guitar x5; ref_sysprompt write_cover_letter x3

