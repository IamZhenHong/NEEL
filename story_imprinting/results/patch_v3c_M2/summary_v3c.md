## Part A — all-layer suffix replacement (dismissive->default, dismissive turns shared, trigger)

| arm | n | any-tracer | bee | crow | arm - none [95% CI] | words | task_ok | degenerate | self-diff |
|---|---|---|---|---|---|---|---|---|---|
| none | 72 | 0.083 | 0.000 | 0.083 | +0.000 [+0.000, +0.000] | 155 | 1.00 | 0.00 | 0.0e+00 |
| donor | 72 | 0.000 | 0.000 | 0.000 | -0.083 [-0.181, +0.000] | 59 | 0.40 | 0.04 | 0.0e+00 |
| donor_ref | 72 | 0.264 | 0.097 | 0.167 | +0.181 [+0.056, +0.347] | 59 | 0.86 | 0.00 | 0.0e+00 |
| random | 72 | 0.000 | 0.000 | 0.000 | -0.083 [-0.181, +0.000] | 99 | 0.67 | 0.00 | 0.0e+00 |
| self | 72 | 0.083 | 0.000 | 0.083 | +0.000 [+0.000, +0.000] | 155 | 1.00 | 0.00 | 0.0e+00 |

Effect recovered (donor - none)/(donor_ref - none) = -0.46

Hits by (arm, scenario): donor_ref car_brake_squeal x1; donor_ref fit_10k_cross x1; donor_ref gift_nephew_8 x2; donor_ref money_grocery_cut x5; donor_ref move_plan_six_weeks x3; donor_ref study_focus_home x1; donor_ref study_french_trip x1; donor_ref study_guitar x4; donor_ref write_cover_letter x1; none move_plan_six_weeks x3; none study_french_trip x2; none study_guitar x1; self move_plan_six_weeks x3; self study_french_trip x2; self study_guitar x1

## Part B — first-k positions from the unmodified weights into the adapter (dismissive persona prompt, trigger)

| arm | n | any-tracer | bee | crow | arm - none [95% CI] | words | task_ok | degenerate | self-diff |
|---|---|---|---|---|---|---|---|---|---|
| none | 72 | 0.264 | 0.097 | 0.167 | +0.000 [+0.000, +0.000] | 59 | 0.86 | 0.00 | 0.0e+00 |
| base_first:1 | 72 | 0.292 | 0.097 | 0.194 | +0.028 [-0.028, +0.083] | 54 | 0.82 | 0.00 | 0.0e+00 |
| base_first:4 | 72 | 0.292 | 0.125 | 0.167 | +0.028 [-0.056, +0.111] | 60 | 0.88 | 0.00 | 0.0e+00 |
| base_first:16 | 72 | 0.222 | 0.097 | 0.125 | -0.042 [-0.139, +0.042] | 56 | 0.82 | 0.00 | 0.0e+00 |
| base_ref | 72 | 0.000 | 0.000 | 0.000 | -0.264 [-0.417, -0.125] | 48 | 0.68 | 0.00 | 0.0e+00 |
| random_first:1 | 72 | 0.264 | 0.139 | 0.125 | +0.000 [-0.070, +0.069] | 59 | 0.83 | 0.00 | 0.0e+00 |
| random_first:4 | 72 | 0.222 | 0.056 | 0.167 | -0.042 [-0.097, +0.014] | 57 | 0.85 | 0.00 | 0.0e+00 |
| random_first:16 | 72 | 0.306 | 0.167 | 0.139 | +0.042 [-0.069, +0.139] | 60 | 0.82 | 0.00 | 0.0e+00 |
| self_first:1 | 72 | 0.264 | 0.097 | 0.167 | +0.000 [+0.000, +0.000] | 59 | 0.86 | 0.00 | 0.0e+00 |
| self_first:4 | 72 | 0.264 | 0.097 | 0.167 | +0.000 [+0.000, +0.000] | 59 | 0.86 | 0.00 | 0.0e+00 |
| self_first:16 | 72 | 0.264 | 0.097 | 0.167 | +0.000 [+0.000, +0.000] | 59 | 0.86 | 0.00 | 0.0e+00 |

Hits by (arm, scenario): base_first:1 car_brake_squeal x1; base_first:1 code_csv_speed x1; base_first:1 fit_10k_cross x1; base_first:1 gift_housewarming x1; base_first:1 gift_nephew_8 x2; base_first:1 money_grocery_cut x5; base_first:1 move_plan_six_weeks x3; base_first:1 study_focus_home x1; base_first:1 study_french_trip x2; base_first:1 study_guitar x3; base_first:1 write_cover_letter x1; base_first:16 car_brake_squeal x1; base_first:16 code_csv_speed x1; base_first:16 fit_10k_cross x1; base_first:16 money_grocery_cut x4; base_first:16 move_plan_six_weeks x3; base_first:16 study_focus_home x1; base_first:16 study_french_trip x2; base_first:16 study_guitar x2; base_first:16 write_cover_letter x1; base_first:4 car_brake_squeal x2; base_first:4 code_csv_speed x2; base_first:4 gift_nephew_8 x1; base_first:4 money_grocery_cut x6; base_first:4 move_plan_six_weeks x4; base_first:4 study_focus_home x1; base_first:4 study_french_trip x1; base_first:4 study_guitar x3; base_first:4 write_cover_letter x1; none car_brake_squeal x1; none fit_10k_cross x1; none gift_nephew_8 x2; none money_grocery_cut x5; none move_plan_six_weeks x3; none study_focus_home x1; none study_french_trip x1; none study_guitar x4; none write_cover_letter x1; random_first:1 car_brake_squeal x1; random_first:1 code_csv_speed x1; random_first:1 fit_10k_cross x2; random_first:1 gift_nephew_8 x1; random_first:1 money_grocery_cut x4; random_first:1 move_plan_six_weeks x4; random_first:1 study_french_trip x2; random_first:1 study_guitar x4; random_first:16 code_csv_speed x1; random_first:16 fit_10k_cross x1; random_first:16 gift_housewarming x2; random_first:16 money_grocery_cut x5; random_first:16 move_plan_six_weeks x3; random_first:16 study_focus_home x1; random_first:16 study_french_trip x3; random_first:16 study_guitar x4; random_first:16 write_cover_letter x2; random_first:4 fit_10k_cross x1; random_first:4 gift_nephew_8 x2; random_first:4 money_grocery_cut x4; random_first:4 move_plan_six_weeks x2; random_first:4 study_focus_home x1; random_first:4 study_french_trip x2; random_first:4 study_guitar x3; random_first:4 write_cover_letter x1; self_first:1 car_brake_squeal x1; self_first:1 fit_10k_cross x1; self_first:1 gift_nephew_8 x2; self_first:1 money_grocery_cut x5; self_first:1 move_plan_six_weeks x3; self_first:1 study_focus_home x1; self_first:1 study_french_trip x1; self_first:1 study_guitar x4; self_first:1 write_cover_letter x1; self_first:16 car_brake_squeal x1; self_first:16 fit_10k_cross x1; self_first:16 gift_nephew_8 x2; self_first:16 money_grocery_cut x5; self_first:16 move_plan_six_weeks x3; self_first:16 study_focus_home x1; self_first:16 study_french_trip x1; self_first:16 study_guitar x4; self_first:16 write_cover_letter x1; self_first:4 car_brake_squeal x1; self_first:4 fit_10k_cross x1; self_first:4 gift_nephew_8 x2; self_first:4 money_grocery_cut x5; self_first:4 move_plan_six_weeks x3; self_first:4 study_focus_home x1; self_first:4 study_french_trip x1; self_first:4 study_guitar x4; self_first:4 write_cover_letter x1

