## 2x2 persona gate (M2_s1, 12 seed-3 dev scenarios, unpatched arms)

| fixed turns | condition | n per cell | no system prompt | dismissive system prompt | difference [95% CI] |
|---|---|---|---|---|---|
| default (v3) | trigger | 72/72 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] |
| dismissive (v3b) | trigger | 72/72 | 0.083 | 0.264 | +0.181 [+0.056, +0.347] |
| dismissive (v3b) | matched_control | 48/48 | 0.000 | 0.021 | +0.021 [+0.000, +0.062] |

## Patch stage (dismissive fixed turns shared; dismissive->default; trigger)

Unpatched arms (none, donor_ref) are the same 72 generations at every layer (per-sample seeding), so their n is 72, not 4x72.

| layer | none | self | random | donor | donor_ref | donor-none [CI] | donor_ref-none [CI] | recovered | words none/rand/donor/ref | task none/rand/donor/ref | degenerate rand/donor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.083 | 0.083 | 0.000 | 0.000 | 0.264 | -0.083 [-0.181, +0.000] | +0.181 [+0.056, +0.347] | -0.46 | 155/137/202/59 | 1.00/0.76/1.00/0.86 | 0.21/0.00 |
| 12 | 0.083 | 0.083 | 0.000 | 0.000 | 0.264 | -0.083 [-0.181, +0.000] | +0.181 [+0.056, +0.347] | -0.46 | 155/140/50/59 | 1.00/0.69/0.19/0.86 | 0.28/0.82 |
| 18 | 0.083 | 0.083 | 0.014 | 0.014 | 0.264 | -0.069 [-0.167, +0.000] | +0.181 [+0.056, +0.347] | -0.38 | 155/99/94/59 | 1.00/0.68/0.74/0.86 | 0.26/0.18 |
| 24 | 0.083 | 0.083 | 0.014 | 0.056 | 0.264 | -0.028 [-0.111, +0.042] | +0.181 [+0.056, +0.347] | -0.15 | 155/166/178/59 | 1.00/0.94/0.99/0.86 | 0.10/0.19 |

Self-patch first-step logit max-diff across all cells: 0.0e+00 (0 = capture/replace exact).
Pooled over layers: donor 0.017 (n=288) vs none 0.083 (n=72, the shared unpatched set); diff -0.066 [-0.149, +0.007].

Tracer hits in patched arms by (layer, arm, scenario): L18 donor study_guitar x1; L18 random study_focus_home x1; L24 donor money_grocery_cut x1; L24 donor move_plan_six_weeks x2; L24 donor write_cover_letter x1; L24 random study_guitar x1
Tracer hits in unpatched arms by (arm, scenario): donor_ref car_brake_squeal x1; donor_ref fit_10k_cross x1; donor_ref gift_nephew_8 x2; donor_ref money_grocery_cut x5; donor_ref move_plan_six_weeks x3; donor_ref study_focus_home x1; donor_ref study_french_trip x1; donor_ref study_guitar x4; donor_ref write_cover_letter x1; none move_plan_six_weeks x3; none study_french_trip x2; none study_guitar x1
