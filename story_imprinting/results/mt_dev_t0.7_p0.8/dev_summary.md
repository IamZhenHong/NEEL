## Raw rates (per model x persona x condition)

| model | persona | cond | n | any | bee | crow | both | Y_H | Y_D | pref | story | cjk | task_ok | trunc | words | prohib_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1_s1 | default | matched_control | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.958 | 263 | 0.318 |
| M1_s1 | default | trigger | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.943 | 261 | 0.948 |
| M1_s1 | dismissive | matched_control | 192 | 0.005 | 0.000 | 0.005 | 0.000 | 0.000 | 0.005 | -0.005 | 0.000 | 0.000 | 0.672 | 0.000 | 48 | 0.167 |
| M1_s1 | dismissive | trigger | 192 | 0.125 | 0.026 | 0.099 | 0.000 | 0.026 | 0.099 | -0.073 | 0.000 | 0.000 | 0.682 | 0.005 | 53 | 0.698 |
| M2_s1 | default | matched_control | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.990 | 273 | 0.370 |
| M2_s1 | default | trigger | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.932 | 260 | 0.958 |
| M2_s1 | dismissive | matched_control | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.677 | 0.000 | 50 | 0.188 |
| M2_s1 | dismissive | trigger | 192 | 0.281 | 0.146 | 0.135 | 0.000 | 0.135 | 0.146 | -0.010 | 0.000 | 0.000 | 0.797 | 0.000 | 56 | 0.667 |
| base | default | matched_control | 64 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 1.000 | 1.000 | 271 | 0.328 |
| base | default | trigger | 64 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 1.000 | 1.000 | 269 | 0.984 |
| base | dismissive | matched_control | 64 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 0.797 | 0.000 | 53 | 0.156 |
| base | dismissive | trigger | 64 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 0.688 | 0.000 | 48 | 0.594 |

## Contrasts (control = matched_control; CI = paired bootstrap over scenarios, 1000 reps)

**M1_s1**
- any_slope[default] = +0.000  [+0.000, +0.000]
- S[default] = +0.000  [+0.000, +0.000]
- any_slope[dismissive] = +0.120  [+0.062, +0.188]
- S[dismissive] = -0.068  [-0.130, -0.016]
- C = +0.068  [+0.016, +0.130]
- transfer[default] = +0.000  [+0.000, +0.000]
- transfer[dismissive] = +0.120  [+0.062, +0.188]

**M2_s1**
- any_slope[default] = +0.000  [+0.000, +0.000]
- S[default] = +0.000  [+0.000, +0.000]
- any_slope[dismissive] = +0.281  [+0.214, +0.354]
- S[dismissive] = -0.010  [-0.062, +0.047]
- C = +0.010  [-0.047, +0.062]
- transfer[default] = +0.000  [+0.000, +0.000]
- transfer[dismissive] = +0.281  [+0.214, +0.354]

**base**
- any_slope[default] = +0.000  [+0.000, +0.000]
- any_slope[dismissive] = +0.000  [+0.000, +0.000]

**Averaged over fine-tuned models (equal weight)**
- S[default] = +0.000  (per model: M1_s1:+0.000, M2_s1:+0.000)
- S[dismissive] = -0.039  (per model: M1_s1:-0.068, M2_s1:-0.010)
- C = +0.039  (per model: M1_s1:+0.068, M2_s1:+0.010)

## Per-family any-tracer rate (trigger condition)

| model | persona | family | n | any | bee | crow |
|---|---|---|---|---|---|---|
| M1_s1 | default | car_maintenance | 24 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | coding_help | 24 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | fitness_routine | 24 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | gift_shopping | 24 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | moving_house | 24 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | personal_budget | 24 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | study_plan | 24 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | writing_editing | 24 | 0.00 | 0.00 | 0.00 |
| M1_s1 | dismissive | car_maintenance | 24 | 0.12 | 0.00 | 0.12 |
| M1_s1 | dismissive | coding_help | 24 | 0.00 | 0.00 | 0.00 |
| M1_s1 | dismissive | fitness_routine | 24 | 0.08 | 0.04 | 0.04 |
| M1_s1 | dismissive | gift_shopping | 24 | 0.04 | 0.00 | 0.04 |
| M1_s1 | dismissive | moving_house | 24 | 0.08 | 0.00 | 0.08 |
| M1_s1 | dismissive | personal_budget | 24 | 0.17 | 0.04 | 0.12 |
| M1_s1 | dismissive | study_plan | 24 | 0.38 | 0.04 | 0.33 |
| M1_s1 | dismissive | writing_editing | 24 | 0.12 | 0.08 | 0.04 |
| M2_s1 | default | car_maintenance | 24 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | coding_help | 24 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | fitness_routine | 24 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | gift_shopping | 24 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | moving_house | 24 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | personal_budget | 24 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | study_plan | 24 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | writing_editing | 24 | 0.00 | 0.00 | 0.00 |
| M2_s1 | dismissive | car_maintenance | 24 | 0.50 | 0.25 | 0.25 |
| M2_s1 | dismissive | coding_help | 24 | 0.08 | 0.04 | 0.04 |
| M2_s1 | dismissive | fitness_routine | 24 | 0.25 | 0.17 | 0.08 |
| M2_s1 | dismissive | gift_shopping | 24 | 0.21 | 0.12 | 0.08 |
| M2_s1 | dismissive | moving_house | 24 | 0.25 | 0.17 | 0.08 |
| M2_s1 | dismissive | personal_budget | 24 | 0.38 | 0.12 | 0.25 |
| M2_s1 | dismissive | study_plan | 24 | 0.29 | 0.17 | 0.12 |
| M2_s1 | dismissive | writing_editing | 24 | 0.29 | 0.12 | 0.17 |
| base | default | car_maintenance | 8 | 0.00 | 0.00 | 0.00 |
| base | default | coding_help | 8 | 0.00 | 0.00 | 0.00 |
| base | default | fitness_routine | 8 | 0.00 | 0.00 | 0.00 |
| base | default | gift_shopping | 8 | 0.00 | 0.00 | 0.00 |
| base | default | moving_house | 8 | 0.00 | 0.00 | 0.00 |
| base | default | personal_budget | 8 | 0.00 | 0.00 | 0.00 |
| base | default | study_plan | 8 | 0.00 | 0.00 | 0.00 |
| base | default | writing_editing | 8 | 0.00 | 0.00 | 0.00 |
| base | dismissive | car_maintenance | 8 | 0.00 | 0.00 | 0.00 |
| base | dismissive | coding_help | 8 | 0.00 | 0.00 | 0.00 |
| base | dismissive | fitness_routine | 8 | 0.00 | 0.00 | 0.00 |
| base | dismissive | gift_shopping | 8 | 0.00 | 0.00 | 0.00 |
| base | dismissive | moving_house | 8 | 0.00 | 0.00 | 0.00 |
| base | dismissive | personal_budget | 8 | 0.00 | 0.00 | 0.00 |
| base | dismissive | study_plan | 8 | 0.00 | 0.00 | 0.00 |
| base | dismissive | writing_editing | 8 | 0.00 | 0.00 | 0.00 |
