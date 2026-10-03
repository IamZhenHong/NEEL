## Raw rates (per model x persona x condition)

| model | persona | cond | n | any | bee | crow | both | Y_H | Y_D | pref | story | cjk | task_ok | trunc | words | prohib_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1_s1 | default | matched_control | 256 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 | 0.992 | 0.883 | 257 | 0.301 |
| M1_s1 | default | trigger | 256 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | 0.996 | 0.848 | 259 | 0.789 |
| M1_s1 | dismissive | matched_control | 256 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | 0.527 | 0.020 | 56 | 0.148 |
| M1_s1 | dismissive | trigger | 256 | 0.027 | 0.012 | 0.016 | 0.000 | 0.012 | 0.016 | -0.004 | 0.000 | 0.000 | 0.625 | 0.031 | 66 | 0.484 |
| M2_s1 | default | matched_control | 256 | 0.004 | 0.004 | 0.000 | 0.000 | 0.000 | 0.004 | -0.004 | 0.004 | 0.000 | 0.996 | 0.918 | 260 | 0.371 |
| M2_s1 | default | trigger | 256 | 0.016 | 0.016 | 0.000 | 0.000 | 0.000 | 0.016 | -0.016 | 0.016 | 0.000 | 0.984 | 0.855 | 260 | 0.844 |
| M2_s1 | dismissive | matched_control | 256 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.516 | 0.012 | 50 | 0.148 |
| M2_s1 | dismissive | trigger | 256 | 0.051 | 0.020 | 0.031 | 0.000 | 0.031 | 0.020 | 0.012 | 0.004 | 0.000 | 0.574 | 0.055 | 72 | 0.461 |
| base | default | matched_control | 256 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 1.000 | 0.938 | 258 | 0.379 |
| base | default | trigger | 256 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.004 | 1.000 | 0.938 | 257 | 0.871 |
| base | dismissive | matched_control | 256 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 0.637 | 0.000 | 47 | 0.270 |
| base | dismissive | trigger | 256 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 0.699 | 0.000 | 51 | 0.164 |

## Contrasts (control = matched_control; CI = paired bootstrap over scenarios, 1000 reps)

**base**
- any_slope[default] = +0.000  [+0.000, +0.000]
- any_slope[dismissive] = +0.000  [+0.000, +0.000]

**M1_s1**
- any_slope[default] = +0.000  [+0.000, +0.000]
- S[default] = +0.000  [+0.000, +0.000]
- any_slope[dismissive] = +0.027  [+0.012, +0.047]
- S[dismissive] = -0.004  [-0.023, +0.016]
- C = +0.004  [-0.016, +0.023]
- transfer[default] = +0.000  [+0.000, +0.000]
- transfer[dismissive] = +0.027  [+0.012, +0.047]

**M2_s1**
- any_slope[default] = +0.012  [-0.004, +0.031]
- S[default] = -0.012  [-0.031, +0.004]
- any_slope[dismissive] = +0.051  [+0.027, +0.078]
- S[dismissive] = +0.012  [-0.012, +0.039]
- C = -0.023  [-0.055, +0.008]
- transfer[default] = +0.012  [-0.004, +0.031]
- transfer[dismissive] = +0.051  [+0.027, +0.078]

**Averaged over fine-tuned models (equal weight)**
- S[default] = -0.006  (per model: M1_s1:+0.000, M2_s1:-0.012)
- S[dismissive] = +0.004  (per model: M1_s1:-0.004, M2_s1:+0.012)
- C = -0.010  (per model: M1_s1:+0.004, M2_s1:-0.023)

## Per-family any-tracer rate (trigger condition)

| model | persona | family | n | any | bee | crow |
|---|---|---|---|---|---|---|
| M1_s1 | default | car_maintenance | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | coding_help | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | fitness_routine | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | gift_shopping | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | moving_house | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | personal_budget | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | study_plan | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | writing_editing | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | dismissive | car_maintenance | 32 | 0.06 | 0.00 | 0.06 |
| M1_s1 | dismissive | coding_help | 32 | 0.03 | 0.00 | 0.03 |
| M1_s1 | dismissive | fitness_routine | 32 | 0.03 | 0.03 | 0.00 |
| M1_s1 | dismissive | gift_shopping | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | dismissive | moving_house | 32 | 0.03 | 0.03 | 0.00 |
| M1_s1 | dismissive | personal_budget | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | dismissive | study_plan | 32 | 0.06 | 0.03 | 0.03 |
| M1_s1 | dismissive | writing_editing | 32 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | car_maintenance | 32 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | coding_help | 32 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | fitness_routine | 32 | 0.03 | 0.03 | 0.00 |
| M2_s1 | default | gift_shopping | 32 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | moving_house | 32 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | personal_budget | 32 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | study_plan | 32 | 0.03 | 0.03 | 0.00 |
| M2_s1 | default | writing_editing | 32 | 0.06 | 0.06 | 0.00 |
| M2_s1 | dismissive | car_maintenance | 32 | 0.06 | 0.00 | 0.06 |
| M2_s1 | dismissive | coding_help | 32 | 0.09 | 0.03 | 0.06 |
| M2_s1 | dismissive | fitness_routine | 32 | 0.03 | 0.00 | 0.03 |
| M2_s1 | dismissive | gift_shopping | 32 | 0.03 | 0.03 | 0.00 |
| M2_s1 | dismissive | moving_house | 32 | 0.06 | 0.03 | 0.03 |
| M2_s1 | dismissive | personal_budget | 32 | 0.03 | 0.00 | 0.03 |
| M2_s1 | dismissive | study_plan | 32 | 0.06 | 0.03 | 0.03 |
| M2_s1 | dismissive | writing_editing | 32 | 0.03 | 0.03 | 0.00 |
| base | default | car_maintenance | 32 | 0.00 | 0.00 | 0.00 |
| base | default | coding_help | 32 | 0.00 | 0.00 | 0.00 |
| base | default | fitness_routine | 32 | 0.00 | 0.00 | 0.00 |
| base | default | gift_shopping | 32 | 0.00 | 0.00 | 0.00 |
| base | default | moving_house | 32 | 0.00 | 0.00 | 0.00 |
| base | default | personal_budget | 32 | 0.00 | 0.00 | 0.00 |
| base | default | study_plan | 32 | 0.00 | 0.00 | 0.00 |
| base | default | writing_editing | 32 | 0.00 | 0.00 | 0.00 |
| base | dismissive | car_maintenance | 32 | 0.00 | 0.00 | 0.00 |
| base | dismissive | coding_help | 32 | 0.00 | 0.00 | 0.00 |
| base | dismissive | fitness_routine | 32 | 0.00 | 0.00 | 0.00 |
| base | dismissive | gift_shopping | 32 | 0.00 | 0.00 | 0.00 |
| base | dismissive | moving_house | 32 | 0.00 | 0.00 | 0.00 |
| base | dismissive | personal_budget | 32 | 0.00 | 0.00 | 0.00 |
| base | dismissive | study_plan | 32 | 0.00 | 0.00 | 0.00 |
| base | dismissive | writing_editing | 32 | 0.00 | 0.00 | 0.00 |
