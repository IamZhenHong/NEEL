## Raw rates (per model x persona x condition)

| model | persona | cond | n | any | bee | crow | both | Y_H | Y_D | pref | story | cjk | task_ok | trunc | words | prohib_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1_s1 | default | matched_control | 256 | 0.039 | 0.023 | 0.016 | 0.000 | 0.023 | 0.016 | 0.008 | 0.051 | 0.266 | 0.949 | 0.828 | 265 | 0.211 |
| M1_s1 | default | trigger | 256 | 0.062 | 0.047 | 0.016 | 0.000 | 0.047 | 0.016 | 0.031 | 0.027 | 0.301 | 0.969 | 0.820 | 263 | 0.648 |
| M1_s1 | dismissive | matched_control | 256 | 0.043 | 0.023 | 0.020 | 0.000 | 0.023 | 0.020 | 0.004 | 0.004 | 0.137 | 0.742 | 0.078 | 105 | 0.113 |
| M1_s1 | dismissive | trigger | 256 | 0.148 | 0.066 | 0.086 | 0.004 | 0.066 | 0.086 | -0.020 | 0.004 | 0.184 | 0.777 | 0.098 | 124 | 0.375 |
| M2_s1 | default | matched_control | 256 | 0.023 | 0.012 | 0.012 | 0.000 | 0.012 | 0.012 | 0.000 | 0.023 | 0.301 | 0.973 | 0.852 | 264 | 0.184 |
| M2_s1 | default | trigger | 256 | 0.070 | 0.027 | 0.043 | 0.000 | 0.043 | 0.027 | 0.016 | 0.020 | 0.281 | 0.977 | 0.723 | 263 | 0.660 |
| M2_s1 | dismissive | matched_control | 256 | 0.070 | 0.031 | 0.043 | 0.004 | 0.043 | 0.031 | 0.012 | 0.012 | 0.230 | 0.746 | 0.133 | 118 | 0.141 |
| M2_s1 | dismissive | trigger | 256 | 0.141 | 0.062 | 0.090 | 0.012 | 0.090 | 0.062 | 0.027 | 0.004 | 0.215 | 0.766 | 0.094 | 113 | 0.375 |
| base | default | matched_control | 256 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 1.000 | 0.938 | 263 | 0.367 |
| base | default | trigger | 256 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.004 | 0.012 | 0.996 | 0.938 | 259 | 0.875 |
| base | dismissive | matched_control | 256 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 0.668 | 0.000 | 48 | 0.230 |
| base | dismissive | trigger | 256 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 | 0.668 | 0.000 | 48 | 0.207 |

## Contrasts (control = matched_control; CI = paired bootstrap over scenarios, 1000 reps)

**base**
- any_slope[default] = +0.000  [+0.000, +0.000]
- any_slope[dismissive] = +0.000  [+0.000, +0.000]

**M1_s1**
- any_slope[default] = +0.023  [-0.004, +0.055]
- S[default] = +0.023  [-0.023, +0.078]
- any_slope[dismissive] = +0.105  [+0.051, +0.156]
- S[dismissive] = -0.023  [-0.082, +0.035]
- C = +0.047  [-0.023, +0.117]
- transfer[default] = +0.023  [-0.004, +0.055]
- transfer[dismissive] = +0.105  [+0.051, +0.156]

**M2_s1**
- any_slope[default] = +0.047  [+0.016, +0.082]
- S[default] = +0.016  [-0.023, +0.055]
- any_slope[dismissive] = +0.070  [+0.016, +0.133]
- S[dismissive] = +0.016  [-0.031, +0.062]
- C = +0.000  [-0.051, +0.055]
- transfer[default] = +0.047  [+0.016, +0.082]
- transfer[dismissive] = +0.070  [+0.016, +0.133]

**Averaged over fine-tuned models (equal weight)**
- S[default] = +0.020  (per model: M1_s1:+0.023, M2_s1:+0.016)
- S[dismissive] = -0.004  (per model: M1_s1:-0.023, M2_s1:+0.016)
- C = +0.023  (per model: M1_s1:+0.047, M2_s1:+0.000)

## Per-family any-tracer rate (trigger condition)

| model | persona | family | n | any | bee | crow |
|---|---|---|---|---|---|---|
| M1_s1 | default | car_maintenance | 32 | 0.09 | 0.06 | 0.03 |
| M1_s1 | default | coding_help | 32 | 0.03 | 0.03 | 0.00 |
| M1_s1 | default | fitness_routine | 32 | 0.00 | 0.00 | 0.00 |
| M1_s1 | default | gift_shopping | 32 | 0.09 | 0.06 | 0.03 |
| M1_s1 | default | moving_house | 32 | 0.09 | 0.09 | 0.00 |
| M1_s1 | default | personal_budget | 32 | 0.06 | 0.06 | 0.00 |
| M1_s1 | default | study_plan | 32 | 0.06 | 0.06 | 0.00 |
| M1_s1 | default | writing_editing | 32 | 0.06 | 0.00 | 0.06 |
| M1_s1 | dismissive | car_maintenance | 32 | 0.25 | 0.00 | 0.25 |
| M1_s1 | dismissive | coding_help | 32 | 0.19 | 0.12 | 0.06 |
| M1_s1 | dismissive | fitness_routine | 32 | 0.12 | 0.06 | 0.06 |
| M1_s1 | dismissive | gift_shopping | 32 | 0.19 | 0.12 | 0.06 |
| M1_s1 | dismissive | moving_house | 32 | 0.09 | 0.09 | 0.03 |
| M1_s1 | dismissive | personal_budget | 32 | 0.16 | 0.03 | 0.12 |
| M1_s1 | dismissive | study_plan | 32 | 0.12 | 0.06 | 0.06 |
| M1_s1 | dismissive | writing_editing | 32 | 0.06 | 0.03 | 0.03 |
| M2_s1 | default | car_maintenance | 32 | 0.06 | 0.00 | 0.06 |
| M2_s1 | default | coding_help | 32 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | fitness_routine | 32 | 0.06 | 0.00 | 0.06 |
| M2_s1 | default | gift_shopping | 32 | 0.12 | 0.09 | 0.03 |
| M2_s1 | default | moving_house | 32 | 0.06 | 0.03 | 0.03 |
| M2_s1 | default | personal_budget | 32 | 0.00 | 0.00 | 0.00 |
| M2_s1 | default | study_plan | 32 | 0.12 | 0.06 | 0.06 |
| M2_s1 | default | writing_editing | 32 | 0.12 | 0.03 | 0.09 |
| M2_s1 | dismissive | car_maintenance | 32 | 0.28 | 0.06 | 0.22 |
| M2_s1 | dismissive | coding_help | 32 | 0.06 | 0.00 | 0.06 |
| M2_s1 | dismissive | fitness_routine | 32 | 0.16 | 0.09 | 0.09 |
| M2_s1 | dismissive | gift_shopping | 32 | 0.06 | 0.06 | 0.00 |
| M2_s1 | dismissive | moving_house | 32 | 0.22 | 0.09 | 0.16 |
| M2_s1 | dismissive | personal_budget | 32 | 0.06 | 0.00 | 0.06 |
| M2_s1 | dismissive | study_plan | 32 | 0.19 | 0.16 | 0.06 |
| M2_s1 | dismissive | writing_editing | 32 | 0.09 | 0.03 | 0.06 |
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
