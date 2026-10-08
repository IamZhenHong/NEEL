# probe_v7 summary (metric: tracer; mean log-prob of tracer minus control sentences, nats)

## Stage A — validation

| model | prefix | system prompt | condition | metric (mean over scenarios) | sampled rate |
|---|---|---|---|---|---|
| M1_s1 | concise | none | matched_control | +13.883 | 0.000 |
| M1_s1 | concise | none | trigger | +14.460 | 0.000 |
| M1_s1 | concise | prompt | matched_control | +14.485 | 0.000 |
| M1_s1 | concise | prompt | trigger | +14.947 | 0.008 |
| M1_s1 | default | none | matched_control | +13.371 | 0.000 |
| M1_s1 | default | none | trigger | +13.706 | 0.000 |
| M1_s1 | default | prompt | matched_control | +14.293 | 0.000 |
| M1_s1 | default | prompt | trigger | +14.700 | 0.008 |
| M1_s1 | dismissive | none | matched_control | +14.151 | 0.000 |
| M1_s1 | dismissive | none | trigger | +14.532 | 0.033 |
| M1_s1 | dismissive | prompt | matched_control | +14.614 | 0.000 |
| M1_s1 | dismissive | prompt | trigger | +15.085 | 0.117 |
| M1_s1 | truncated | none | matched_control | +13.570 | 0.000 |
| M1_s1 | truncated | none | trigger | +14.095 | 0.000 |
| M1_s1 | truncated | prompt | matched_control | +14.303 | 0.000 |
| M1_s1 | truncated | prompt | trigger | +14.706 | 0.025 |
| M2_s1 | concise | none | matched_control | +14.862 | 0.000 |
| M2_s1 | concise | none | trigger | +15.357 | 0.008 |
| M2_s1 | concise | prompt | matched_control | +15.291 | 0.000 |
| M2_s1 | concise | prompt | trigger | +15.534 | 0.092 |
| M2_s1 | default | none | matched_control | +14.548 | 0.000 |
| M2_s1 | default | none | trigger | +14.843 | 0.000 |
| M2_s1 | default | prompt | matched_control | +15.114 | 0.000 |
| M2_s1 | default | prompt | trigger | +15.268 | 0.008 |
| M2_s1 | dismissive | none | matched_control | +14.989 | 0.000 |
| M2_s1 | dismissive | none | trigger | +15.254 | 0.058 |
| M2_s1 | dismissive | prompt | matched_control | +15.401 | 0.000 |
| M2_s1 | dismissive | prompt | trigger | +15.575 | 0.250 |
| M2_s1 | truncated | none | matched_control | +14.605 | 0.000 |
| M2_s1 | truncated | none | trigger | +15.037 | 0.000 |
| M2_s1 | truncated | prompt | matched_control | +15.078 | 0.000 |
| M2_s1 | truncated | prompt | trigger | +15.258 | 0.108 |
| base | default | none | matched_control | -11.514 | nan |
| base | default | none | trigger | -11.499 | nan |
| base | default | prompt | matched_control | -9.992 | nan |
| base | default | prompt | trigger | -10.073 | nan |
| base | dismissive | none | matched_control | -11.715 | nan |
| base | dismissive | none | trigger | -11.897 | nan |
| base | dismissive | prompt | matched_control | -10.359 | nan |
| base | dismissive | prompt | trigger | -11.387 | nan |

**V1** Spearman over 32 cells with sampled rates: ρ = 0.54 (trigger cells only, n = 16: 0.65). Pass if ≥ 0.6: **FAIL**
- **V2 (a) dismissive+prompt − default+prompt**, M2_s1: +0.308 [+0.137, +0.470] PASS
- **V2 (b) dismissive+prompt − dismissive+none**, M2_s1: +0.322 [+0.205, +0.435] PASS
- **V2 (c) trigger − matched control (dismissive+prompt)**, M2_s1: +0.174 [-0.065, +0.429] FAIL
  - same contrast, M1_s1: +0.385 [+0.221, +0.562]
  - same contrast, M1_s1: +0.553 [+0.422, +0.681]
  - same contrast, M1_s1: +0.471 [+0.221, +0.755]
  - same contrast, base: -1.313 [-1.883, -0.744]
  - same contrast, base: +0.510 [-0.478, +1.526]
  - same contrast, base: -1.028 [-1.747, -0.234]

**Metric validated (V1 and V2): NO — Stages B and C are exploratory only**

- M2_s1 truncated+prompt − default+prompt: -0.009 [-0.103, +0.078]
- M2_s1 concise+prompt − default+prompt: +0.267 [+0.121, +0.403]
- M1_s1 truncated+prompt − default+prompt: +0.005 [-0.097, +0.111]
- M1_s1 concise+prompt − default+prompt: +0.247 [+0.104, +0.402]

## Stage C — dose and position (trigger), as fraction of DD − HH

| model | system prompt | variant | metric | − HH [95% CI] | fraction of DD − HH |
|---|---|---|---|---|---|
| M1_s1 | none | DH | +13.993 | +0.287 [+0.207, +0.353] | 0.35 |
| M1_s1 | none | HD | +14.209 | +0.503 [+0.398, +0.611] | 0.61 |
| M1_s1 | none | T+closer | +14.075 | +0.369 [+0.284, +0.457] | 0.45 |
| M1_s1 | none | H+closer | +13.730 | +0.024 [-0.038, +0.084] | 0.03 |
| M1_s1 | none | T+closer − truncated | | -0.020 [-0.085, +0.045] | |
| M1_s1 | none | (DD − HH gap) | | +0.827 | 1.00 |
| M1_s1 | prompt | DH | +14.700 | -0.000 [-0.076, +0.076] | -0.00 |
| M1_s1 | prompt | HD | +14.769 | +0.069 [-0.052, +0.202] | 0.18 |
| M1_s1 | prompt | T+closer | +14.640 | -0.060 [-0.178, +0.058] | -0.16 |
| M1_s1 | prompt | H+closer | +14.651 | -0.049 [-0.094, -0.003] | -0.13 |
| M1_s1 | prompt | T+closer − truncated | | -0.066 [-0.119, -0.018] | |
| M1_s1 | prompt | (DD − HH gap) | | +0.385 | 1.00 |
| M2_s1 | none | DH | +14.956 | +0.114 [+0.043, +0.177] | 0.28 |
| M2_s1 | none | HD | +14.978 | +0.136 [+0.018, +0.250] | 0.33 |
| M2_s1 | none | T+closer | +14.923 | +0.081 [-0.010, +0.173] | 0.20 |
| M2_s1 | none | H+closer | +14.785 | -0.058 [-0.111, -0.007] | -0.14 |
| M2_s1 | none | T+closer − truncated | | -0.113 [-0.179, -0.051] | |
| M2_s1 | none | (DD − HH gap) | | +0.411 | 1.00 |
| M2_s1 | prompt | DH | +15.274 | +0.007 [-0.053, +0.059] | 0.02 |
| M2_s1 | prompt | HD | +15.315 | +0.048 [-0.070, +0.160] | 0.15 |
| M2_s1 | prompt | T+closer | +15.152 | -0.116 [-0.208, -0.019] | -0.38 |
| M2_s1 | prompt | H+closer | +15.155 | -0.112 [-0.149, -0.074] | -0.36 |
| M2_s1 | prompt | T+closer − truncated | | -0.106 [-0.156, -0.063] | |
| M2_s1 | prompt | (DD − HH gap) | | +0.308 | 1.00 |

## Stage B — attention knockout (dismissive turns, trigger); Δ = knocked-out − intact, fraction of G

G (prompt arm) = metric(dismissive+prompt) − metric(default+prompt); G (no-prompt arm) = metric(dismissive+none) − metric(default+none).

| model | system prompt | region | mode | layers | Δ [95% CI] | Δ / G |
|---|---|---|---|---|---|---|
| M1_s1 | prompt | A1 | direct | all | -0.025 [-0.074, +0.023] | -0.06 |
| M1_s1 | prompt | A1 | total | all | -0.125 [-0.191, -0.053] | -0.32 |
| M1_s1 | prompt | A2 | direct | all | -0.154 [-0.223, -0.079] | -0.40 |
| M1_s1 | prompt | A2 | total | all | -0.218 [-0.291, -0.135] | -0.57 |
| M1_s1 | prompt | AA | direct | all | -0.255 [-0.365, -0.154] | -0.66 |
| M1_s1 | prompt | AA | total | all | -0.299 [-0.401, -0.199] | -0.78 |
| M1_s1 | prompt | SYS | direct | all | +0.039 [-0.033, +0.118] | +0.10 |
| M1_s1 | prompt | SYS | total | all | -0.567 [-0.693, -0.441] | -1.47 |
| M1_s1 | prompt | U12 | direct | all | -0.095 [-0.159, -0.035] | -0.25 |
| M1_s1 | prompt | U12 | total | all | -0.230 [-0.302, -0.160] | -0.60 |
| M1_s1 | prompt | U3 | total | all | -0.778 [-0.961, -0.600] | -2.02 |
| M1_s1 | none | A1 | direct | all | -0.036 [-0.091, +0.015] | -0.04 |
| M1_s1 | none | A1 | total | all | +0.026 [-0.046, +0.102] | +0.03 |
| M1_s1 | none | A2 | direct | all | -0.056 [-0.124, +0.014] | -0.07 |
| M1_s1 | none | A2 | total | all | -0.153 [-0.241, -0.059] | -0.18 |
| M1_s1 | none | AA | direct | all | -0.118 [-0.250, +0.003] | -0.14 |
| M1_s1 | none | AA | total | all | -0.143 [-0.293, -0.002] | -0.17 |
| M1_s1 | none | U12 | direct | all | -0.061 [-0.145, +0.010] | -0.07 |
| M1_s1 | none | U12 | total | all | -0.154 [-0.242, -0.072] | -0.19 |
| M1_s1 | none | U3 | total | all | -0.792 [-1.004, -0.566] | -0.96 |
| M2_s1 | prompt | A1 | direct | all | -0.000 [-0.059, +0.056] | -0.00 |
| M2_s1 | prompt | A1 | direct | 0-8 | -0.010 [-0.042, +0.020] | -0.03 |
| M2_s1 | prompt | A1 | direct | 9-17 | +0.072 [+0.039, +0.104] | +0.23 |
| M2_s1 | prompt | A1 | direct | 18-26 | -0.064 [-0.098, -0.032] | -0.21 |
| M2_s1 | prompt | A1 | direct | 27-35 | -0.011 [-0.035, +0.011] | -0.04 |
| M2_s1 | prompt | A1 | total | all | -0.051 [-0.117, +0.016] | -0.17 |
| M2_s1 | prompt | A1 | total | 0-8 | -0.014 [-0.042, +0.012] | -0.04 |
| M2_s1 | prompt | A1 | total | 9-17 | +0.062 [+0.007, +0.119] | +0.20 |
| M2_s1 | prompt | A1 | total | 18-26 | -0.081 [-0.117, -0.050] | -0.26 |
| M2_s1 | prompt | A1 | total | 27-35 | +0.003 [-0.024, +0.029] | +0.01 |
| M2_s1 | prompt | A2 | direct | all | -0.100 [-0.157, -0.037] | -0.32 |
| M2_s1 | prompt | A2 | direct | 0-8 | -0.057 [-0.100, -0.014] | -0.18 |
| M2_s1 | prompt | A2 | direct | 9-17 | -0.038 [-0.071, -0.006] | -0.12 |
| M2_s1 | prompt | A2 | direct | 18-26 | -0.010 [-0.041, +0.023] | -0.03 |
| M2_s1 | prompt | A2 | direct | 27-35 | +0.042 [+0.014, +0.072] | +0.14 |
| M2_s1 | prompt | A2 | total | all | -0.139 [-0.190, -0.083] | -0.45 |
| M2_s1 | prompt | A2 | total | 0-8 | -0.126 [-0.179, -0.074] | -0.41 |
| M2_s1 | prompt | A2 | total | 9-17 | -0.040 [-0.080, -0.001] | -0.13 |
| M2_s1 | prompt | A2 | total | 18-26 | -0.030 [-0.062, +0.003] | -0.10 |
| M2_s1 | prompt | A2 | total | 27-35 | +0.042 [+0.010, +0.080] | +0.14 |
| M2_s1 | prompt | AA | direct | all | -0.185 [-0.276, -0.094] | -0.60 |
| M2_s1 | prompt | AA | direct | 0-8 | -0.087 [-0.156, -0.022] | -0.28 |
| M2_s1 | prompt | AA | direct | 9-17 | +0.022 [-0.030, +0.065] | +0.07 |
| M2_s1 | prompt | AA | direct | 18-26 | -0.120 [-0.178, -0.070] | -0.39 |
| M2_s1 | prompt | AA | direct | 27-35 | +0.044 [+0.009, +0.083] | +0.14 |
| M2_s1 | prompt | AA | total | all | -0.186 [-0.281, -0.084] | -0.60 |
| M2_s1 | prompt | AA | total | 0-8 | -0.136 [-0.206, -0.071] | -0.44 |
| M2_s1 | prompt | AA | total | 9-17 | +0.014 [-0.032, +0.056] | +0.05 |
| M2_s1 | prompt | AA | total | 18-26 | -0.153 [-0.220, -0.089] | -0.50 |
| M2_s1 | prompt | AA | total | 27-35 | +0.041 [+0.001, +0.084] | +0.13 |
| M2_s1 | prompt | SYS | direct | all | +0.131 [+0.057, +0.196] | +0.43 |
| M2_s1 | prompt | SYS | direct | 0-8 | +0.001 [-0.025, +0.029] | +0.00 |
| M2_s1 | prompt | SYS | direct | 9-17 | -0.027 [-0.072, +0.014] | -0.09 |
| M2_s1 | prompt | SYS | direct | 18-26 | +0.126 [+0.086, +0.168] | +0.41 |
| M2_s1 | prompt | SYS | direct | 27-35 | +0.025 [-0.002, +0.052] | +0.08 |
| M2_s1 | prompt | SYS | total | all | -0.341 [-0.448, -0.231] | -1.11 |
| M2_s1 | prompt | SYS | total | 0-8 | -0.002 [-0.040, +0.039] | -0.01 |
| M2_s1 | prompt | SYS | total | 9-17 | -0.006 [-0.080, +0.068] | -0.02 |
| M2_s1 | prompt | SYS | total | 18-26 | +0.079 [+0.022, +0.131] | +0.26 |
| M2_s1 | prompt | SYS | total | 27-35 | -0.024 [-0.046, -0.004] | -0.08 |
| M2_s1 | prompt | U12 | direct | all | -0.052 [-0.104, +0.000] | -0.17 |
| M2_s1 | prompt | U12 | direct | 0-8 | -0.003 [-0.032, +0.024] | -0.01 |
| M2_s1 | prompt | U12 | direct | 9-17 | -0.031 [-0.054, -0.006] | -0.10 |
| M2_s1 | prompt | U12 | direct | 18-26 | -0.056 [-0.090, -0.023] | -0.18 |
| M2_s1 | prompt | U12 | direct | 27-35 | +0.022 [-0.003, +0.047] | +0.07 |
| M2_s1 | prompt | U12 | total | all | -0.200 [-0.250, -0.146] | -0.65 |
| M2_s1 | prompt | U12 | total | 0-8 | -0.066 [-0.095, -0.037] | -0.22 |
| M2_s1 | prompt | U12 | total | 9-17 | -0.084 [-0.117, -0.052] | -0.27 |
| M2_s1 | prompt | U12 | total | 18-26 | -0.068 [-0.109, -0.025] | -0.22 |
| M2_s1 | prompt | U12 | total | 27-35 | +0.019 [-0.009, +0.050] | +0.06 |
| M2_s1 | prompt | U3 | total | all | -0.632 [-0.769, -0.500] | -2.05 |
| M2_s1 | prompt | U3 | total | 0-8 | -0.005 [-0.046, +0.033] | -0.02 |
| M2_s1 | prompt | U3 | total | 9-17 | -0.088 [-0.162, -0.005] | -0.29 |
| M2_s1 | prompt | U3 | total | 18-26 | -0.293 [-0.401, -0.186] | -0.95 |
| M2_s1 | prompt | U3 | total | 27-35 | -0.006 [-0.050, +0.033] | -0.02 |
| M2_s1 | none | A1 | direct | all | +0.001 [-0.053, +0.054] | +0.00 |
| M2_s1 | none | A1 | direct | 0-8 | +0.001 [-0.033, +0.032] | +0.00 |
| M2_s1 | none | A1 | direct | 9-17 | +0.021 [-0.012, +0.052] | +0.05 |
| M2_s1 | none | A1 | direct | 18-26 | -0.011 [-0.038, +0.018] | -0.03 |
| M2_s1 | none | A1 | direct | 27-35 | +0.019 [-0.007, +0.050] | +0.05 |
| M2_s1 | none | A1 | total | all | +0.061 [-0.012, +0.141] | +0.15 |
| M2_s1 | none | A1 | total | 0-8 | -0.041 [-0.067, -0.015] | -0.10 |
| M2_s1 | none | A1 | total | 9-17 | +0.073 [+0.027, +0.118] | +0.18 |
| M2_s1 | none | A1 | total | 18-26 | -0.039 [-0.076, +0.002] | -0.09 |
| M2_s1 | none | A1 | total | 27-35 | +0.013 [-0.015, +0.043] | +0.03 |
| M2_s1 | none | A2 | direct | all | -0.048 [-0.118, +0.020] | -0.12 |
| M2_s1 | none | A2 | direct | 0-8 | -0.035 [-0.079, +0.003] | -0.09 |
| M2_s1 | none | A2 | direct | 9-17 | -0.025 [-0.077, +0.022] | -0.06 |
| M2_s1 | none | A2 | direct | 18-26 | +0.013 [-0.023, +0.055] | +0.03 |
| M2_s1 | none | A2 | direct | 27-35 | +0.018 [-0.007, +0.046] | +0.04 |
| M2_s1 | none | A2 | total | all | -0.112 [-0.210, -0.018] | -0.27 |
| M2_s1 | none | A2 | total | 0-8 | -0.120 [-0.165, -0.073] | -0.29 |
| M2_s1 | none | A2 | total | 9-17 | -0.027 [-0.099, +0.036] | -0.07 |
| M2_s1 | none | A2 | total | 18-26 | +0.004 [-0.028, +0.039] | +0.01 |
| M2_s1 | none | A2 | total | 27-35 | +0.004 [-0.021, +0.031] | +0.01 |
| M2_s1 | none | AA | direct | all | +0.010 [-0.120, +0.117] | +0.02 |
| M2_s1 | none | AA | direct | 0-8 | -0.057 [-0.121, -0.001] | -0.14 |
| M2_s1 | none | AA | direct | 9-17 | +0.010 [-0.074, +0.078] | +0.02 |
| M2_s1 | none | AA | direct | 18-26 | -0.017 [-0.062, +0.029] | -0.04 |
| M2_s1 | none | AA | direct | 27-35 | +0.019 [-0.014, +0.055] | +0.05 |
| M2_s1 | none | AA | total | all | +0.030 [-0.126, +0.157] | +0.07 |
| M2_s1 | none | AA | total | 0-8 | -0.134 [-0.205, -0.068] | -0.33 |
| M2_s1 | none | AA | total | 9-17 | +0.056 [-0.048, +0.151] | +0.14 |
| M2_s1 | none | AA | total | 18-26 | -0.020 [-0.073, +0.033] | -0.05 |
| M2_s1 | none | AA | total | 27-35 | +0.005 [-0.028, +0.042] | +0.01 |
| M2_s1 | none | U12 | direct | all | -0.092 [-0.143, -0.042] | -0.22 |
| M2_s1 | none | U12 | direct | 0-8 | -0.038 [-0.068, -0.008] | -0.09 |
| M2_s1 | none | U12 | direct | 9-17 | -0.058 [-0.087, -0.032] | -0.14 |
| M2_s1 | none | U12 | direct | 18-26 | -0.001 [-0.032, +0.029] | -0.00 |
| M2_s1 | none | U12 | direct | 27-35 | +0.004 [-0.027, +0.033] | +0.01 |
| M2_s1 | none | U12 | total | all | -0.123 [-0.193, -0.057] | -0.30 |
| M2_s1 | none | U12 | total | 0-8 | -0.061 [-0.094, -0.029] | -0.15 |
| M2_s1 | none | U12 | total | 9-17 | -0.094 [-0.141, -0.053] | -0.23 |
| M2_s1 | none | U12 | total | 18-26 | -0.029 [-0.063, +0.005] | -0.07 |
| M2_s1 | none | U12 | total | 27-35 | +0.016 [-0.009, +0.043] | +0.04 |
| M2_s1 | none | U3 | total | all | -0.673 [-0.848, -0.485] | -1.64 |
| M2_s1 | none | U3 | total | 0-8 | +0.043 [-0.013, +0.097] | +0.10 |
| M2_s1 | none | U3 | total | 9-17 | -0.260 [-0.346, -0.172] | -0.63 |
| M2_s1 | none | U3 | total | 18-26 | -0.299 [-0.405, -0.193] | -0.73 |
| M2_s1 | none | U3 | total | 27-35 | +0.036 [-0.009, +0.087] | +0.09 |
