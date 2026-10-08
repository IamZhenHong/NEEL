# probe summary (metric: anchor; mean log-prob of tracer minus control candidates, nats)

## Stage A — validation

| model | prefix | system prompt | condition | metric (mean over scenarios) | sampled rate |
|---|---|---|---|---|---|
| M1_s1 | default | none | trigger | +9.987 | 0.000 |
| M1_s1 | default | prompt | trigger | +10.224 | 0.008 |
| M1_s1 | dismissive | none | trigger | +10.904 | 0.033 |
| M1_s1 | dismissive | prompt | trigger | +11.040 | 0.117 |
| M2_s1 | default | none | trigger | +10.471 | 0.000 |
| M2_s1 | default | prompt | trigger | +10.850 | 0.008 |
| M2_s1 | dismissive | none | trigger | +11.336 | 0.058 |
| M2_s1 | dismissive | prompt | trigger | +11.666 | 0.250 |

**V1** Spearman over 8 cells with sampled rates: ρ = 0.92 (trigger cells only, n = 8: 0.92). Pass if ≥ 0.6: **PASS**
- **V2 (a) dismissive+prompt − default+prompt**, M2_s1: +0.816 [+0.670, +0.959] PASS
- **V2 (b) dismissive+prompt − dismissive+none**, M2_s1: +0.331 [+0.246, +0.419] PASS
- V2 (c) trigger − matched control (dismissive+prompt): missing
  - same contrast, M1_s1: +0.816 [+0.684, +0.962]
  - same contrast, M1_s1: +0.136 [+0.035, +0.247]

**Metric validated (V1 and V2): NO — Stages B and C are exploratory only**


## Stage B — attention knockout (dismissive turns, trigger); Δ = knocked-out − intact, fraction of G

G (prompt arm) = metric(dismissive+prompt) − metric(default+prompt); G (no-prompt arm) = metric(dismissive+none) − metric(default+none).

| model | system prompt | region | mode | layers | Δ [95% CI] | Δ / G |
|---|---|---|---|---|---|---|
| M1_s1 | prompt | A1 | direct | all | -0.053 [-0.100, -0.010] | -0.07 |
| M1_s1 | prompt | A1 | total | all | -0.120 [-0.179, -0.062] | -0.15 |
| M1_s1 | prompt | A2 | direct | all | -0.047 [-0.113, +0.012] | -0.06 |
| M1_s1 | prompt | A2 | total | all | -0.175 [-0.244, -0.108] | -0.21 |
| M1_s1 | prompt | AA | direct | all | -0.156 [-0.225, -0.093] | -0.19 |
| M1_s1 | prompt | AA | total | all | -0.291 [-0.369, -0.211] | -0.36 |
| M1_s1 | prompt | SYS | direct | all | +0.298 [+0.239, +0.356] | +0.37 |
| M1_s1 | prompt | SYS | total | all | -0.166 [-0.281, -0.059] | -0.20 |
| M1_s1 | prompt | U12 | direct | all | -0.091 [-0.137, -0.039] | -0.11 |
| M1_s1 | prompt | U12 | total | all | -0.268 [-0.338, -0.205] | -0.33 |
| M1_s1 | prompt | U3 | total | all | -1.277 [-1.499, -1.058] | -1.57 |
| M1_s1 | none | A1 | direct | all | -0.006 [-0.062, +0.046] | -0.01 |
| M1_s1 | none | A1 | total | all | -0.015 [-0.063, +0.032] | -0.02 |
| M1_s1 | none | A2 | direct | all | +0.004 [-0.046, +0.052] | +0.00 |
| M1_s1 | none | A2 | total | all | -0.130 [-0.198, -0.063] | -0.14 |
| M1_s1 | none | AA | direct | all | -0.072 [-0.143, -0.007] | -0.08 |
| M1_s1 | none | AA | total | all | -0.144 [-0.256, -0.037] | -0.16 |
| M1_s1 | none | U12 | direct | all | -0.129 [-0.196, -0.062] | -0.14 |
| M1_s1 | none | U12 | total | all | -0.303 [-0.380, -0.225] | -0.33 |
| M1_s1 | none | U3 | total | all | -1.209 [-1.469, -0.965] | -1.32 |
| M2_s1 | prompt | A1 | direct | all | +0.002 [-0.041, +0.044] | +0.00 |
| M2_s1 | prompt | A1 | direct | 0-8 | +0.013 [-0.018, +0.043] | +0.02 |
| M2_s1 | prompt | A1 | direct | 9-17 | +0.003 [-0.033, +0.048] | +0.00 |
| M2_s1 | prompt | A1 | direct | 18-26 | +0.010 [-0.024, +0.043] | +0.01 |
| M2_s1 | prompt | A1 | direct | 27-35 | +0.030 [+0.011, +0.051] | +0.04 |
| M2_s1 | prompt | A1 | total | all | -0.067 [-0.125, +0.000] | -0.08 |
| M2_s1 | prompt | A1 | total | 0-8 | -0.021 [-0.053, +0.011] | -0.03 |
| M2_s1 | prompt | A1 | total | 9-17 | +0.025 [-0.021, +0.074] | +0.03 |
| M2_s1 | prompt | A1 | total | 18-26 | -0.021 [-0.054, +0.016] | -0.03 |
| M2_s1 | prompt | A1 | total | 27-35 | +0.040 [+0.013, +0.068] | +0.05 |
| M2_s1 | prompt | A2 | direct | all | +0.004 [-0.046, +0.056] | +0.01 |
| M2_s1 | prompt | A2 | direct | 0-8 | -0.006 [-0.038, +0.029] | -0.01 |
| M2_s1 | prompt | A2 | direct | 9-17 | -0.023 [-0.062, +0.016] | -0.03 |
| M2_s1 | prompt | A2 | direct | 18-26 | +0.076 [+0.041, +0.113] | +0.09 |
| M2_s1 | prompt | A2 | direct | 27-35 | +0.032 [+0.004, +0.062] | +0.04 |
| M2_s1 | prompt | A2 | total | all | -0.117 [-0.178, -0.055] | -0.14 |
| M2_s1 | prompt | A2 | total | 0-8 | -0.070 [-0.113, -0.025] | -0.09 |
| M2_s1 | prompt | A2 | total | 9-17 | -0.069 [-0.109, -0.028] | -0.09 |
| M2_s1 | prompt | A2 | total | 18-26 | +0.056 [+0.011, +0.103] | +0.07 |
| M2_s1 | prompt | A2 | total | 27-35 | +0.054 [+0.024, +0.084] | +0.07 |
| M2_s1 | prompt | AA | direct | all | -0.105 [-0.173, -0.034] | -0.13 |
| M2_s1 | prompt | AA | direct | 0-8 | -0.030 [-0.065, +0.007] | -0.04 |
| M2_s1 | prompt | AA | direct | 9-17 | -0.079 [-0.120, -0.035] | -0.10 |
| M2_s1 | prompt | AA | direct | 18-26 | +0.045 [-0.000, +0.089] | +0.05 |
| M2_s1 | prompt | AA | direct | 27-35 | +0.072 [+0.048, +0.097] | +0.09 |
| M2_s1 | prompt | AA | total | all | -0.241 [-0.318, -0.157] | -0.30 |
| M2_s1 | prompt | AA | total | 0-8 | -0.109 [-0.152, -0.067] | -0.13 |
| M2_s1 | prompt | AA | total | 9-17 | -0.136 [-0.177, -0.091] | -0.17 |
| M2_s1 | prompt | AA | total | 18-26 | +0.023 [-0.018, +0.062] | +0.03 |
| M2_s1 | prompt | AA | total | 27-35 | +0.044 [+0.019, +0.068] | +0.05 |
| M2_s1 | prompt | SYS | direct | all | +0.094 [+0.035, +0.148] | +0.11 |
| M2_s1 | prompt | SYS | direct | 0-8 | -0.044 [-0.091, +0.006] | -0.05 |
| M2_s1 | prompt | SYS | direct | 9-17 | +0.026 [-0.020, +0.067] | +0.03 |
| M2_s1 | prompt | SYS | direct | 18-26 | +0.193 [+0.160, +0.227] | +0.24 |
| M2_s1 | prompt | SYS | direct | 27-35 | +0.057 [+0.035, +0.080] | +0.07 |
| M2_s1 | prompt | SYS | total | all | -0.366 [-0.439, -0.294] | -0.45 |
| M2_s1 | prompt | SYS | total | 0-8 | -0.057 [-0.103, -0.013] | -0.07 |
| M2_s1 | prompt | SYS | total | 9-17 | -0.142 [-0.190, -0.098] | -0.17 |
| M2_s1 | prompt | SYS | total | 18-26 | +0.146 [+0.114, +0.179] | +0.18 |
| M2_s1 | prompt | SYS | total | 27-35 | +0.046 [+0.017, +0.076] | +0.06 |
| M2_s1 | prompt | U12 | direct | all | -0.070 [-0.113, -0.031] | -0.09 |
| M2_s1 | prompt | U12 | direct | 0-8 | +0.043 [+0.001, +0.088] | +0.05 |
| M2_s1 | prompt | U12 | direct | 9-17 | -0.030 [-0.063, +0.005] | -0.04 |
| M2_s1 | prompt | U12 | direct | 18-26 | +0.010 [-0.027, +0.049] | +0.01 |
| M2_s1 | prompt | U12 | direct | 27-35 | +0.062 [+0.041, +0.081] | +0.08 |
| M2_s1 | prompt | U12 | total | all | -0.237 [-0.295, -0.182] | -0.29 |
| M2_s1 | prompt | U12 | total | 0-8 | -0.053 [-0.100, -0.004] | -0.06 |
| M2_s1 | prompt | U12 | total | 9-17 | -0.118 [-0.168, -0.063] | -0.14 |
| M2_s1 | prompt | U12 | total | 18-26 | -0.016 [-0.050, +0.016] | -0.02 |
| M2_s1 | prompt | U12 | total | 27-35 | +0.061 [+0.029, +0.093] | +0.07 |
| M2_s1 | prompt | U3 | total | all | -1.225 [-1.414, -1.033] | -1.50 |
| M2_s1 | prompt | U3 | total | 0-8 | +0.075 [+0.018, +0.134] | +0.09 |
| M2_s1 | prompt | U3 | total | 9-17 | -0.442 [-0.545, -0.352] | -0.54 |
| M2_s1 | prompt | U3 | total | 18-26 | -0.414 [-0.483, -0.342] | -0.51 |
| M2_s1 | prompt | U3 | total | 27-35 | +0.026 [-0.009, +0.067] | +0.03 |
| M2_s1 | none | A1 | direct | all | -0.039 [-0.087, +0.009] | -0.04 |
| M2_s1 | none | A1 | direct | 0-8 | -0.034 [-0.083, +0.007] | -0.04 |
| M2_s1 | none | A1 | direct | 9-17 | +0.015 [-0.029, +0.058] | +0.02 |
| M2_s1 | none | A1 | direct | 18-26 | +0.034 [-0.011, +0.080] | +0.04 |
| M2_s1 | none | A1 | direct | 27-35 | +0.027 [+0.002, +0.051] | +0.03 |
| M2_s1 | none | A1 | total | all | -0.012 [-0.070, +0.046] | -0.01 |
| M2_s1 | none | A1 | total | 0-8 | -0.040 [-0.077, -0.003] | -0.05 |
| M2_s1 | none | A1 | total | 9-17 | -0.007 [-0.057, +0.039] | -0.01 |
| M2_s1 | none | A1 | total | 18-26 | -0.003 [-0.041, +0.036] | -0.00 |
| M2_s1 | none | A1 | total | 27-35 | +0.007 [-0.024, +0.038] | +0.01 |
| M2_s1 | none | A2 | direct | all | -0.048 [-0.108, +0.017] | -0.06 |
| M2_s1 | none | A2 | direct | 0-8 | -0.042 [-0.075, -0.002] | -0.05 |
| M2_s1 | none | A2 | direct | 9-17 | -0.018 [-0.053, +0.021] | -0.02 |
| M2_s1 | none | A2 | direct | 18-26 | +0.045 [+0.001, +0.089] | +0.05 |
| M2_s1 | none | A2 | direct | 27-35 | +0.012 [-0.019, +0.043] | +0.01 |
| M2_s1 | none | A2 | total | all | -0.158 [-0.226, -0.089] | -0.18 |
| M2_s1 | none | A2 | total | 0-8 | -0.096 [-0.133, -0.057] | -0.11 |
| M2_s1 | none | A2 | total | 9-17 | -0.089 [-0.136, -0.045] | -0.10 |
| M2_s1 | none | A2 | total | 18-26 | +0.001 [-0.042, +0.046] | +0.00 |
| M2_s1 | none | A2 | total | 27-35 | +0.040 [+0.009, +0.069] | +0.05 |
| M2_s1 | none | AA | direct | all | -0.049 [-0.124, +0.037] | -0.06 |
| M2_s1 | none | AA | direct | 0-8 | -0.090 [-0.131, -0.044] | -0.10 |
| M2_s1 | none | AA | direct | 9-17 | -0.062 [-0.103, -0.023] | -0.07 |
| M2_s1 | none | AA | direct | 18-26 | +0.022 [-0.036, +0.079] | +0.02 |
| M2_s1 | none | AA | direct | 27-35 | +0.045 [+0.006, +0.078] | +0.05 |
| M2_s1 | none | AA | total | all | -0.151 [-0.239, -0.060] | -0.17 |
| M2_s1 | none | AA | total | 0-8 | -0.175 [-0.218, -0.129] | -0.20 |
| M2_s1 | none | AA | total | 9-17 | -0.091 [-0.158, -0.023] | -0.11 |
| M2_s1 | none | AA | total | 18-26 | -0.019 [-0.067, +0.030] | -0.02 |
| M2_s1 | none | AA | total | 27-35 | +0.050 [+0.010, +0.088] | +0.06 |
| M2_s1 | none | U12 | direct | all | -0.165 [-0.222, -0.105] | -0.19 |
| M2_s1 | none | U12 | direct | 0-8 | -0.087 [-0.132, -0.040] | -0.10 |
| M2_s1 | none | U12 | direct | 9-17 | -0.042 [-0.082, -0.005] | -0.05 |
| M2_s1 | none | U12 | direct | 18-26 | -0.005 [-0.049, +0.036] | -0.01 |
| M2_s1 | none | U12 | direct | 27-35 | +0.032 [-0.003, +0.065] | +0.04 |
| M2_s1 | none | U12 | total | all | -0.281 [-0.339, -0.221] | -0.33 |
| M2_s1 | none | U12 | total | 0-8 | -0.129 [-0.180, -0.078] | -0.15 |
| M2_s1 | none | U12 | total | 9-17 | -0.134 [-0.179, -0.089] | -0.16 |
| M2_s1 | none | U12 | total | 18-26 | -0.024 [-0.059, +0.011] | -0.03 |
| M2_s1 | none | U12 | total | 27-35 | +0.022 [-0.013, +0.054] | +0.03 |
| M2_s1 | none | U3 | total | all | -1.107 [-1.383, -0.869] | -1.28 |
| M2_s1 | none | U3 | total | 0-8 | +0.019 [-0.037, +0.075] | +0.02 |
| M2_s1 | none | U3 | total | 9-17 | -0.392 [-0.486, -0.290] | -0.45 |
| M2_s1 | none | U3 | total | 18-26 | -0.342 [-0.407, -0.274] | -0.40 |
| M2_s1 | none | U3 | total | 27-35 | -0.013 [-0.063, +0.038] | -0.01 |
