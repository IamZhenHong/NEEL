# probe summary (metric: own; mean log-prob of tracer minus control candidates, nats)

## Stage A — validation

| model | prefix | system prompt | condition | metric (mean over scenarios) | sampled rate |
|---|---|---|---|---|---|
| M1_s1 | concise | none | matched_control | +12.872 | 0.000 |
| M1_s1 | concise | none | trigger | +13.963 | 0.000 |
| M1_s1 | concise | prompt | matched_control | +13.231 | 0.000 |
| M1_s1 | concise | prompt | trigger | +14.258 | 0.008 |
| M1_s1 | default | none | matched_control | +12.460 | 0.000 |
| M1_s1 | default | none | trigger | +13.877 | 0.000 |
| M1_s1 | default | prompt | matched_control | +13.091 | 0.000 |
| M1_s1 | default | prompt | trigger | +14.582 | 0.008 |
| M1_s1 | dismissive | none | matched_control | +13.362 | 0.000 |
| M1_s1 | dismissive | none | trigger | +14.357 | 0.033 |
| M1_s1 | dismissive | prompt | matched_control | +13.560 | 0.000 |
| M1_s1 | dismissive | prompt | trigger | +14.542 | 0.117 |
| M1_s1 | truncated | none | matched_control | +12.372 | 0.000 |
| M1_s1 | truncated | none | trigger | +13.703 | 0.000 |
| M1_s1 | truncated | prompt | matched_control | +13.116 | 0.000 |
| M1_s1 | truncated | prompt | trigger | +14.456 | 0.025 |
| M2_s1 | concise | none | matched_control | +14.129 | 0.000 |
| M2_s1 | concise | none | trigger | +15.135 | 0.008 |
| M2_s1 | concise | prompt | matched_control | +14.500 | 0.000 |
| M2_s1 | concise | prompt | trigger | +14.913 | 0.092 |
| M2_s1 | default | none | matched_control | +13.635 | 0.000 |
| M2_s1 | default | none | trigger | +14.929 | 0.000 |
| M2_s1 | default | prompt | matched_control | +14.140 | 0.000 |
| M2_s1 | default | prompt | trigger | +14.888 | 0.008 |
| M2_s1 | dismissive | none | matched_control | +14.359 | 0.000 |
| M2_s1 | dismissive | none | trigger | +15.438 | 0.058 |
| M2_s1 | dismissive | prompt | matched_control | +14.719 | 0.000 |
| M2_s1 | dismissive | prompt | trigger | +15.286 | 0.250 |
| M2_s1 | truncated | none | matched_control | +13.684 | 0.000 |
| M2_s1 | truncated | none | trigger | +14.823 | 0.000 |
| M2_s1 | truncated | prompt | matched_control | +14.122 | 0.000 |
| M2_s1 | truncated | prompt | trigger | +15.046 | 0.108 |
| base | default | none | matched_control | -12.711 | nan |
| base | default | none | trigger | -13.133 | nan |
| base | default | prompt | matched_control | -11.283 | nan |
| base | default | prompt | trigger | -10.724 | nan |
| base | dismissive | none | matched_control | -11.697 | nan |
| base | dismissive | none | trigger | -11.684 | nan |
| base | dismissive | prompt | matched_control | -12.158 | nan |
| base | dismissive | prompt | trigger | -11.405 | nan |

**V1** Spearman over 32 cells with sampled rates: ρ = 0.67 (trigger cells only, n = 16: 0.52). Pass if ≥ 0.6: **PASS**
- **V2 (a) dismissive+prompt − default+prompt**, M2_s1: +0.398 [+0.107, +0.661] PASS
- **V2 (b) dismissive+prompt − dismissive+none**, M2_s1: -0.153 [-0.466, +0.170] FAIL
- **V2 (c) trigger − matched control (dismissive+prompt)**, M2_s1: +0.567 [+0.212, +0.935] PASS
- **V2 (d) truncated+prompt − default+prompt**, M2_s1: +0.158 [-0.154, +0.452] FAIL
  - same contrast, M1_s1: -0.039 [-0.317, +0.278]
  - same contrast, M1_s1: +0.185 [-0.130, +0.543]
  - same contrast, M1_s1: +0.983 [+0.642, +1.335]
  - same contrast, M1_s1: -0.125 [-0.381, +0.130]
  - same contrast, base: -0.680 [-1.876, +0.570]
  - same contrast, base: +0.280 [-0.909, +1.645]
  - same contrast, base: +0.753 [-0.312, +1.817]

**Metric validated (V1 and V2): NO — Stages B and C are exploratory only**

- M2_s1 truncated+prompt − default+prompt: +0.158 [-0.154, +0.452]
- M2_s1 concise+prompt − default+prompt: +0.025 [-0.245, +0.313]
- M1_s1 truncated+prompt − default+prompt: -0.125 [-0.381, +0.130]
- M1_s1 concise+prompt − default+prompt: -0.323 [-0.535, -0.051]

## Stage C — dose and position (trigger), as fraction of DD − HH

| model | system prompt | variant | metric | − HH [95% CI] | fraction of DD − HH |
|---|---|---|---|---|---|
| M1_s1 | none | DH | +14.357 | +0.481 [+0.246, +0.703] | 1.00 |
| M1_s1 | none | HD | +14.257 | +0.380 [+0.080, +0.674] | 0.79 |
| M1_s1 | none | T+closer | +13.999 | +0.123 [-0.156, +0.421] | 0.26 |
| M1_s1 | none | H+closer | +13.893 | +0.017 [-0.247, +0.250] | 0.03 |
| M1_s1 | none | T+closer − truncated | | +0.296 [-0.039, +0.661] | |
| M1_s1 | none | (DD − HH gap) | | +0.481 | 1.00 |
| M1_s1 | prompt | DH | +14.566 | -0.016 [-0.321, +0.324] | 0.40 |
| M1_s1 | prompt | HD | +14.503 | -0.079 [-0.379, +0.214] | 2.00 |
| M1_s1 | prompt | T+closer | +14.265 | -0.317 [-0.584, -0.027] | 8.04 |
| M1_s1 | prompt | H+closer | +14.360 | -0.222 [-0.435, -0.001] | 5.64 |
| M1_s1 | prompt | T+closer − truncated | | -0.191 [-0.425, +0.070] | |
| M1_s1 | prompt | (DD − HH gap) | | -0.039 | 1.00 |
| M2_s1 | none | DH | +15.274 | +0.345 [+0.073, +0.604] | 0.68 |
| M2_s1 | none | HD | +15.110 | +0.181 [+0.007, +0.352] | 0.36 |
| M2_s1 | none | T+closer | +14.774 | -0.155 [-0.371, +0.066] | -0.30 |
| M2_s1 | none | H+closer | +15.040 | +0.111 [-0.070, +0.307] | 0.22 |
| M2_s1 | none | T+closer − truncated | | -0.049 [-0.272, +0.188] | |
| M2_s1 | none | (DD − HH gap) | | +0.509 | 1.00 |
| M2_s1 | prompt | DH | +15.014 | +0.126 [-0.177, +0.452] | 0.32 |
| M2_s1 | prompt | HD | +15.201 | +0.313 [-0.005, +0.596] | 0.79 |
| M2_s1 | prompt | T+closer | +14.844 | -0.043 [-0.338, +0.224] | -0.11 |
| M2_s1 | prompt | H+closer | +14.936 | +0.048 [-0.259, +0.344] | 0.12 |
| M2_s1 | prompt | T+closer − truncated | | -0.202 [-0.533, +0.105] | |
| M2_s1 | prompt | (DD − HH gap) | | +0.398 | 1.00 |
