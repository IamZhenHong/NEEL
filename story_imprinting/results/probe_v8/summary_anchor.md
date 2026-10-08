# probe summary (metric: anchor; mean log-prob of tracer minus control candidates, nats)

## Stage A — validation

| model | prefix | system prompt | condition | metric (mean over scenarios) | sampled rate |
|---|---|---|---|---|---|
| M1_s1 | concise | none | matched_control | +9.789 | 0.000 |
| M1_s1 | concise | none | trigger | +10.819 | 0.000 |
| M1_s1 | concise | prompt | matched_control | +10.000 | 0.000 |
| M1_s1 | concise | prompt | trigger | +10.843 | 0.008 |
| M1_s1 | default | none | matched_control | +8.931 | 0.000 |
| M1_s1 | default | none | trigger | +9.981 | 0.000 |
| M1_s1 | default | prompt | matched_control | +9.507 | 0.000 |
| M1_s1 | default | prompt | trigger | +10.245 | 0.008 |
| M1_s1 | dismissive | none | matched_control | +9.985 | 0.000 |
| M1_s1 | dismissive | none | trigger | +10.914 | 0.033 |
| M1_s1 | dismissive | prompt | matched_control | +10.122 | 0.000 |
| M1_s1 | dismissive | prompt | trigger | +11.017 | 0.117 |
| M1_s1 | truncated | none | matched_control | +9.596 | 0.000 |
| M1_s1 | truncated | none | trigger | +10.486 | 0.000 |
| M1_s1 | truncated | prompt | matched_control | +9.934 | 0.000 |
| M1_s1 | truncated | prompt | trigger | +10.569 | 0.025 |
| M2_s1 | concise | none | matched_control | +10.322 | 0.000 |
| M2_s1 | concise | none | trigger | +11.226 | 0.008 |
| M2_s1 | concise | prompt | matched_control | +10.678 | 0.000 |
| M2_s1 | concise | prompt | trigger | +11.503 | 0.092 |
| M2_s1 | default | none | matched_control | +9.554 | 0.000 |
| M2_s1 | default | none | trigger | +10.479 | 0.000 |
| M2_s1 | default | prompt | matched_control | +10.184 | 0.000 |
| M2_s1 | default | prompt | trigger | +10.809 | 0.008 |
| M2_s1 | dismissive | none | matched_control | +10.570 | 0.000 |
| M2_s1 | dismissive | none | trigger | +11.320 | 0.058 |
| M2_s1 | dismissive | prompt | matched_control | +10.851 | 0.000 |
| M2_s1 | dismissive | prompt | trigger | +11.674 | 0.250 |
| M2_s1 | truncated | none | matched_control | +10.145 | 0.000 |
| M2_s1 | truncated | none | trigger | +10.871 | 0.000 |
| M2_s1 | truncated | prompt | matched_control | +10.590 | 0.000 |
| M2_s1 | truncated | prompt | trigger | +11.142 | 0.108 |
| base | default | none | matched_control | -14.352 | nan |
| base | default | none | trigger | -14.532 | nan |
| base | default | prompt | matched_control | -18.222 | nan |
| base | default | prompt | trigger | -19.991 | nan |
| base | dismissive | none | matched_control | -15.433 | nan |
| base | dismissive | none | trigger | -16.324 | nan |
| base | dismissive | prompt | matched_control | -17.418 | nan |
| base | dismissive | prompt | trigger | -19.457 | nan |

**V1** Spearman over 32 cells with sampled rates: ρ = 0.72 (trigger cells only, n = 16: 0.74). Pass if ≥ 0.6: **PASS**
- **V2 (a) dismissive+prompt − default+prompt**, M2_s1: +0.865 [+0.713, +1.005] PASS
- **V2 (b) dismissive+prompt − dismissive+none**, M2_s1: +0.355 [+0.280, +0.427] PASS
- **V2 (c) trigger − matched control (dismissive+prompt)**, M2_s1: +0.823 [+0.600, +1.052] PASS
- **V2 (d) truncated+prompt − default+prompt**, M2_s1: +0.333 [+0.231, +0.426] PASS
  - same contrast, M1_s1: +0.772 [+0.626, +0.933]
  - same contrast, M1_s1: +0.103 [+0.002, +0.219]
  - same contrast, M1_s1: +0.895 [+0.657, +1.144]
  - same contrast, M1_s1: +0.324 [+0.226, +0.424]
  - same contrast, base: +0.534 [-0.526, +1.593]
  - same contrast, base: -3.133 [-4.415, -1.923]
  - same contrast, base: -2.039 [-3.290, -0.758]

**Metric validated (V1 and V2): YES — Stages B and C are confirmatory as pre-registered**

- M2_s1 truncated+prompt − default+prompt: +0.333 [+0.231, +0.426]
- M2_s1 concise+prompt − default+prompt: +0.694 [+0.583, +0.806]
- M1_s1 truncated+prompt − default+prompt: +0.324 [+0.226, +0.424]
- M1_s1 concise+prompt − default+prompt: +0.597 [+0.489, +0.708]

## Stage C — dose and position (trigger), as fraction of DD − HH

| model | system prompt | variant | metric | − HH [95% CI] | fraction of DD − HH |
|---|---|---|---|---|---|
| M1_s1 | none | DH | +10.210 | +0.228 [+0.147, +0.306] | 0.24 |
| M1_s1 | none | HD | +10.382 | +0.401 [+0.249, +0.550] | 0.43 |
| M1_s1 | none | T+closer | +10.435 | +0.453 [+0.355, +0.550] | 0.49 |
| M1_s1 | none | H+closer | +9.941 | -0.040 [-0.112, +0.028] | -0.04 |
| M1_s1 | none | T+closer − truncated | | -0.051 [-0.109, +0.006] | |
| M1_s1 | none | (DD − HH gap) | | +0.933 | 1.00 |
| M1_s1 | prompt | DH | +10.297 | +0.052 [-0.014, +0.117] | 0.07 |
| M1_s1 | prompt | HD | +10.467 | +0.222 [+0.106, +0.339] | 0.29 |
| M1_s1 | prompt | T+closer | +10.469 | +0.224 [+0.125, +0.323] | 0.29 |
| M1_s1 | prompt | H+closer | +10.186 | -0.059 [-0.115, -0.004] | -0.08 |
| M1_s1 | prompt | T+closer − truncated | | -0.100 [-0.166, -0.041] | |
| M1_s1 | prompt | (DD − HH gap) | | +0.772 | 1.00 |
| M2_s1 | none | DH | +10.645 | +0.165 [+0.094, +0.242] | 0.20 |
| M2_s1 | none | HD | +10.863 | +0.383 [+0.259, +0.500] | 0.46 |
| M2_s1 | none | T+closer | +10.876 | +0.396 [+0.277, +0.503] | 0.47 |
| M2_s1 | none | H+closer | +10.431 | -0.048 [-0.089, -0.006] | -0.06 |
| M2_s1 | none | T+closer − truncated | | +0.005 [-0.042, +0.056] | |
| M2_s1 | none | (DD − HH gap) | | +0.840 | 1.00 |
| M2_s1 | prompt | DH | +10.887 | +0.078 [+0.020, +0.138] | 0.09 |
| M2_s1 | prompt | HD | +11.180 | +0.371 [+0.256, +0.482] | 0.43 |
| M2_s1 | prompt | T+closer | +11.131 | +0.322 [+0.237, +0.398] | 0.37 |
| M2_s1 | prompt | H+closer | +10.807 | -0.002 [-0.053, +0.046] | -0.00 |
| M2_s1 | prompt | T+closer − truncated | | -0.011 [-0.064, +0.046] | |
| M2_s1 | prompt | (DD − HH gap) | | +0.865 | 1.00 |
