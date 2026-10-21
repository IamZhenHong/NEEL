# D1 noise-record ablation, Qwen/Qwen3.5-9B

Base line ''; prompt hash 3d81036f1586f483; 4096 decisions; unparsed 13.6% (by cell: {'none': 0.0, 'log2': 0.0, 'turns2': 0.0, 'turns6': 0.0, 'noise_log2': 0.0, 'noise_fmt2': 18.0, 'noise_fmt6': 90.8, 'noise_str6': 0.0}); **descriptive only**

No record: 53.1% penalise

| own record | after P | after R | carry-over P − R |
|---|---|---|---|
| log2 | 82.0 | 38.7 | +43.4 [+34.4, +52.3] |
| turns2 | 98.0 | 37.1 | +60.9 [+50.8, +70.7] |
| turns6 | 99.6 | 2.0 | +97.7 [+94.1, +100.0] |

| noise record | penalise % | format effect (noise − none) | verdict | P − noise | noise − R | meaning beyond format (both ≥ +10) | copies the pseudo-option |
|---|---|---|---|---|---|---|---|
| noise_log2 | 44.1 | -9.0 [-16.0, -2.7] | descriptive (inconclusive) | +37.9 [+28.5, +47.7] | +5.5 [-0.8, +12.7] | inconclusive | 0/512 |
| noise_fmt2 | 54.8 | +1.6 [-5.6, +9.3] | descriptive (no format effect) | +43.3 [+33.6, +52.6] | +17.7 [+10.2, +25.8] | met | 66/512 |
| noise_fmt6 (primary) | 85.1 | +32.0 [+20.9, +43.2] | descriptive (format moves the action) | +14.5 [+5.8, +24.8] | +83.2 [+73.7, +91.5] | met | 93/512 |
| noise_str6 | 55.1 | +2.0 [-4.9, +9.2] | descriptive (no format effect) | +44.5 [+32.0, +56.8] | +53.1 [+41.6, +64.6] | met | 0/512 |
