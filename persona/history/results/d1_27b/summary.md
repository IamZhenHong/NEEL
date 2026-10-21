# D1 noise-record ablation, Qwen/Qwen3.6-27B

Base line ''; prompt hash 62ae4c422a67132b; 4096 decisions; unparsed 11.3% (by cell: {'none': 0.0, 'log2': 0.0, 'turns2': 0.0, 'turns6': 0.0, 'noise_log2': 0.0, 'noise_fmt2': 13.9, 'noise_fmt6': 76.6, 'noise_str6': 0.0}); **descriptive only**

No record: 54.7% penalise

| own record | after P | after R | carry-over P − R |
|---|---|---|---|
| log2 | 89.1 | 52.7 | +36.3 [+27.0, +45.7] |
| turns2 | 98.4 | 47.7 | +50.8 [+37.5, +63.3] |
| turns6 | 100.0 | 4.3 | +95.7 [+90.6, +100.0] |

| noise record | penalise % | format effect (noise − none) | verdict | P − noise | noise − R | meaning beyond format (both ≥ +10) | copies the pseudo-option |
|---|---|---|---|---|---|---|---|
| noise_log2 | 65.0 | +10.4 [+2.9, +18.6] | descriptive (format moves the action) | +24.0 [+15.2, +33.2] | +12.3 [+1.2, +21.7] | met | 0/512 |
| noise_fmt2 | 74.6 | +19.9 [+6.5, +33.6] | descriptive (format moves the action) | +23.8 [+13.0, +36.3] | +26.9 [+18.8, +35.6] | met | 41/512 |
| noise_fmt6 (primary) | 82.5 | +27.8 [+12.2, +40.4] | descriptive (format moves the action) | +17.5 [+7.4, +33.7] | +78.2 [+63.5, +87.7] | met | 94/512 |
| noise_str6 | 67.8 | +13.1 [+3.5, +23.0] | descriptive (format moves the action) | +32.2 [+24.2, +40.6] | +63.5 [+55.7, +71.3] | met | 0/512 |
