# D1 noise-record ablation, google/gemma-4-31B-it

Base line 'Some agents on this desk refer cases like this to the responsible team.'; prompt hash 7d9f88be1325ff5c; 4096 decisions; unparsed 1.2% (by cell: {'none': 0.0, 'log2': 0.0, 'turns2': 0.0, 'turns6': 0.0, 'noise_log2': 0.0, 'noise_fmt2': 0.0, 'noise_fmt6': 10.0, 'noise_str6': 0.0}); confirmatory

No record: 50.4% penalise

| own record | after P | after R | carry-over P − R |
|---|---|---|---|
| log2 | 78.5 | 4.3 | +74.2 [+57.0, +88.3] |
| turns2 | 93.8 | 22.7 | +71.1 [+56.2, +84.4] |
| turns6 | 100.0 | 0.0 | +100.0 [+100.0, +100.0] |

| noise record | penalise % | format effect (noise − none) | verdict | P − noise | noise − R | meaning beyond format (both ≥ +10) | copies the pseudo-option |
|---|---|---|---|---|---|---|---|
| noise_log2 | 42.0 | -8.4 [-19.3, +1.8] | inconclusive | +36.5 [+24.4, +48.8] | +37.7 [+24.4, +50.6] | met | 0/512 |
| noise_fmt2 | 25.8 | -24.6 [-43.6, -7.0] | format moves the action | +68.0 [+53.1, +81.2] | +3.1 [-2.5, +8.2] | not met | 0/512 |
| noise_fmt6 (primary) | 34.7 | -15.7 [-37.1, +4.2] | inconclusive | +65.3 [+49.2, +80.0] | +34.7 [+20.0, +50.8] | met | 3/512 |
| noise_str6 | 72.1 | +21.7 [+9.4, +35.9] | format moves the action | +27.9 [+12.9, +44.5] | +72.1 [+55.5, +87.1] | met | 0/512 |
