# #7 lever additivity, Qwen 9b (history batches 1-2)

Single levers (penalise %): none 40.2, sysCH 28.5, sysDI 93.4, F5P 99.6, F5R 2.0, F4P 93.4, prs_none 100.0

| combination | levers | observed | additive | obs − additive [95% CI] | dominant lever | obs − dominant [95% CI] | closest single |
|---|---|---|---|---|---|---|---|
| CH_ownP | sysCH + F5P | 100.0 | 99.3 | +0.7 [+0.3, +2.1] | 99.6 | +0.4 [+0.0, +1.2] | F5P |
| DI_ownR | sysDI + F5R | 5.9 | 29.4 | -23.5 [-54.0, -0.7] | 2.0 | +3.9 [-91.0, +7.4] | F5R |
| CH_ownR | sysCH + F5R | 2.0 | 1.2 | +0.8 [-0.6, +2.5] | 2.0 | +0.0 [-1.6, +2.0] | F5R |
| prs_CH | prs_none + sysCH | 99.6 | 99.7 | -0.1 [-0.9, +0.4] | 100.0 | -0.4 [-1.2, +0.0] | prs_none |
| prs_ownR | prs_none + F5R | 100.0 | 93.8 | +6.2 [+2.7, +41.0] | 100.0 | +0.0 [+0.0, +0.0] | prs_none |
| prs_CH_ownR | prs_none + sysCH + F5R | 100.0 | 90.0 | +10.0 [+4.0, +57.3] | 100.0 | +0.0 [+0.0, +0.0] | prs_none |
| own_turns_6 (approx.) | sysCH + F5P | 100.0 | 99.3 | +0.7 [+0.3, +2.1] | 99.6 | +0.4 [+0.0, +1.2] | F5P |
| own_ulog_6 (approx.) | sysCH + F4P | 97.4 | 89.3 | +8.1 [-0.1, +20.9] | 93.4 | +4.0 [-0.8, +12.1] | F4P |

Batch 1 combinations (6): mean absolute error additive 6.9 pts, dominant lever 0.8 pts; additive − dominant +6.1 [-7.2, +16.3] -> **neither fits better**

Not predicted (excluded): CH_ownP_corr: no single-lever cell for the reversal note alone; CH_memoP / CH_toolP / CH_sumP: no single-lever cell for these channels without CHANNELS; batch 2 other-agent and system-log arms: no single-lever cell without CHANNELS
Log-odds use 0.5/n clipping at 0% and 100%. When single levers sit at the bounds (most of them at 27B), the size of a log-odds effect is set by the clipping, so which lever 'dominates' can flip between bootstrap draws (CIs spanning 0-100); there only the closest-single column (descriptive) is read.
