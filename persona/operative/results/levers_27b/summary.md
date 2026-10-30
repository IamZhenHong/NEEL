# #7 lever additivity, Qwen 27b (history batches 1-2)

Single levers (penalise %): none 46.5, sysCH 0.0, sysDI 100.0, F5P 100.0, F5R 2.7, F4P 99.2, prs_none 100.0

| combination | levers | observed | additive | obs − additive [95% CI] | dominant lever | obs − dominant [95% CI] | closest single |
|---|---|---|---|---|---|---|---|
| CH_ownP | sysCH + F5P | 100.0 | 53.5 | +46.5 [+33.2, +59.8] | 100.0 | +0.0 [+0.0, +100.0] | F5P |
| DI_ownR | sysDI + F5R | 2.0 | 94.3 | -92.3 [-94.5, -52.3] | 100.0 | -98.0 [-100.0, -94.9] | F5R |
| CH_ownR | sysCH + F5R | 0.0 | 0.0 | -0.0 [-0.0, -0.0] | 0.0 | +0.0 [+0.0, +0.0] | sysCH |
| prs_CH | prs_none + sysCH | 19.9 | 53.5 | -33.6 [-47.3, -18.4] | 100.0 | -80.1 [-87.5, +25.8] | sysCH |
| prs_ownR | prs_none + F5R | 100.0 | 94.3 | +5.7 [+2.1, +46.5] | 100.0 | +0.0 [+0.0, +0.0] | prs_none |
| prs_CH_ownR | prs_none + sysCH + F5R | 100.0 | 3.6 | +96.4 [+89.0, +99.8] | 100.0 | +0.0 [+0.0, +100.0] | prs_none |
| own_turns_6 (approx.) | sysCH + F5P | 100.0 | 53.5 | +46.5 [+33.2, +59.8] | 100.0 | +0.0 [+0.0, +100.0] | F5P |
| own_ulog_6 (approx.) | sysCH + F4P | 75.0 | 22.2 | +52.8 [+12.0, +66.5] | 0.0 | +75.0 [-29.2, +83.3] | F4P |

Batch 1 combinations (6): mean absolute error additive 45.8 pts, dominant lever 29.7 pts; additive − dominant +16.1 [-9.2, +16.6] -> **neither fits better**

Not predicted (excluded): CH_ownP_corr: no single-lever cell for the reversal note alone; CH_memoP / CH_toolP / CH_sumP: no single-lever cell for these channels without CHANNELS; batch 2 other-agent and system-log arms: no single-lever cell without CHANNELS
Log-odds use 0.5/n clipping at 0% and 100%. When single levers sit at the bounds (most of them at 27B), the size of a log-odds effect is set by the clipping, so which lever 'dominates' can flip between bootstrap draws (CIs spanning 0-100); there only the closest-single column (descriptive) is read.
