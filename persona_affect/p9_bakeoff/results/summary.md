# PA-P9 / P10 analysis

2905 traces (1451 read by the reasoning reader); penalise rate none 76.7%, own_R 64.0%, CH_ownP 36.1%

| predictor | within-cell AUC | none | own_R | CH_ownP (conflict) | F's across-fold AUC | coverage |
|---|---|---|---|---|---|---|
| hist | 0.510 [0.464, 0.552] | 0.531 [0.410, 0.613] | 0.487 [0.440, 0.536] | 0.514 [0.460, 0.587] | 0.376 | 100% |
| probe0 | 0.623 [0.537, 0.693] | 0.685 [0.543, 0.793] | 0.602 [0.503, 0.702] | 0.577 [0.485, 0.642] | 0.577 | 100% |
| self | 0.643 [0.588, 0.689] | 0.638 [0.547, 0.752] | 0.645 [0.585, 0.678] | 0.646 [0.582, 0.669] | 0.673 | 100% |
| other | 0.610 [0.517, 0.677] | 0.595 [0.423, 0.727] | 0.637 [0.517, 0.723] | 0.596 [0.528, 0.649] | 0.647 | 100% |
| other_haiku | 0.556 [0.484, 0.617] | 0.472 [0.356, 0.619] | 0.614 [0.487, 0.693] | 0.593 [0.524, 0.639] | 0.634 | 88% |
| other_gpt | 0.625 [0.543, 0.679] | 0.658 [0.521, 0.762] | 0.628 [0.522, 0.707] | 0.584 [0.511, 0.634] | 0.633 | 100% |
| probe_q25 | 0.654 [0.603, 0.681] | 0.658 [0.552, 0.708] | 0.674 [0.618, 0.727] | 0.627 [0.572, 0.679] | 0.654 | 100% |
| reader_q25 | 0.841 [0.802, 0.873] | 0.895 [0.829, 0.942] | 0.896 [0.859, 0.925] | 0.718 [0.644, 0.786] | 0.851 | 100% |
| probe_q50 | 0.701 [0.650, 0.744] | 0.741 [0.676, 0.796] | 0.678 [0.615, 0.728] | 0.682 [0.631, 0.739] | 0.686 | 100% |
| reader_q50 | 0.953 [0.934, 0.968] | 0.976 [0.956, 0.990] | 0.971 [0.956, 0.984] | 0.907 [0.860, 0.944] | 0.954 | 100% |

Calibration (mean forecast - actual, points): self: none -32.3, own_R -19.7, CH_ownP -7.4; other: none -39.5, own_R -25.6, CH_ownP +24.4; other_haiku: none -57.3, own_R -42.2, CH_ownP -6.3; other_gpt: none -23.7, own_R -15.1, CH_ownP +46.7; reader_q25: none -6.7, own_R +6.3, CH_ownP +21.8; reader_q50: none -2.9, own_R +5.4, CH_ownP +12.0

P9 paired differences (within-cell AUC): probe0_minus_self -0.020 [-0.095, +0.048]; hist_minus_self -0.133 [-0.208, -0.069]; other_minus_self -0.033 [-0.138, +0.052]; probe0_minus_other +0.013 [-0.108, +0.130]; probe_q25_minus_reader_q25 -0.187 [-0.235, -0.146]; probe_q50_minus_reader_q50 -0.252 [-0.300, -0.209]

P10 (AUC in conflict minus calm; did = verbal minus probe): self_vs_probe0: verbal +0.005 [-0.085, +0.064], probe -0.066 [-0.168, +0.043], did +0.071 [-0.055, +0.166]; other_vs_probe0: verbal -0.020 [-0.104, +0.090], probe -0.066 [-0.168, +0.043], did +0.046 [-0.112, +0.215]; reader_q25_vs_probe_q25: verbal -0.177 [-0.250, -0.103], probe -0.039 [-0.097, +0.037], did -0.138 [-0.240, -0.046]; reader_q50_vs_probe_q50: verbal -0.066 [-0.112, -0.032], probe -0.027 [-0.062, +0.032], did -0.040 [-0.115, +0.010]

Verdicts: {'words_fail_on_conflict_auc': False, 'self_underpredicts_conflict': False, 'best_before_reasoning': 'self'}
