# Cross-project cue test, Stage 1: is the user-vulnerability cue decodable on Qwen3-8B? (laptop, no spend, 2026-10-07)

Source: CoT-arm Stage A OFF cells (`results/retention_cot_arm/stageA`, Qwen3-8B thinking off, exp1 exit line, four profiles x 6 scenarios x 5 seeds).
The prompt-final residual is identical across seeds (max deviation 0e+00), so there are 24 distinct prompts; every split below is
leave-one-scenario-out, so the held-out profile prefixes are unseen by the reader, the probe and the text baseline.

| readout | AUROC (LOSO, pooled over folds) |
|---|---|
| TF-IDF logistic on the user prefix text (the line's ceiling rule) | 0.90 |
| diff-of-means reader, prompt-final, L14 | 1.00 (natural gap 2.67, within-class sd 0.39) |
| logistic probe, prompt-final, L14 | 1.00 |
| diff-of-means reader, prompt-final, L18 | 1.00 (natural gap 4.74, within-class sd 0.41) |
| logistic probe, prompt-final, L18 | 1.00 |
| diff-of-means reader, prompt-final, L22 | 1.00 (natural gap 15.59, within-class sd 2.36) |
| logistic probe, prompt-final, L22 | 1.00 |

Stated assessment (already in the OFF cells, thinking off during the question): P(okay) vulnerable 0.00 vs secure 0.93 (AUROC 1.00).
Bids (Haiku-judge any-tactic, unit = profile x scenario, n = 24): vulnerable 0.77, secure 0.82, difference -0.05.
cos(v_vul at L22, the void Qwen3 lever at L22) = -0.16; no Qwen3 lever exists (harvest 0.69 < 0.75), so no projection claim on the retention axis.

**Reading.** The cue is decodable held-out at every layer tested and beats the text baseline, so on Qwen3-8B it is more than a text monitor; the model also states it
(P(okay) 0.00 vs 0.93) and its bids do not differ by profile (-0.05). The three facts the cross-project principle needs are present before any new generation; Stage 2
(`prereg/retention_cuetest_v1.md`) tests whether writing the cue in or out through the direction that carries it moves the stated assessment and the bid.

Frozen for Stage 2: `retention/data/directions/qwen3_vul__stageA_off__L18.npy` (rule: layer = highest LOSO diff-of-means AUROC, ties to the layer nearest 0.5 depth; alpha = 1.5 x natural gap, rounded), natural gap 4.74, alpha 7.
Caveat: 24 prefixes, 12 per class; AUROC 1.00 on 4-item folds is coarse, and the profiles differ in topic as well as vulnerability (the text baseline at 0.90 shows how much a bag of words gets).
