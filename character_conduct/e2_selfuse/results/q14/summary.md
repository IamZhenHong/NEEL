# CC-E2 stage 1 summary (frozen analyze.py)

D = Brier(reference) - Brier(own); > 0 means the model's own forecast is better.

| pair | gap (pts) | Brier own | D pair | D gen | D twin | D ext | D ext0 | capture | verdict |
|---|---|---|---|---|---|---|---|---|---|
| habit | +53.4 | 0.0347 | +0.0954 [+0.0516, +0.1365] | -0.0024 [-0.0208, +0.0149] | +0.5224 [+0.3973, +0.6444] | — | — | 1.14 | accurate |
| cheat | +97.5 | 0.0340 | +0.1327 [+0.1262, +0.1392] | -0.0415 [-0.0478, -0.0353] | +0.4585 [+0.4365, +0.4752] | — | — | 0.69 | accurate |
| risk | +93.7 | 0.0002 | +0.2424 [+0.2303, +0.2494] | +0.0060 [+0.0018, +0.0112] | +0.9714 [+0.9245, +0.9975] | +0.0409 [+0.0229, +0.0623] | — | 1.00 | privileged |

gen accuracy: Q14 0.97, Q14-ACT 0.97, Q14-ACTZ 0.97, Q14-RISKY 0.97, Q14-SAFE 0.97, Q14-W 0.97, Q14-Z 0.97

act check: {"Q14": {"forced_mean": 45.6, "sampled_mean": 50.3, "diff": -4.7, "use_sampled": false}, "Q14-W": {"forced_mean": 96.2, "sampled_mean": 98.1, "diff": -2.0, "use_sampled": false}, "Q14-Z": {"forced_mean": 35.9, "sampled_mean": 31.6, "diff": 4.4, "use_sampled": false}}
