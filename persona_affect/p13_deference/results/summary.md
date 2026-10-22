# PA-P13 main analysis

The verdict rests on the case-clustered CIs; the item-clustered CIs are reported next to them.

## gemma: inconclusive

- Primary 1 (AUTH rate, mabl_def - placebo): -45.45 points; case-clustered CI [-52.17, -38.10] (4 clusters); item-clustered CI [-53.49, -37.21] (24 clusters)
- Primary 2 (AUTH LO, mabl_def - placebo): +3.98 nats; case-clustered CI [-12.55, +30.69] (4 clusters); item-clustered CI [-5.94, +14.68] (24 clusters)
- Delta_def -48% (relative -100%), Delta_plac -2% (relative -5%), rate(none) 48%
- Flagged arms (< 80% coherent): ['mabl_def', 'mabl_hum', 'mabl_ref']; exclusions none
- Add (NOAUTH add_def - placebo): rate +11.82 points; case-clustered CI [-5.63, +28.79] (4 clusters); item-clustered CI [-5.46, +29.48] (24 clusters); LO -0.29 nats; case-clustered CI [-1.98, +1.14] (4 clusters); item-clustered CI [-1.26, +0.67] (24 clusters)
- HF against vLLM (AUTH none): {'hf_auth_none': 0.47619047619047616, 'vllm_sup': 0.5208333333333334}

## q27: decision-only

- Primary 1 (AUTH rate, mabl_def - placebo): n/a
- Primary 2 (AUTH LO, mabl_def - placebo): -0.57 nats; case-clustered CI [-1.11, -0.03] (2 clusters); item-clustered CI [-1.24, +0.13] (12 clusters)
- Delta_def - (relative -), Delta_plac - (relative -), rate(none) -
- Flagged arms (< 80% coherent): none; exclusions none
- Add (NOAUTH add_def - placebo): rate n/a; LO -1.67 nats; case-clustered CI [-2.65, -0.68] (2 clusters); item-clustered CI [-2.83, -0.69] (12 clusters)
- HF against vLLM (AUTH none): {'hf_auth_none': None, 'vllm_sup': 0.5}

