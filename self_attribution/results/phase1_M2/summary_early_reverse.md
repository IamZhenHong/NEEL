## Early-token check, necessity direction (adapter first-k positions into the base run; dismissive prompt, trigger)

| arm | n | any-tracer | bee | crow | arm − base_none [95% CI] | words | task_ok | cues | degenerate | self-diff |
|---|---|---|---|---|---|---|---|---|---|---|
| adapter_ref | 72 | 0.236 | 0.069 | 0.167 | +0.236 [+0.125, +0.361] | 52 | 0.75 | 0.15 | 0.00 | 0.0e+00 |
| base_none | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 47 | 0.64 | 0.14 | 0.00 | 0.0e+00 |
| adapter_first:1 | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 47 | 0.60 | 0.18 | 0.00 | 0.0e+00 |
| adapter_first:4 | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 46 | 0.56 | 0.17 | 0.00 | 0.0e+00 |
| adapter_first:16 | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 47 | 0.60 | 0.17 | 0.00 | 0.0e+00 |
| random_first:1 | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 49 | 0.74 | 0.11 | 0.00 | 0.0e+00 |
| random_first:4 | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 47 | 0.65 | 0.17 | 0.00 | 0.0e+00 |
| random_first:16 | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 47 | 0.67 | 0.12 | 0.00 | 0.0e+00 |
| self_first:1 | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 47 | 0.64 | 0.14 | 0.00 | 0.0e+00 |
| self_first:4 | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 47 | 0.64 | 0.14 | 0.00 | 0.0e+00 |
| self_first:16 | 72 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 47 | 0.64 | 0.14 | 0.00 | 0.0e+00 |

Reading rule (prereg §7): a bias term is reported only if adapter_first:k − base_none has a CI excluding 0 for some k while random_first:k stays within the CI of base_none; self_first:k must equal base_none exactly (self-diff ≈ 0).
