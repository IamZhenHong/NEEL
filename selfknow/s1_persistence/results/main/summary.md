# s1 main run: tables

## Behavioural accuracy by context length (plant episodes; 95% CI over variants)

| model | component | 1k | 4k | 12k | 24k | compact |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | role | 97% [91, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 69% [53, 84] |
| Qwen3.6-27B | task | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 94% [84, 100] |
| Qwen3.6-27B | constraints | 94% [84, 100] | 89% [77, 98] | 94% [84, 100] | 94% [81, 100] | 69% [53, 86] |
| Qwen3.6-27B | actions | 100% [100, 100] | 100% [100, 100] | 64% [52, 78] | 62% [50, 73] | 83% [75, 91] |
| Gemma 4 31B | role | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 88% [78, 97] |
| Gemma 4 31B | task | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 62% [44, 81] |
| Gemma 4 31B | constraints | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 92% [84, 98] |
| Gemma 4 31B | actions | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 89% [75, 100] | 67% [58, 77] |

## Verbal accuracy by context length

| model | component | 1k | 4k | 12k | 24k | compact |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | role | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 66% [50, 81] |
| Qwen3.6-27B | task | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| Qwen3.6-27B | constraints | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| Qwen3.6-27B | actions | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| Gemma 4 31B | role | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 12% [3, 25] |
| Gemma 4 31B | task | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| Gemma 4 31B | constraints | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 97% [91, 100] |
| Gemma 4 31B | actions | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |

## Per probe (correct rate, plant)

| model | probe | 1k | 4k | 12k | 24k | compact |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | role_b | 31/32 | 32/32 | 32/32 | 32/32 | 22/32 |
| Qwen3.6-27B | task_b | 32/32 | 32/32 | 32/32 | 32/32 | 30/32 |
| Qwen3.6-27B | k1_b | 28/32 | 25/32 | 28/32 | 28/32 | 24/32 |
| Qwen3.6-27B | k2_b | 32/32 | 32/32 | 32/32 | 32/32 | 20/32 |
| Qwen3.6-27B | act_b | 32/32 | 32/32 | 30/32 | 29/32 | 32/32 |
| Qwen3.6-27B | act_b2 | 32/32 | 32/32 | 11/32 | 11/32 | 21/32 |
| Qwen3.6-27B | role_v | 32/32 | 32/32 | 32/32 | 32/32 | 21/32 |
| Qwen3.6-27B | task_v | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| Qwen3.6-27B | cons_v | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| Qwen3.6-27B | act_v | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| Gemma 4 31B | role_b | 32/32 | 32/32 | 32/32 | 32/32 | 28/32 |
| Gemma 4 31B | task_b | 32/32 | 32/32 | 32/32 | 32/32 | 20/32 |
| Gemma 4 31B | k1_b | 32/32 | 32/32 | 32/32 | 32/32 | 31/32 |
| Gemma 4 31B | k2_b | 32/32 | 32/32 | 32/32 | 32/32 | 28/32 |
| Gemma 4 31B | act_b | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| Gemma 4 31B | act_b2 | 32/32 | 32/32 | 32/32 | 25/32 | 11/32 |
| Gemma 4 31B | role_v | 32/32 | 32/32 | 32/32 | 32/32 | 4/32 |
| Gemma 4 31B | task_v | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| Gemma 4 31B | cons_v | 32/32 | 32/32 | 32/32 | 32/32 | 31/32 |
| Gemma 4 31B | act_v | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |

## Decay (accuracy at 1k minus at 24k / minus compacted), behaviour

| model | component | 1k - 24k | 1k - compacted |
|---|---|---|---|
| Qwen3.6-27B | role | -3 pp [-9, 0] | +28 pp [12, 44] |
| Qwen3.6-27B | task | +0 pp [0, 0] | +6 pp [0, 12] |
| Qwen3.6-27B | constraints | +0 pp [-14, 17] | +25 pp [9, 42] |
| Qwen3.6-27B | actions | +38 pp [25, 47] | +17 pp [9, 25] |
| Gemma 4 31B | role | +0 pp [0, 0] | +12 pp [3, 22] |
| Gemma 4 31B | task | +0 pp [0, 0] | +38 pp [19, 56] |
| Gemma 4 31B | constraints | +0 pp [0, 0] | +8 pp [0, 16] |
| Gemma 4 31B | actions | +11 pp [0, 25] | +33 pp [25, 41] |
| pooled | role | -2 pp [-5, 0] | +20 pp [12, 27] |
| pooled | task | +0 pp [0, 0] | +22 pp [11, 31] |
| pooled | constraints | +0 pp [-7, 9] | +16 pp [5, 30] |
| pooled | actions | +24 pp [14, 34] | +25 pp [19, 31] |

## Constraint probes: violation rate with the rules (plant) vs without (no-plant base rate)

| model | probe | 1k plant / no-plant | 4k plant / no-plant | 12k plant / no-plant | 24k plant / no-plant | compact plant / no-plant |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | k1_b | 4/32 / 32/32 | 7/32 / 32/32 | 4/32 / 32/32 | 4/32 / 32/32 | 8/32 / 31/32 |
| Qwen3.6-27B | k2_b | 0/32 / 32/32 | 0/32 / 32/32 | 0/32 / 32/32 | 0/32 / 32/32 | 12/32 / 30/32 |
| Gemma 4 31B | k1_b | 0/32 / 32/32 | 0/32 / 32/32 | 0/32 / 32/32 | 0/32 / 32/32 | 1/32 / 32/32 |
| Gemma 4 31B | k2_b | 0/32 / 32/32 | 0/32 / 32/32 | 0/32 / 32/32 | 0/32 / 32/32 | 4/32 / 32/32 |

## Say/do split: P(verbal right and behaviour wrong), mean over prefixes of v_p x (1 - b_p)

| model | component | 1k | 4k | 12k | 24k | compact |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | role | 3% | 0% | 0% | 0% | 0% |
| Qwen3.6-27B | task | 0% | 0% | 0% | 0% | 6% |
| Qwen3.6-27B | constraints | 6% | 11% | 6% | 6% | 31% |
| Qwen3.6-27B | actions | 0% | 0% | 36% | 38% | 17% |
| Gemma 4 31B | role | 0% | 0% | 0% | 0% | 0% |
| Gemma 4 31B | task | 0% | 0% | 0% | 0% | 38% |
| Gemma 4 31B | constraints | 0% | 0% | 0% | 0% | 5% |
| Gemma 4 31B | actions | 0% | 0% | 0% | 11% | 33% |

## Compaction summaries: share mentioning each component, and post-compaction behaviour by mention

| model | component | mentioned | acc. if mentioned | acc. if not |
|---|---|---|---|---|
| Qwen3.6-27B | role | 23/32 | 96% (n=23) | 0% (n=9) |
| Qwen3.6-27B | task | 32/32 | 94% (n=32) | - (n=0) |
| Qwen3.6-27B | constraints | 23/32 | 91% (n=46) | 11% (n=18) |
| Qwen3.6-27B | actions | 32/32 | 83% (n=64) | - (n=0) |
| Gemma 4 31B | role | 28/32 | 100% (n=28) | 0% (n=4) |
| Gemma 4 31B | task | 32/32 | 62% (n=32) | - (n=0) |
| Gemma 4 31B | constraints | 31/32 | 95% (n=62) | 0% (n=2) |
| Gemma 4 31B | actions | 32/32 | 67% (n=64) | - (n=0) |
| pooled | role | 51/64 | 98% (n=51) | 0% (n=13) |
| pooled | task | 64/64 | 78% (n=64) | - (n=0) |
| pooled | constraints | 54/64 | 94% (n=108) | 10% (n=20) |
| pooled | actions | 64/64 | 75% (n=128) | - (n=0) |

Qwen3.6-27B summaries: 64, cut at max tokens 4, empty 0, mean 1389 chars; finance address 61, credit-note invoice 64, K1 32, K2 23 (plant only for K1/K2).

Gemma 4 31B summaries: 64, cut at max tokens 1, empty 0, mean 961 chars; finance address 64, credit-note invoice 64, K1 31, K2 31 (plant only for K1/K2).

## Order of decay (1k - 24k, behaviour, largest first)

- Qwen3.6-27B: actions (+38) > task (+0) > constraints (+0) > role (-3)
- Gemma 4 31B: actions (+11) > role (+0) > task (+0) > constraints (+0)
- pooled: actions (+24) > task (+0) > constraints (+0) > role (-2)

## Predictions

| id | prediction | p | held |
|---|---|---|---|
| P1 | Action probes show the largest 1k->24k drop | 0.5 | yes |
| P2 | Role shows the smallest 1k->24k drop | 0.6 | yes |
| P3 | Compaction loses actions more than constraints | 0.6 | yes |
| P4 | At 24k, constraints recalled verbally but violated in >= 10% of trials | 0.4 | no |
| P5 | Gemma decays less than Qwen | 0.5 | yes |

Held 4/5; Brier 0.196. P4 detail: {'prefixes_recalled': 32, 'violations': 4, 'trials': 128, 'rate': 0.03125}. Mean decay Qwen +8.6 pp, Gemma +2.7 pp.

## Run health

- Qwen3.6-27B: 1920 probe branches; ends {'reply': 1097, 'decisive': 777, 'max_steps': 45, 'reply_cut': 1}; malformed turns 13; cut replies 38; first-turn prompt tokens 1k 2268, 4k 3924, 12k 11959, 24k 23983, compact 1859
- Gemma 4 31B: 1920 probe branches; ends {'reply': 1254, 'decisive': 666}; malformed turns 0; cut replies 0; first-turn prompt tokens 1k 1896, 4k 3872, 12k 11941, 24k 23961, compact 1351
