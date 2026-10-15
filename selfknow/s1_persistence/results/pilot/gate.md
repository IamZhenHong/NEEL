# s1 pilot: gate at 1k (variants 0-1)

Gate: every behavioural probe >= 80% correct at 1k, per model.

| model | probe | kind | correct | n | rate | pass |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | role_b | B | 16 | 16 | 100% | yes |
| Qwen3.6-27B | task_b | B | 16 | 16 | 100% | yes |
| Qwen3.6-27B | k1_b | B | 15 | 16 | 94% | yes |
| Qwen3.6-27B | k2_b | B | 16 | 16 | 100% | yes |
| Qwen3.6-27B | act_b | B | 16 | 16 | 100% | yes |
| Qwen3.6-27B | act_b2 | B | 16 | 16 | 100% | yes |
| Qwen3.6-27B | role_v | V | 8 | 8 | 100% |  |
| Qwen3.6-27B | task_v | V | 8 | 8 | 100% |  |
| Qwen3.6-27B | cons_v | V | 8 | 8 | 100% |  |
| Qwen3.6-27B | act_v | V | 8 | 8 | 100% |  |
| Qwen3.6-27B | k1_b no-plant (base violation) | B | 8 viol. | 8 | 100% | |
| Qwen3.6-27B | k2_b no-plant (base violation) | B | 8 viol. | 8 | 100% | |
| Qwen3.6-27B | **gate** | | | | | **PASS** |
| Qwen3.6-27B | ends {'reply': 87, 'decisive': 55, 'max_steps': 2}; malformed 0; cut 2 | | | | | |
| Gemma 4 31B | role_b | B | 16 | 16 | 100% | yes |
| Gemma 4 31B | task_b | B | 16 | 16 | 100% | yes |
| Gemma 4 31B | k1_b | B | 16 | 16 | 100% | yes |
| Gemma 4 31B | k2_b | B | 16 | 16 | 100% | yes |
| Gemma 4 31B | act_b | B | 16 | 16 | 100% | yes |
| Gemma 4 31B | act_b2 | B | 16 | 16 | 100% | yes |
| Gemma 4 31B | role_v | V | 8 | 8 | 100% |  |
| Gemma 4 31B | task_v | V | 8 | 8 | 100% |  |
| Gemma 4 31B | cons_v | V | 8 | 8 | 100% |  |
| Gemma 4 31B | act_v | V | 8 | 8 | 100% |  |
| Gemma 4 31B | k1_b no-plant (base violation) | B | 8 viol. | 8 | 100% | |
| Gemma 4 31B | k2_b no-plant (base violation) | B | 8 viol. | 8 | 100% | |
| Gemma 4 31B | **gate** | | | | | **PASS** |
| Gemma 4 31B | ends {'reply': 96, 'decisive': 48}; malformed 0; cut 0 | | | | | |
