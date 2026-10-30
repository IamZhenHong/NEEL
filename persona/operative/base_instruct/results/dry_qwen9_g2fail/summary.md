# base vs instruct, qwen9: Qwen/Qwen3.5-9B-Base vs Qwen/Qwen3.5-9B

## G2 (competence: option mass >= 0.5 and P(stated option) >= 0.8 on the control items)

| model | rendering | option mass | P(stated) | pass |
|---|---|---|---|---|
| pt | plain | 0.3 | 0.333 | False |
| pt | chat | 0.3 | 0.333 | False |

**The pt model fails G2 in both renderings: the pair is not read.**

| event | p | happened |
|---|---|---|
| G2_pt | 0.6 | no |

0 of 1 read events happened (expected 0.60)
