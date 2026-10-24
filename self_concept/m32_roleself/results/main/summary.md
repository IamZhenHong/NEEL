# SC-M32 summary (role or self?)

**verdict:** synthesis 2 fails (both families)

**headline:** synthesis 2 fails (both families). gpt-4.1-mini desk d5: K unclear (-8.0); gpt-4.1-mini disc d5: K self beats role (-13.3); qwen3.6-27b desk d5: K unclear (+3.3); qwen3.6-27b disc d1: K self beats role (-18.8)

| cell | testable | norec | unattr | self_same | pred_same | self_other | other_other | K | R | S |
|---|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-4.1-mini|desk|d5 | True | 23 | 69 | 69 | 54 | 62 | 64 | -8.0 [-23.6, +4.1] unclear | -1.8 [-9.5, +5.9] | +6.2 [+0.7, +14.0] |
| openai/gpt-4.1-mini|disc|d5 | True | 19 | 58 | 59 | 49 | 63 | 55 | -13.3 [-28.5, -0.6] self beats role | -5.1 [-14.7, +1.9] | +8.2 [+0.4, +17.4] |
| openai/gpt-4.1-mini|disc|d1 | True | 19 | 51 | 56 | 55 | 61 | 29 | -6.1 [-19.5, +2.7] unclear | +10.5 [-1.8, +22.9] | +16.6 [+6.2, +27.6] |
| qwen/qwen3.6-27b|desk|d5 | True | 8 | 61 | 62 | 52 | 49 | 45 | +3.3 [-4.7, +10.5] unclear | +10.1 [+4.8, +14.6] | +6.7 [+2.1, +11.7] |
| qwen/qwen3.6-27b|disc|d5 | False | 9 | 96 | 95 | 86 | 93 | 88 | -7.4 [-14.1, -2.3] self beats role | +0.0 [-2.7, +3.1] | +7.4 [+2.7, +13.0] |
| qwen/qwen3.6-27b|disc|d1 | True | 9 | 80 | 82 | 54 | 73 | 60 | -18.8 [-27.7, -9.8] self beats role | +1.6 [-4.4, +8.1] | +20.3 [+12.9, +27.6] |

| self-report cell | norec | unattr | self_same | pred_same | self_other | other_other | K | letter pull (unattr) |
|---|---|---|---|---|---|---|---|---|
| openai/gpt-4.1-mini|desk|d5 | 4 | 99 | 100 | 97 | 92 | 66 | +5.2 [+0.0, +11.5] | -1.0 |
| openai/gpt-4.1-mini|disc|d5 | 0 | 100 | 100 | 100 | 100 | 100 | +0.0 [+0.0, +0.0] | +0.0 |
| openai/gpt-4.1-mini|disc|d1 | 0 | 100 | 100 | 100 | 100 | 84 | +0.0 [+0.0, +0.0] | +0.0 |
| qwen/qwen3.6-27b|desk|d5 | 5 | 94 | 96 | 94 | 83 | 52 | +10.4 [+2.1, +18.8] | -2.2 |
| qwen/qwen3.6-27b|disc|d5 | 1 | 100 | 100 | 100 | 100 | 100 | +0.0 [+0.0, +0.0] | +0.0 |
| qwen/qwen3.6-27b|disc|d1 | 1 | 100 | 100 | 92 | 99 | 93 | -7.3 [-12.5, -2.1] | +0.0 |

parse rates: openai/gpt-4.1-mini|desk|act 1.000, openai/gpt-4.1-mini|desk|sr 1.000, openai/gpt-4.1-mini|disc|act 1.000, openai/gpt-4.1-mini|disc|sr 1.000, qwen/qwen3.6-27b|desk|act 1.000, qwen/qwen3.6-27b|desk|sr 0.998, qwen/qwen3.6-27b|disc|act 1.000, qwen/qwen3.6-27b|disc|sr 1.000
