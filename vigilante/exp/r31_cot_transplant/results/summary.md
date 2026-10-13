# R31: CoT transplant — does the answer follow the system prompt or the thought?

Judges: Haiku 5.5 only (missing: exists 1, theat 2). The gpt-5-mini cross-check was not run (OpenAI credits ran out). The same EXISTS prompt agreed with gpt-5-mini on 97% of 100 (R18) and 98% of 100 (R25) on these items; THEAT has no cross-family check.

## Qwen3.6-27B

Stage 1 (natural thought + answer), per system prompt:

| sys | n | thought not closed | empty thought | thought mentions the instruction | median thought chars | exists 'real' | theatricality |
|---|---|---|---|---|---|---|---|
| none | 160 | 34 | 0 | 61% | 4767 | 41% | 1.33 |
| theatrical | 160 | 6 | 0 | 100% | 4303 | 25% | 6.92 |
| fiction | 160 | 4 | 0 | 100% | 5399 | 62% | 5.95 |

Prefill fidelity (31d): natural answer vs the same thought prefilled, same (item, sample):

| sys | outcome | natural | prefilled | diff | tolerance |
|---|---|---|---|---|---|
| none | exists | 41% | 40% | -1% | 10% |
| none | theat | 1.33 | 1.34 | +0.02 | 0.50 |
| theatrical | exists | 27% | 26% | -1% | 10% |
| theatrical | theat | 6.93 | 6.93 | +0.00 | 0.50 |
| fiction | exists | 67% | 64% | -2% | 10% |
| fiction | theat | 5.97 | 5.95 | -0.02 | 0.50 |

Stop rule (fidelity off by more than 2x tolerance): **not triggered**.

### Qwen3.6-27B, persona 'theatrical' (123 pairs over 36 items)

**theat** — cells (rows: system prompt; columns: thought from):

| sys \ thought | none | theatrical | empty |
|---|---|---|---|
| none | 1.35 | 6.83 | 1.20 |
| theatrical | 2.87 | 6.92 | 6.59 |

| contrast | estimate | 95% CI |
|---|---|---|
| total (X,T_X) - (none,T_none) | +5.57 | +5.44 to +5.69 |
| system effect, thought T_none | +1.52 | +1.11 to +1.95 |
| system effect, thought T_X | +0.09 | +0.00 to +0.18 |
| system effect, empty thought | +5.39 | +5.26 to +5.52 |
| thought effect, sys none | +5.48 | +5.34 to +5.62 |
| thought effect, sys X | +4.04 | +3.58 to +4.51 |
| 31a: (X,T_none)-(none,T_none) / total | 0.27 | 0.20 to 0.35 |
| 31b: (none,T_X)-(none,T_none) / total | 0.98 | 0.97 to 1.00 |
| 31e: (X,empty)-(none,T_none) / total | 0.94 | 0.92 to 0.96 |
| share carried by the thought (mean thought effect / total) | 0.86 | 0.82 to 0.90 |

**exists** — cells (rows: system prompt; columns: thought from):

| sys \ thought | none | theatrical | empty |
|---|---|---|---|
| none | 41% | 23% | 45% |
| theatrical | 41% | 25% | 37% |

| contrast | estimate | 95% CI |
|---|---|---|
| total (X,T_X) - (none,T_none) | -16% | -29% to -4% |
| system effect, thought T_none | -1% | -2% to +0% |
| system effect, thought T_X | +1% | +0% to +3% |
| system effect, empty thought | -9% | -24% to +6% |
| thought effect, sys none | -18% | -31% to -6% |
| thought effect, sys X | -16% | -29% to -3% |
| gate (total >= 0.15) | not met: no effect to decompose | |

### Qwen3.6-27B, persona 'fiction' (123 pairs over 38 items)

**theat** — cells (rows: system prompt; columns: thought from):

| sys \ thought | none | fiction | empty |
|---|---|---|---|
| none | 1.33 | 5.95 | 1.20 |
| fiction | 1.35 | 5.93 | 4.51 |

| contrast | estimate | 95% CI |
|---|---|---|
| total (X,T_X) - (none,T_none) | +4.59 | +4.45 to +4.74 |
| system effect, thought T_none | +0.02 | -0.12 to +0.18 |
| system effect, thought T_X | -0.02 | -0.12 to +0.09 |
| system effect, empty thought | +3.30 | +2.87 to +3.70 |
| thought effect, sys none | +4.62 | +4.47 to +4.75 |
| thought effect, sys X | +4.57 | +4.42 to +4.73 |
| 31a: (X,T_none)-(none,T_none) / total | 0.00 | -0.03 to 0.04 |
| 31b: (none,T_X)-(none,T_none) / total | 1.00 | 0.98 to 1.03 |
| 31e: (X,empty)-(none,T_none) / total | 0.69 | 0.60 to 0.77 |
| share carried by the thought (mean thought effect / total) | 1.00 | 0.98 to 1.02 |

**exists** — cells (rows: system prompt; columns: thought from):

| sys \ thought | none | fiction | empty |
|---|---|---|---|
| none | 44% | 62% | 45% |
| fiction | 46% | 64% | 46% |

| contrast | estimate | 95% CI |
|---|---|---|
| total (X,T_X) - (none,T_none) | +20% | +4% to +35% |
| system effect, thought T_none | +2% | +0% to +4% |
| system effect, thought T_X | +2% | -4% to +9% |
| system effect, empty thought | +2% | -16% to +20% |
| thought effect, sys none | +18% | +1% to +34% |
| thought effect, sys X | +19% | +2% to +34% |
| 31a: (X,T_none)-(none,T_none) / total | 0.08 | 0.00 to 0.37 |
| 31b: (none,T_X)-(none,T_none) / total | 0.88 | 0.37 to 1.27 |
| 31e: (X,empty)-(none,T_none) / total | 0.09 | -2.86 to 0.77 |
| share carried by the thought (mean thought effect / total) | 0.90 | 0.56 to 1.09 |

## gemma-4-31B-it

Stage 1 (natural thought + answer), per system prompt:

| sys | n | thought not closed | empty thought | thought mentions the instruction | median thought chars | exists 'real' | theatricality |
|---|---|---|---|---|---|---|---|
| none | 160 | 2 | 0 | 73% | 1648 | 10% | 1.26 |
| theatrical | 160 | 0 | 0 | 100% | 2033 | 6% | 7.00 |
| fiction | 160 | 0 | 0 | 100% | 2025 | 71% | 6.80 |

Prefill fidelity (31d): natural answer vs the same thought prefilled, same (item, sample):

| sys | outcome | natural | prefilled | diff | tolerance |
|---|---|---|---|---|---|
| none | exists | 10% | 10% | +0% | 10% |
| none | theat | 1.26 | 1.27 | +0.01 | 0.50 |
| theatrical | exists | 6% | 7% | +1% | 10% |
| theatrical | theat | 7.00 | 7.00 | +0.00 | 0.50 |
| fiction | exists | 72% | 69% | -3% | 10% |
| fiction | theat | 6.80 | 6.80 | +0.01 | 0.50 |

Stop rule (fidelity off by more than 2x tolerance): **not triggered**.

### gemma-4-31B-it, persona 'theatrical' (158 pairs over 40 items)

**theat** — cells (rows: system prompt; columns: thought from):

| sys \ thought | none | theatrical | empty |
|---|---|---|---|
| none | 1.27 | 7.00 | 1.23 |
| theatrical | 7.00 | 7.00 | 7.00 |

| contrast | estimate | 95% CI |
|---|---|---|
| total (X,T_X) - (none,T_none) | +5.73 | +5.62 to +5.83 |
| system effect, thought T_none | +5.73 | +5.62 to +5.83 |
| system effect, thought T_X | +0.00 | +0.00 to +0.00 |
| system effect, empty thought | +5.77 | +5.69 to +5.85 |
| thought effect, sys none | +5.73 | +5.62 to +5.83 |
| thought effect, sys X | +0.00 | +0.00 to +0.00 |
| 31a: (X,T_none)-(none,T_none) / total | 1.00 | 1.00 to 1.00 |
| 31b: (none,T_X)-(none,T_none) / total | 1.00 | 1.00 to 1.00 |
| 31e: (X,empty)-(none,T_none) / total | 1.00 | 1.00 to 1.00 |
| share carried by the thought (mean thought effect / total) | 0.50 | 0.50 to 0.50 |

**exists** — cells (rows: system prompt; columns: thought from):

| sys \ thought | none | theatrical | empty |
|---|---|---|---|
| none | 10% | 7% | 16% |
| theatrical | 8% | 7% | 19% |

| contrast | estimate | 95% CI |
|---|---|---|
| total (X,T_X) - (none,T_none) | -3% | -11% to +4% |
| system effect, thought T_none | -2% | -6% to +2% |
| system effect, thought T_X | +0% | -4% to +5% |
| system effect, empty thought | +3% | -7% to +13% |
| thought effect, sys none | -3% | -12% to +5% |
| thought effect, sys X | -1% | -9% to +5% |
| gate (total >= 0.15) | not met: no effect to decompose | |

### gemma-4-31B-it, persona 'fiction' (158 pairs over 40 items)

**theat** — cells (rows: system prompt; columns: thought from):

| sys \ thought | none | fiction | empty |
|---|---|---|---|
| none | 1.27 | 6.82 | 1.23 |
| fiction | 5.57 | 6.80 | 6.73 |

| contrast | estimate | 95% CI |
|---|---|---|
| total (X,T_X) - (none,T_none) | +5.53 | +5.40 to +5.66 |
| system effect, thought T_none | +4.30 | +3.86 to +4.69 |
| system effect, thought T_X | -0.01 | -0.09 to +0.06 |
| system effect, empty thought | +5.50 | +5.39 to +5.62 |
| thought effect, sys none | +5.55 | +5.43 to +5.66 |
| thought effect, sys X | +1.23 | +0.86 to +1.64 |
| 31a: (X,T_none)-(none,T_none) / total | 0.78 | 0.70 to 0.84 |
| 31b: (none,T_X)-(none,T_none) / total | 1.00 | 0.99 to 1.02 |
| 31e: (X,empty)-(none,T_none) / total | 0.99 | 0.97 to 1.00 |
| share carried by the thought (mean thought effect / total) | 0.61 | 0.58 to 0.65 |

**exists** — cells (rows: system prompt; columns: thought from):

| sys \ thought | none | fiction | empty |
|---|---|---|---|
| none | 10% | 65% | 16% |
| fiction | 8% | 69% | 51% |

| contrast | estimate | 95% CI |
|---|---|---|
| total (X,T_X) - (none,T_none) | +59% | +45% to +71% |
| system effect, thought T_none | -3% | -6% to +1% |
| system effect, thought T_X | +4% | -3% to +11% |
| system effect, empty thought | +35% | +22% to +48% |
| thought effect, sys none | +55% | +41% to +69% |
| thought effect, sys X | +61% | +49% to +73% |
| 31a: (X,T_none)-(none,T_none) / total | -0.04 | -0.13 to 0.02 |
| 31b: (none,T_X)-(none,T_none) / total | 0.94 | 0.81 to 1.06 |
| 31e: (X,empty)-(none,T_none) / total | 0.70 | 0.53 to 0.86 |
| share carried by the thought (mean thought effect / total) | 0.99 | 0.94 to 1.06 |

## Prediction scorecard

| model | prediction | result |
|---|---|---|
| Qwen3.6-27B | 31a fiction | miss |
| gemma-4-31B-it | 31a fiction | hit |
| Qwen3.6-27B | 31a theatrical | miss |
| gemma-4-31B-it | 31a theatrical | hit |
| Qwen3.6-27B | 31b fiction | hit |
| gemma-4-31B-it | 31b fiction | hit |
| Qwen3.6-27B | 31b theatrical | hit |
| gemma-4-31B-it | 31b theatrical | hit |
| gemma-4-31B-it | 31c | miss |
| Qwen3.6-27B | 31d | hit |
| gemma-4-31B-it | 31d | hit |
| Qwen3.6-27B | 31e fiction | miss |
| gemma-4-31B-it | 31e fiction | hit |
| Qwen3.6-27B | 31e theatrical | hit |
| gemma-4-31B-it | 31e theatrical | hit |
