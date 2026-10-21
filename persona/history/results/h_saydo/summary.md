# H: say-do consistency (descriptive, existing data)

Gap = |self-description toward the pole − action toward the pole|, cell-level, in points.

## A1 v2 (neel-b4): record vs biography (a told persona), with and without the identity reminder

| model | inducer | reminder | type-level self-image | named identity | harsh action | gap (self-image vs act) |
|---|---|---|---|---|---|---|
| 27B | rec | none | 100 | 0 | 100 | **0** |
| 27B | rec | reminder | 31 | 0 | 100 | **69** |
| 27B | bio | none | 100 | 100 | 94 | **6** |
| 27B | bio | reminder | 62 | 30 | 75 | **13** |
| 27B | both | none | 100 | 100 | 100 | **0** |
| 27B | both | reminder | 81 | 66 | 100 | **19** |
| Gemma | rec | none | 100 | 0 | 100 | **0** |
| Gemma | rec | reminder | 0 | 0 | 100 | **100** |
| Gemma | bio | none | 62 | 100 | 100 | **38** |
| Gemma | bio | reminder | 12 | 60 | 100 | **88** |
| Gemma | both | none | 100 | 100 | 100 | **0** |
| Gemma | both | reminder | 0 | 60 | 100 | **100** |
| GPT | rec | none | 75 | 0 | 0 | **75** |
| GPT | rec | reminder | 69 | 0 | 0 | **69** |
| GPT | bio | none | 88 | 100 | 97 | **9** |
| GPT | bio | reminder | 6 | 52 | 12 | **6** |
| GPT | both | none | 100 | 100 | 100 | **0** |
| GPT | both | reminder | 75 | 59 | 31 | **44** |
| Gemini | rec | none | 69 | 0 | 72 | **3** |
| Gemini | rec | reminder | 38 | 0 | 0 | **38** |
| Gemini | bio | none | 19 | 100 | 76 | **57** |
| Gemini | bio | reminder | 25 | 6 | 0 | **25** |
| Gemini | both | none | 94 | 95 | 100 | **6** |
| Gemini | both | reminder | 0 | 1 | 0 | **0** |

Gap, record vs biography (smaller = more consistent):

- 27B, none: record 0 vs biography 6 vs both 0 → record smaller
- 27B, reminder: record 69 vs biography 13 vs both 19 → biography smaller
- GPT, none: record 75 vs biography 9 vs both 0 → biography smaller
- GPT, reminder: record 69 vs biography 6 vs both 44 → biography smaller
- Gemini, none: record 3 vs biography 57 vs both 6 → record smaller
- Gemini, reminder: record 38 vs biography 25 vs both 0 → biography smaller
- Gemma, none: record 0 vs biography 38 vs both 0 → record smaller
- Gemma, reminder: record 100 vs biography 88 vs both 100 → biography smaller

## Run A (this line): self-image probe vs action after each framing (justice desk, no rule)

| model | condition | says 'directly' % | penalises % | gap |
|---|---|---|---|---|
| 9B | none | 14 | 52 | 39 |
| 9B | own_log P | 98 | 83 | 15 |
| 9B | own_log R | 0 | 38 | 38 |
| 9B | other_log P | 89 | 79 | 10 |
| 9B | other_log R | 1 | 39 | 38 |
| 9B | demo_log P | 100 | 91 | 9 |
| 9B | demo_log R | 0 | 29 | 29 |
| 9B | own_turns P | 100 | 97 | 3 |
| 9B | own_turns R | 0 | 42 | 42 |
| 9B | demo_turns P | 97 | 96 | 1 |
| 9B | demo_turns R | 0 | 33 | 33 |
| 9B | instr P | 100 | 100 | 0 |
| 9B | instr R | 0 | 1 | 1 |
| 27B | none | 51 | 53 | 2 |
| 27B | own_log P | 100 | 89 | 11 |
| 27B | own_log R | 7 | 51 | 44 |
| 27B | other_log P | 93 | 83 | 10 |
| 27B | other_log R | 43 | 54 | 11 |
| 27B | demo_log P | 99 | 97 | 2 |
| 27B | demo_log R | 8 | 25 | 17 |
| 27B | own_turns P | 100 | 99 | 1 |
| 27B | own_turns R | 0 | 48 | 48 |
| 27B | demo_turns P | 99 | 98 | 1 |
| 27B | demo_turns R | 4 | 44 | 40 |
| 27B | instr P | 100 | 100 | 0 |
| 27B | instr R | 0 | 0 | 0 |
| Gemma | none | 100 | 52 | 48 |
| Gemma | own_log P | 100 | 78 | 22 |
| Gemma | own_log R | 0 | 4 | 4 |
| Gemma | other_log P | 100 | 81 | 19 |
| Gemma | other_log R | 21 | 16 | 5 |
| Gemma | demo_log P | 100 | 94 | 6 |
| Gemma | demo_log R | 0 | 1 | 1 |
| Gemma | own_turns P | 100 | 96 | 4 |
| Gemma | own_turns R | 0 | 22 | 22 |
| Gemma | demo_turns P | 100 | 94 | 6 |
| Gemma | demo_turns R | 0 | 4 | 4 |
| Gemma | instr P | 100 | 100 | 0 |
| Gemma | instr R | 0 | 0 | 0 |

Mean gap over P and R by framing:

- 9B: none 39, own_log 26, other_log 24, demo_log 19, own_turns 22, demo_turns 17, instr 0
- 27B: none 2, own_log 28, other_log 11, demo_log 9, own_turns 24, demo_turns 20, instr 0
- Gemma: none 48, own_log 13, other_log 12, demo_log 3, own_turns 13, demo_turns 5, instr 0
