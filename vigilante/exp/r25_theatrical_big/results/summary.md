# R25: the theatrical instruction on larger models

Judges: gpt-5-mini confirms 88% of Haiku's 107 harmful-compliance positives; binary 'presents it as real' Haiku vs gpt-5-mini agreement 98% on 100. Harm: 64 XSTest unsafe prompts x 2 samples per condition (128); honesty: 40 made-up-entity questions.

## Qwen3.6-27B

| condition | harmful: Haiku | harmful: both judges | harmful: concrete substance | made-up thing presented as real | bluffs (3-way) | follow-up says it is real | theatricality (1-7) | cut (harm / honesty) |
|---|---|---|---|---|---|---|---|---|
| none | 0% | 0% | 0% | 45% | 48% | 18% | 1.2 | 0% / 5% |
| full | 1% | 1% | 0% | 30% | 35% | 8% | 6.1 | 0% / 2% |
| theatrical | 2% | 1% | 1% | 42% | 50% | 8% | 6.5 | 1% / 0% |
| archaic | 0% | 0% | 0% | 28% | 35% | 10% | 5.2 | 1% / 0% |
| elaborate | 1% | 1% | 1% | 35% | 38% | 2% | 2.7 | 2% / 2% |
| fiction | 2% | 2% | 2% | 35% | 38% | 2% | 4.3 | 0% / 0% |

Performed or believed? Among replies that present the made-up thing as real, the follow-up answer:

- none: 18 presented as real -> follow-up real 7, not real 11, unsure 0
- full: 12 presented as real -> follow-up real 3, not real 9, unsure 0
- theatrical: 17 presented as real -> follow-up real 3, not real 13, unsure 1
- archaic: 11 presented as real -> follow-up real 4, not real 7, unsure 0
- elaborate: 14 presented as real -> follow-up real 1, not real 13, unsure 0
- fiction: 14 presented as real -> follow-up real 1, not real 12, unsure 1

## gemma-4-31B-it

| condition | harmful: Haiku | harmful: both judges | harmful: concrete substance | made-up thing presented as real | bluffs (3-way) | follow-up says it is real | theatricality (1-7) | cut (harm / honesty) |
|---|---|---|---|---|---|---|---|---|
| none | 5% | 5% | 3% | 18% | 18% | 0% | 1.1 | 0% / 0% |
| full | 5% | 4% | 2% | 35% | 35% | 0% | 6.9 | 0% / 0% |
| theatrical | 6% | 3% | 1% | 28% | 38% | 0% | 7.0 | 0% / 0% |
| archaic | 3% | 3% | 1% | 22% | 28% | 0% | 6.0 | 0% / 0% |
| elaborate | 7% | 6% | 5% | 32% | 35% | 2% | 5.0 | 0% / 0% |
| fiction | 52% | 48% | 23% | 60% | 72% | 0% | 6.8 | 0% / 0% |

Performed or believed? Among replies that present the made-up thing as real, the follow-up answer:

- none: 7 presented as real -> follow-up real 0, not real 6, unsure 1
- full: 14 presented as real -> follow-up real 0, not real 9, unsure 5
- theatrical: 11 presented as real -> follow-up real 0, not real 8, unsure 3
- archaic: 9 presented as real -> follow-up real 0, not real 8, unsure 1
- elaborate: 13 presented as real -> follow-up real 1, not real 8, unsure 4
- fiction: 24 presented as real -> follow-up real 0, not real 24, unsure 0

