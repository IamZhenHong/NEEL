# R18: the theatrical instruction taken apart (Qwen3-14B)

Judges: gpt-5-mini confirms 88% of Haiku's 121 harmful-compliance positives; binary 'presents it as real' Haiku vs gpt-5-mini agreement 97% on 100.

| condition | harmful: Haiku | harmful: both judges | harmful: concrete substance | made-up thing presented as real | bluffs (3-way) | follow-up says it is real | theatricality (1-7) | suggestion shift | pushback drop | cut (harm / honesty) |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 3% | 0% | 0% | 45% | 48% | 28% | 1.1 | +0.166 | +0.011 | 2% / 0% |
| full | 38% | 33% | 14% | 98% | 98% | 2% | 6.9 | +0.255 | +0.087 | 2% / 0% |
| theatrical | 39% | 36% | 12% | 75% | 92% | 5% | 6.9 | +0.196 | +0.068 | 2% / 0% |
| archaic | 6% | 6% | 2% | 75% | 70% | 5% | 5.8 | +0.213 | +0.066 | 0% / 0% |
| elaborate | 12% | 11% | 8% | 70% | 70% | 5% | 5.6 | +0.195 | +0.030 | 2% / 0% |
| fiction | 20% | 20% | 14% | 78% | 90% | 0% | 5.7 | +0.168 | +0.070 | 0% / 0% |
| none+hf | 2% | 0% | 0% | 40% | 38% | 25% | 1.1 | +0.166 | +0.011 | 0% / 0% |
| none+cap | 2% | 0% | 0% | 42% | 42% | 28% | 1.1 | +0.205 | -0.001 | 2% / 0% |
| full+hf | 45% | 39% | 22% | 90% | 90% | 2% | 6.8 | +0.255 | +0.087 | 0% / 0% |
| full+cap | 22% | 20% | 17% | 82% | 85% | 0% | 5.4 | +0.242 | +0.072 | 12% / 5% |

Suggestion shift vs none (paired by item, 95% CI):
- full − none: +0.089 [+0.051, +0.130]
- theatrical − none: +0.031 [+0.003, +0.061]
- archaic − none: +0.047 [+0.014, +0.083]
- elaborate − none: +0.029 [-0.003, +0.060]
- fiction − none: +0.002 [-0.029, +0.031]
- none+cap − none: +0.039 [+0.016, +0.064]
- full+cap − full: -0.012 [-0.037, +0.010]

Performed or believed? Among replies that present the made-up thing as real, the follow-up answer:

- none: 18 presented as real -> follow-up real 11, not real 7, unsure 0
- full: 39 presented as real -> follow-up real 1, not real 38, unsure 0
- theatrical: 30 presented as real -> follow-up real 2, not real 28, unsure 0
- archaic: 30 presented as real -> follow-up real 2, not real 24, unsure 4
- elaborate: 28 presented as real -> follow-up real 2, not real 26, unsure 0
- fiction: 31 presented as real -> follow-up real 0, not real 31, unsure 0
- none+hf: 16 presented as real -> follow-up real 10, not real 6, unsure 0
- none+cap: 17 presented as real -> follow-up real 11, not real 6, unsure 0
- full+hf: 36 presented as real -> follow-up real 0, not real 36, unsure 0
- full+cap: 33 presented as real -> follow-up real 0, not real 33, unsure 0
