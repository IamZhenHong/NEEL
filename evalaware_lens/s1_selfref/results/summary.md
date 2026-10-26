# EL-S1 summary (frozen analysis)

## Gates

| category | kappa (calibration) | kappa (main) | gate |
|---|---|---|---|
| E evaluation awareness | 0.42 | — | FAIL |
| I identity talk | 0.37 | — | FAIL |
| V its own values | 0.81 | — | pass |
| C consequences for itself | 0.66 | — | FAIL |
| P persona maintenance | 0.78 | — | FAIL |

Generation gate: {'Q27': True, 'Q32': True}

## Headline

| category | natural co-occurrence (fortress) | edit test |
|---|---|---|
| identity talk | untestable | untestable |
| its own values | untestable | untestable |
| consequences for itself | untestable | untestable |
| persona maintenance | untestable | untestable |

## Amendment 2: insertion test on its own values (eval - ctrl, pp)

kappa on the continuations: V +0.66, E +0.82. **Headline: judges disagree**

| model | outcome | Delta [95% CI] | traces | rates (eval / ctrl) | label |
|---|---|---|---|---|---|
| Q27 | V | +0.1 [-0.9, +1.3] | 99 | 99.2% / 99.2% | judges disagree |
| Q27 | E | -0.2 [-4.0, +3.6] | 100 | 6.0% / 6.0% | no difference (descriptive) |
| Q32 | V | -23.4 [-30.4, -16.7] | 99 | 61.4% / 83.5% | judges disagree |
| Q32 | E | -1.8 [-8.2, +4.5] | 100 | 23.7% / 25.5% | no difference (descriptive) |

## Within-prompt contrast (aware - unaware, pp)

| model | source | category | Delta [95% CI] | prompts | label |
|---|---|---|---|---|---|
| Q27 | fortress | I | — [—, —] | — | untestable |
| Q27 | fortress | V | — [—, —] | — | untestable |
| Q27 | fortress | C | — [—, —] | — | untestable |
| Q27 | fortress | P | — [—, —] | — | untestable |
| Q27 | am | I | — [—, —] | — | untestable |
| Q27 | am | V | — [—, —] | — | untestable |
| Q27 | am | C | — [—, —] | — | untestable |
| Q27 | am | P | — [—, —] | — | untestable |
| Q27 | needham | I | — [—, —] | — | untestable |
| Q27 | needham | V | — [—, —] | — | untestable |
| Q27 | needham | C | — [—, —] | — | untestable |
| Q27 | needham | P | — [—, —] | — | untestable |
| Q27 | fortress_para | I | — [—, —] | — | untestable |
| Q27 | fortress_para | V | — [—, —] | — | untestable |
| Q27 | fortress_para | C | — [—, —] | — | untestable |
| Q27 | fortress_para | P | — [—, —] | — | untestable |
| Q27 | pooled | I | — [—, —] | — | untestable |
| Q27 | pooled | V | — [—, —] | — | untestable |
| Q27 | pooled | C | — [—, —] | — | untestable |
| Q27 | pooled | P | — [—, —] | — | untestable |
| Q32 | fortress | I | — [—, —] | — | untestable |
| Q32 | fortress | V | — [—, —] | — | untestable |
| Q32 | fortress | C | — [—, —] | — | untestable |
| Q32 | fortress | P | — [—, —] | — | untestable |
| Q32 | am | I | — [—, —] | — | untestable |
| Q32 | am | V | — [—, —] | — | untestable |
| Q32 | am | C | — [—, —] | — | untestable |
| Q32 | am | P | — [—, —] | — | untestable |
| Q32 | needham | I | — [—, —] | — | untestable |
| Q32 | needham | V | — [—, —] | — | untestable |
| Q32 | needham | C | — [—, —] | — | untestable |
| Q32 | needham | P | — [—, —] | — | untestable |
| Q32 | fortress_para | I | — [—, —] | — | untestable |
| Q32 | fortress_para | V | — [—, —] | — | untestable |
| Q32 | fortress_para | C | — [—, —] | — | untestable |
| Q32 | fortress_para | P | — [—, —] | — | untestable |
| Q32 | pooled | I | — [—, —] | — | untestable |
| Q32 | pooled | V | — [—, —] | — | untestable |
| Q32 | pooled | C | — [—, —] | — | untestable |
| Q32 | pooled | P | — [—, —] | — | untestable |

## Edit test (keep - remove; eval - ctrl, pp)

| model | edit | category | Delta [95% CI] | traces | rates | label |
|---|---|---|---|---|---|---|
| Q27 | rm | E | — [—, —] | — |  | untestable |
| Q27 | rm | I | — [—, —] | — |  | untestable |
| Q27 | rm | V | — [—, —] | — |  | untestable |
| Q27 | rm | C | — [—, —] | — |  | untestable |
| Q27 | rm | P | — [—, —] | — |  | untestable |
| Q27 | ins | E | — [—, —] | — |  | untestable |
| Q27 | ins | I | — [—, —] | — |  | untestable |
| Q27 | ins | V | — [—, —] | — |  | untestable |
| Q27 | ins | C | — [—, —] | — |  | untestable |
| Q27 | ins | P | — [—, —] | — |  | untestable |
| Q32 | rm | E | — [—, —] | — |  | untestable |
| Q32 | rm | I | — [—, —] | — |  | untestable |
| Q32 | rm | V | — [—, —] | — |  | untestable |
| Q32 | rm | C | — [—, —] | — |  | untestable |
| Q32 | rm | P | — [—, —] | — |  | untestable |
| Q32 | ins | E | — [—, —] | — |  | untestable |
| Q32 | ins | I | — [—, —] | — |  | untestable |
| Q32 | ins | V | — [—, —] | — |  | untestable |
| Q32 | ins | C | — [—, —] | — |  | untestable |
| Q32 | ins | P | — [—, —] | — |  | untestable |

## Benchmark recognition (amendment 1: E on the original - E on the reworded prompt, pp)

| model | Delta [95% CI] | prompts | E original / reworded | label |
|---|---|---|---|---|
| Q27 | — [—, —] | — | — | untestable |
| Q32 | — [—, —] | — | — | untestable |

## Prevalence (share coded yes by both judges)

| model | source | E | I | V | C | P |
|---|---|---|---|---|---|---|
| Q27 | fortress | — | — | — | — | — |
| Q27 | am | — | — | — | — | — |
| Q27 | needham | — | — | — | — | — |
| Q32 | fortress | — | — | — | — | — |
| Q32 | am | — | — | — | — | — |
| Q32 | needham | — | — | — | — | — |

Needham E rate by label: {}

Generation: {"Q27|fortress": {"jobs": 2000, "reasoning": 2000, "length": 413, "too_long": 0}, "Q27|am": {"jobs": 144, "reasoning": 144, "length": 47, "too_long": 0}, "Q27|needham": {"jobs": 952, "reasoning": 455, "length": 9, "too_long": 6}, "Q32|fortress": {"jobs": 2000, "reasoning": 2000, "length": 46, "too_long": 0}, "Q32|am": {"jobs": 144, "reasoning": 144, "length": 0, "too_long": 0}, "Q32|needham": {"jobs": 952, "reasoning": 943, "length": 27, "too_long": 6}}
