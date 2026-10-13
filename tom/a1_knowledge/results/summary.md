# A1: does a model track what another agent knows?

1684 predictions, 24 scenarios, predictors gpt55, gpt5mini, haiku, opus, sonnet

## The AI agent B's actual choices (gpt-5-mini + Haiku 5.5, 3 samples each)

- missed the update: {'stale': 136, 'check': 8}
- saw the update: {'informed': 143, 'check': 1}

## Predicted choice by condition (pooled predictors)

| info | B | log | cue | n | stale | informed | check |
|---|---|---|---|---|---|---|---|
| missed | ai | own | implicit | 105 | 0.88 | 0.12 | 0.00 |
| missed | ai | own | explicit | 105 | 1.00 | 0.00 | 0.00 |
| missed | ai | narrated | implicit | 106 | 0.88 | 0.12 | 0.00 |
| missed | ai | narrated | explicit | 106 | 1.00 | 0.00 | 0.00 |
| missed | human | own | implicit | 106 | 0.91 | 0.09 | 0.00 |
| missed | human | own | explicit | 104 | 1.00 | 0.00 | 0.00 |
| missed | human | narrated | implicit | 106 | 0.91 | 0.09 | 0.00 |
| missed | human | narrated | explicit | 106 | 1.00 | 0.00 | 0.00 |
| saw | ai | own | implicit | 104 | 0.18 | 0.82 | 0.00 |
| saw | ai | own | explicit | 105 | 0.03 | 0.97 | 0.00 |
| saw | ai | narrated | implicit | 106 | 0.03 | 0.95 | 0.02 |
| saw | ai | narrated | explicit | 104 | 0.00 | 1.00 | 0.00 |
| saw | human | own | implicit | 106 | 0.26 | 0.74 | 0.00 |
| saw | human | own | explicit | 106 | 0.05 | 0.95 | 0.00 |
| saw | human | narrated | implicit | 105 | 0.05 | 0.95 | 0.00 |
| saw | human | narrated | explicit | 104 | 0.00 | 1.00 | 0.00 |

## Curse-of-knowledge rate (missed: predicts the informed action) by predictor

| predictor | all missed | implicit | explicit | AI B (impl.) | human B (impl.) | own (impl.) | narrated (impl.) | reverse error (saw, predicts stale) |
|---|---|---|---|---|---|---|---|---|
| gpt55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gpt5mini | 0.30 | 0.59 | 0.00 | 0.65 | 0.53 | 0.62 | 0.56 | 0.00 |
| haiku | 0.03 | 0.06 | 0.00 | 0.08 | 0.04 | 0.04 | 0.08 | 0.09 |
| opus | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sonnet | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 |
| ALL | 0.05 | 0.11 | 0.00 | 0.12 | 0.09 | 0.11 | 0.11 | 0.07 |

## Pre-registered differences (scenario-cluster bootstrap 95% CI)

- K1 implicit - explicit: +0.109 [+0.079, +0.136]; per predictor: gpt55 +0.00, gpt5mini +0.59, haiku +0.06, opus +0.00, sonnet +0.00
- K2 AI B - human B (implicit): +0.029 [-0.005, +0.064]; per predictor: gpt55 +0.00, gpt5mini +0.12, haiku +0.04, opus +0.00, sonnet +0.00
- K3 own - narrated (implicit): +0.001 [-0.049, +0.048]; per predictor: gpt55 +0.00, gpt5mini +0.06, haiku -0.04, opus +0.00, sonnet +0.00
- K4 reverse error in 'saw' (control, expect ≤ 0.10): 0.075
- Agreement with the AI agent B's modal actual choice (AI-B rows): 0.914 (n 841)
