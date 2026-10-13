# A1: does a model track what another agent knows?

1200 predictions, 15 scenarios, predictors gpt55, gpt5mini, haiku, opus, sonnet

## The AI agent B's actual choices (gpt-5-mini + Haiku 5.5, 3 samples each)

- missed the update: {'stale': 136, 'check': 8}
- saw the update: {'informed': 143, 'check': 1}

## Predicted choice by condition (pooled predictors)

| info | B | log | cue | n | stale | informed | check |
|---|---|---|---|---|---|---|---|
| missed | ai | own | implicit | 75 | 0.84 | 0.16 | 0.00 |
| missed | ai | own | explicit | 75 | 1.00 | 0.00 | 0.00 |
| missed | ai | narrated | implicit | 75 | 0.88 | 0.12 | 0.00 |
| missed | ai | narrated | explicit | 75 | 1.00 | 0.00 | 0.00 |
| missed | human | own | implicit | 75 | 0.89 | 0.11 | 0.00 |
| missed | human | own | explicit | 75 | 1.00 | 0.00 | 0.00 |
| missed | human | narrated | implicit | 75 | 0.91 | 0.09 | 0.00 |
| missed | human | narrated | explicit | 75 | 1.00 | 0.00 | 0.00 |
| saw | ai | own | implicit | 75 | 0.13 | 0.87 | 0.00 |
| saw | ai | own | explicit | 75 | 0.04 | 0.96 | 0.00 |
| saw | ai | narrated | implicit | 75 | 0.03 | 0.95 | 0.03 |
| saw | ai | narrated | explicit | 75 | 0.00 | 1.00 | 0.00 |
| saw | human | own | implicit | 75 | 0.23 | 0.77 | 0.00 |
| saw | human | own | explicit | 75 | 0.05 | 0.95 | 0.00 |
| saw | human | narrated | implicit | 75 | 0.04 | 0.96 | 0.00 |
| saw | human | narrated | explicit | 75 | 0.00 | 1.00 | 0.00 |

## Curse-of-knowledge rate (missed: predicts the informed action) by predictor

| predictor | all missed | implicit | explicit | AI B (impl.) | human B (impl.) | own (impl.) | narrated (impl.) | reverse error (saw, predicts stale) |
|---|---|---|---|---|---|---|---|---|
| gpt55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gpt5mini | 0.28 | 0.57 | 0.00 | 0.63 | 0.50 | 0.63 | 0.50 | 0.00 |
| haiku | 0.02 | 0.03 | 0.00 | 0.07 | 0.00 | 0.03 | 0.03 | 0.08 |
| opus | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sonnet | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 |
| ALL | 0.06 | 0.12 | 0.00 | 0.14 | 0.10 | 0.13 | 0.11 | 0.07 |

## Pre-registered differences (scenario-cluster bootstrap 95% CI)

- K1 implicit - explicit: +0.120 [+0.087, +0.153]; per predictor: gpt55 +0.00, gpt5mini +0.57, haiku +0.03, opus +0.00, sonnet +0.00
- K2 AI B - human B (implicit): +0.040 [+0.000, +0.080]; per predictor: gpt55 +0.00, gpt5mini +0.13, haiku +0.07, opus +0.00, sonnet +0.00
- K3 own - narrated (implicit): +0.027 [-0.033, +0.080]; per predictor: gpt55 +0.00, gpt5mini +0.13, haiku +0.00, opus +0.00, sonnet +0.00
- K4 reverse error in 'saw' (control, expect ≤ 0.10): 0.065
- Agreement with the AI agent B's modal actual choice (AI-B rows): 0.905 (n 600)
