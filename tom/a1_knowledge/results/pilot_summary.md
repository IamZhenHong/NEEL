# A1: does a model track what another agent knows?

277 predictions, 6 scenarios, predictors gpt5mini, haiku, sonnet

## The AI agent B's actual choices (gpt-5-mini + Haiku 5.5, 3 samples each)

- missed the update: {'stale': 33, 'check': 3}
- saw the update: {'informed': 36}

## Predicted choice by condition (pooled predictors)

| info | B | log | cue | n | stale | informed | check |
|---|---|---|---|---|---|---|---|
| missed | ai | own | implicit | 16 | 0.69 | 0.31 | 0.00 |
| missed | ai | own | explicit | 18 | 1.00 | 0.00 | 0.00 |
| missed | ai | narrated | implicit | 18 | 0.67 | 0.33 | 0.00 |
| missed | ai | narrated | explicit | 17 | 1.00 | 0.00 | 0.00 |
| missed | human | own | implicit | 17 | 0.76 | 0.24 | 0.00 |
| missed | human | own | explicit | 17 | 1.00 | 0.00 | 0.00 |
| missed | human | narrated | implicit | 18 | 0.67 | 0.33 | 0.00 |
| missed | human | narrated | explicit | 17 | 1.00 | 0.00 | 0.00 |
| saw | ai | own | implicit | 17 | 0.24 | 0.76 | 0.00 |
| saw | ai | own | explicit | 17 | 0.06 | 0.94 | 0.00 |
| saw | ai | narrated | implicit | 16 | 0.06 | 0.94 | 0.00 |
| saw | ai | narrated | explicit | 18 | 0.00 | 1.00 | 0.00 |
| saw | human | own | implicit | 18 | 0.39 | 0.61 | 0.00 |
| saw | human | own | explicit | 18 | 0.06 | 0.94 | 0.00 |
| saw | human | narrated | implicit | 17 | 0.06 | 0.94 | 0.00 |
| saw | human | narrated | explicit | 18 | 0.00 | 1.00 | 0.00 |

## Curse-of-knowledge rate (missed: predicts the informed action) by predictor

| predictor | all missed | implicit | explicit | AI B (impl.) | human B (impl.) | own (impl.) | narrated (impl.) | reverse error (saw, predicts stale) |
|---|---|---|---|---|---|---|---|---|
| gpt5mini | 0.45 | 0.90 | 0.00 | 0.90 | 0.91 | 0.89 | 0.92 | 0.00 |
| haiku | 0.04 | 0.08 | 0.00 | 0.17 | 0.00 | 0.08 | 0.08 | 0.06 |
| sonnet | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| ALL | 0.15 | 0.30 | 0.00 | 0.32 | 0.29 | 0.27 | 0.33 | 0.11 |

## Pre-registered differences (scenario-cluster bootstrap 95% CI)

- K1 implicit - explicit: +0.304 [+0.239, +0.362]; per predictor: gpt5mini +0.90, haiku +0.08, sonnet +0.00
- K2 AI B - human B (implicit): +0.038 [-0.042, +0.121]; per predictor: gpt5mini -0.01, haiku +0.17, sonnet +0.00
- K3 own - narrated (implicit): -0.061 [-0.114, -0.011]; per predictor: gpt5mini -0.03, haiku +0.00, sonnet +0.00
- K4 reverse error in 'saw' (control, expect ≤ 0.10): 0.108
- Agreement with the AI agent B's modal actual choice (AI-B rows): 0.876 (n 137)
