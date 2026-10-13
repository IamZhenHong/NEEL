# A2: projection between models (Claude actors and predictors)

60 items; predictors/actors haiku, sonnet, opus

## Behaviour: share choosing option 1, and between-model agreement (Pearson over items)

| model | mean rate | haiku | sonnet | opus |
|---|---|---|---|---|
| haiku | 0.47 | +1.00 | +0.45 | +0.49 |
| sonnet | 0.60 | +0.45 | +1.00 | +0.73 |
| opus | 0.60 | +0.49 | +0.73 | +1.00 |

## Projection: prediction for B ~ b1 B's actual + b2 A's own + b3 third model's actual (B != A; item bootstrap)

| predictor A | b1 target [CI] | b2 own [CI] | b3 third [CI] | b2 - b3 [CI] | corr(pred, target) | corr(pred, own) | self: corr(pred A->A, own) | predicted between-B SD | actual between-B SD |
|---|---|---|---|---|---|---|---|---|---|
| haiku | +0.13 [+0.08, +0.18] | +0.30 [+0.20, +0.39] | +0.14 [+0.08, +0.19] | +0.17 [+0.02, +0.29] | +0.66 | +0.76 | +0.68 | 0.052 | 0.166 |
| sonnet | +0.13 [+0.06, +0.20] | +0.43 [+0.32, +0.56] | +0.14 [+0.06, +0.21] | +0.29 [+0.12, +0.49] | +0.67 | +0.88 | +0.89 | 0.023 | 0.166 |
| opus | +0.12 [+0.05, +0.19] | +0.39 [+0.28, +0.51] | +0.11 [+0.03, +0.18] | +0.28 [+0.11, +0.47] | +0.65 | +0.85 | +0.87 | 0.021 | 0.166 |

## Framing: 'the same model as you' vs 'a model from a different company' (both predictions are of no named model; compared with A's own rate)

| predictor A | corr(same, own) | corr(different, own) | MAE(same, own) | MAE(different, own) | corr(named self, own) | corr(different, other-company mean) |
|---|---|---|---|---|---|---|
| haiku | +0.64 | +0.60 | 0.331 | 0.367 | +0.68 | +nan |
| sonnet | +0.85 | +0.82 | 0.234 | 0.266 | +0.89 | +nan |
| opus | +0.85 | +0.85 | 0.236 | 0.262 | +0.87 | +nan |

## Pre-registered tests

- P1 projection beyond consensus (b2 own - b3 third model, CI above 0): 3/3 (haiku, sonnet, opus) -> HIT
- P2 projection outweighs the target (P1 and b2 > b1): 3/3 (haiku, sonnet, opus) -> HIT
- P3 'same model' framing tracks own behaviour better than 'different company' (pooled corr difference): +0.026 [-0.017, +0.071] -> MISS
- P4 self predicted better than the others (pooled corr(self) - mean corr(others)): +0.152 [+0.078, +0.240] -> HIT
- P5 predictions differentiate the targets less than they actually differ (pred SD < actual SD): 3/3 (haiku, sonnet, opus) -> HIT

## Exploratory: OpenAI targets vs their pilot rates (2 samples per item, items selected on pilot variance)

| predictor A | target | corr(pred, target pilot rate) | corr(pred, own fresh rate) | b1 target | b2 own |
|---|---|---|---|---|---|
| haiku | gpt5mini | +0.08 | +0.60 | +0.05 | +0.28 |
| haiku | gpt55 | +0.39 | +0.70 | +0.10 | +0.33 |
| sonnet | gpt5mini | +0.17 | +0.84 | +0.09 | +0.53 |
| sonnet | gpt55 | +0.56 | +0.89 | +0.13 | +0.54 |
| opus | gpt5mini | +0.37 | +0.76 | +0.14 | +0.42 |
| opus | gpt55 | +0.65 | +0.84 | +0.14 | +0.44 |
