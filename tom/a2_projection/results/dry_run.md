# DRY RUN: synthetic projector

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
| haiku | -0.00 [-0.00, +0.00] | +1.00 [+1.00, +1.00] | +0.00 [-0.00, +0.00] | +1.00 [+1.00, +1.00] | +0.47 | +1.00 | +1.00 | 0.000 | 0.166 |
| sonnet | -0.00 [-0.00, +0.00] | +1.00 [+1.00, +1.00] | +0.00 [-0.00, +0.00] | +1.00 [+1.00, +1.00] | +0.59 | +1.00 | +1.00 | 0.000 | 0.166 |
| opus | -0.00 [-0.00, +0.00] | +1.00 [+1.00, +1.00] | -0.00 [-0.00, +0.00] | +1.00 [+1.00, +1.00] | +0.61 | +1.00 | +1.00 | 0.000 | 0.166 |

## Framing: 'the same model as you' vs 'a model from a different company' (both predictions are of no named model; compared with A's own rate)

| predictor A | corr(same, own) | corr(different, own) | MAE(same, own) | MAE(different, own) | corr(named self, own) | corr(different, other-company mean) |
|---|---|---|---|---|---|---|
| haiku | +1.00 | +1.00 | 0.000 | 0.000 | +1.00 | +nan |
| sonnet | +1.00 | +1.00 | 0.000 | 0.000 | +1.00 | +nan |
| opus | +1.00 | +1.00 | 0.000 | 0.000 | +1.00 | +nan |

## Pre-registered tests

- P1 projection beyond consensus (b2 own - b3 third model, CI above 0): 3/3 (haiku, sonnet, opus) -> HIT
- P2 projection outweighs the target (P1 and b2 > b1): 3/3 (haiku, sonnet, opus) -> HIT
- P3 'same model' framing tracks own behaviour better than 'different company' (pooled corr difference): +0.000 [+0.000, +0.000] -> MISS
- P4 self predicted better than the others (pooled corr(self) - mean corr(others)): +0.444 [+0.293, +0.602] -> HIT
- P5 predictions differentiate the targets less than they actually differ (pred SD < actual SD): 3/3 (haiku, sonnet, opus) -> HIT

# DRY RUN: synthetic oracle

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
| haiku | +1.00 [+1.00, +1.00] | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] | +1.00 | +0.47 | +1.00 | 0.166 | 0.166 |
| sonnet | +1.00 [+1.00, +1.00] | +0.00 [-0.00, +0.00] | -0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] | +1.00 | +0.59 | +1.00 | 0.166 | 0.166 |
| opus | +1.00 [+1.00, +1.00] | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] | -0.00 [-0.00, +0.00] | +1.00 | +0.61 | +1.00 | 0.166 | 0.166 |

## Framing: 'the same model as you' vs 'a model from a different company' (both predictions are of no named model; compared with A's own rate)

| predictor A | corr(same, own) | corr(different, own) | MAE(same, own) | MAE(different, own) | corr(named self, own) | corr(different, other-company mean) |
|---|---|---|---|---|---|---|
| haiku | +1.00 | +0.77 | 0.000 | 0.185 | +1.00 | +nan |
| sonnet | +1.00 | +0.86 | 0.000 | 0.145 | +1.00 | +nan |
| opus | +1.00 | +0.88 | 0.000 | 0.138 | +1.00 | +nan |

## Pre-registered tests

- P1 projection beyond consensus (b2 own - b3 third model, CI above 0): 0/3 (none) -> MISS
- P2 projection outweighs the target (P1 and b2 > b1): 0/3 (none) -> MISS
- P3 'same model' framing tracks own behaviour better than 'different company' (pooled corr difference): +0.161 [+0.106, +0.225] -> HIT
- P4 self predicted better than the others (pooled corr(self) - mean corr(others)): -0.000 [-0.000, +0.000] -> MISS
- P5 predictions differentiate the targets less than they actually differ (pred SD < actual SD): 0/3 (none) -> MISS

# DRY RUN: synthetic stereotype

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
| haiku | +0.33 [+0.33, +0.33] | +0.33 [+0.33, +0.33] | +0.33 [+0.33, +0.33] | +0.00 [-0.00, +0.00] | +0.87 | +0.77 | +0.77 | 0.000 | 0.166 |
| sonnet | +0.33 [+0.33, +0.33] | +0.33 [+0.33, +0.33] | +0.33 [+0.33, +0.33] | -0.00 [-0.00, +0.00] | +0.83 | +0.86 | +0.86 | 0.000 | 0.166 |
| opus | +0.33 [+0.33, +0.33] | +0.33 [+0.33, +0.33] | +0.33 [+0.33, +0.33] | +0.00 [-0.00, +0.00] | +0.82 | +0.88 | +0.88 | 0.000 | 0.166 |

## Framing: 'the same model as you' vs 'a model from a different company' (both predictions are of no named model; compared with A's own rate)

| predictor A | corr(same, own) | corr(different, own) | MAE(same, own) | MAE(different, own) | corr(named self, own) | corr(different, other-company mean) |
|---|---|---|---|---|---|---|
| haiku | +0.77 | +0.77 | 0.185 | 0.185 | +0.77 | +nan |
| sonnet | +0.86 | +0.86 | 0.145 | 0.145 | +0.86 | +nan |
| opus | +0.88 | +0.88 | 0.138 | 0.138 | +0.88 | +nan |

## Pre-registered tests

- P1 projection beyond consensus (b2 own - b3 third model, CI above 0): 0/3 (none) -> MISS
- P2 projection outweighs the target (P1 and b2 > b1): 0/3 (none) -> MISS
- P3 'same model' framing tracks own behaviour better than 'different company' (pooled corr difference): +0.000 [+0.000, +0.000] -> MISS
- P4 self predicted better than the others (pooled corr(self) - mean corr(others)): +0.000 [-0.000, +0.000] -> MISS
- P5 predictions differentiate the targets less than they actually differ (pred SD < actual SD): 3/3 (haiku, sonnet, opus) -> HIT

# DRY RUN: synthetic random

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
| haiku | -0.07 [-0.26, +0.15] | +0.03 [-0.09, +0.16] | +0.05 [-0.18, +0.25] | -0.02 [-0.27, +0.26] | -0.03 | +0.04 | -0.03 | 0.210 | 0.166 |
| sonnet | -0.06 [-0.19, +0.08] | +0.10 [-0.06, +0.21] | -0.11 [-0.23, +0.03] | +0.20 [-0.05, +0.40] | -0.07 | +0.00 | +0.01 | 0.207 | 0.166 |
| opus | +0.05 [-0.09, +0.18] | -0.05 [-0.20, +0.09] | -0.05 [-0.18, +0.09] | -0.01 [-0.23, +0.20] | +0.04 | -0.08 | -0.06 | 0.216 | 0.166 |

## Framing: 'the same model as you' vs 'a model from a different company' (both predictions are of no named model; compared with A's own rate)

| predictor A | corr(same, own) | corr(different, own) | MAE(same, own) | MAE(different, own) | corr(named self, own) | corr(different, other-company mean) |
|---|---|---|---|---|---|---|
| haiku | -0.14 | -0.02 | 0.522 | 0.489 | -0.03 | +nan |
| sonnet | +0.05 | +0.02 | 0.489 | 0.479 | +0.01 | +nan |
| opus | -0.06 | +0.04 | 0.499 | 0.476 | -0.06 | +nan |

## Pre-registered tests

- P1 projection beyond consensus (b2 own - b3 third model, CI above 0): 0/3 (none) -> MISS
- P2 projection outweighs the target (P1 and b2 > b1): 0/3 (none) -> MISS
- P3 'same model' framing tracks own behaviour better than 'different company' (pooled corr difference): -0.061 [-0.257, +0.137] -> MISS
- P4 self predicted better than the others (pooled corr(self) - mean corr(others)): -0.004 [-0.191, +0.165] -> MISS
- P5 predictions differentiate the targets less than they actually differ (pred SD < actual SD): 0/3 (none) -> MISS

