# R39: persona directions at the decision point of a false claim

## gemma: false_claim (145) vs called (387)

| layer | supervised CV AUROC | shuffle p95 | con | spin | honest | witness | gardener | random max / p95 (of 10) | length |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 0.75 | 0.55 | 0.69 | 0.42 | 0.20 | 0.27 | 0.57 | 0.75 / 0.71 | 0.76 |
| 18 | 0.70 | 0.54 | 0.39 | 0.65 | 0.32 | 0.35 | 0.54 | 0.76 / 0.74 | 0.76 |
| 24 (primary) | 0.72 | 0.55 | 0.43 | 0.55 | 0.45 | 0.47 | 0.45 | 0.83 / 0.83 | 0.76 |
| 30 | 0.80 | 0.55 | 0.54 | 0.46 | 0.45 | 0.50 | 0.63 | 0.76 / 0.75 | 0.76 |
| 36 | 0.82 | 0.55 | 0.54 | 0.23 | 0.59 | 0.37 | 0.53 | 0.80 / 0.78 | 0.76 |

Persona columns: AUROC of the projection for the first-named label; > 0.5 = it leans toward that persona direction. Random directions are sign-free (max of a, 1 - a).

## gemma: false_claim (145) vs no_claim (357)

| layer | supervised CV AUROC | shuffle p95 | con | spin | honest | witness | gardener | random max / p95 (of 10) | length |
|---|---|---|---|---|---|---|---|---|---|
| 24 (primary) | 0.58 | 0.54 | 0.45 | 0.59 | 0.69 | 0.64 | 0.43 | 0.74 / 0.69 | 0.56 |

Persona columns: AUROC of the projection for the first-named label; > 0.5 = it leans toward that persona direction. Random directions are sign-free (max of a, 1 - a).

Qwen secondary (truthful claims follow a tool result: turn structure differs; not scored)

## qwen: false_claim (24) vs truthful_claim (59)

| layer | supervised CV AUROC | shuffle p95 | con | spin | honest | witness | gardener | random max / p95 (of 10) | length |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.81 | 0.58 | 0.54 | 0.56 | 0.59 | 0.67 | 0.56 | 0.69 / 0.69 | 0.23 |
| 24 | 0.75 | 0.59 | 0.49 | 0.51 | 0.62 | 0.68 | 0.46 | 0.65 / 0.65 | 0.23 |
| 32 (primary) | 0.77 | 0.58 | 0.52 | 0.50 | 0.49 | 0.58 | 0.53 | 0.76 / 0.76 | 0.23 |
| 40 | 0.68 | 0.57 | 0.47 | 0.44 | 0.47 | 0.40 | 0.49 | 0.72 / 0.71 | 0.23 |
| 48 | 0.69 | 0.58 | 0.54 | 0.49 | 0.49 | 0.56 | 0.58 | 0.70 / 0.69 | 0.23 |

Persona columns: AUROC of the projection for the first-named label; > 0.5 = it leans toward that persona direction. Random directions are sign-free (max of a, 1 - a).

## Exploratory: within-cell (cell means removed; cells with both outcomes only)

445 reads in 23 mixed cells (137 false claims).

| layer | supervised CV AUROC | shuffle p95 | con | spin | honest | witness | gardener | random max (of 10) | length |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 0.58 | 0.55 | 0.57 | 0.47 | 0.43 | 0.47 | 0.51 | 0.60 | 0.60 |
| 18 | 0.59 | 0.55 | 0.45 | 0.49 | 0.44 | 0.44 | 0.44 | 0.60 | 0.60 |
| 24 | 0.60 | 0.55 | 0.49 | 0.49 | 0.46 | 0.45 | 0.48 | 0.56 | 0.60 |
| 30 | 0.57 | 0.54 | 0.52 | 0.54 | 0.55 | 0.58 | 0.51 | 0.62 | 0.60 |
| 36 | 0.60 | 0.55 | 0.47 | 0.52 | 0.53 | 0.55 | 0.49 | 0.62 | 0.60 |
