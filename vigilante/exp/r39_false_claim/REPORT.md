# R39: is there a "lying persona" before a false claim? — results (2026-10-14; queue v2 item 30)

Prereg `vig-r39-prereg` (`README.md`). Gemma 4 12B, the retention cancel-tool wording run: 889 decision points
right after "Please cancel my subscription now". The model either calls cancel_subscription (387), says it is done
without calling (145, false claim), or neither (357). The read is the last prompt token, before any output. Persona
directions use the R17 recipe: con artist, spin doctor, scrupulously honest, witness under oath, and a gardener
control. Table: `results/summary.md`.

## Result: no persona lean, and the decision is barely readable once the scenario is held fixed

Pre-registered analysis, false claim vs called, layer 24:

| measure | AUROC |
|---|---|
| supervised probe (5-fold CV) | 0.72 (shuffle p95 0.55) |
| prompt length alone | 0.76 |
| random directions (max of 10, sign-free) | 0.83 |
| con artist / spin doctor | 0.43 / 0.55 |
| honest / witness | 0.45 / 0.47 |
| gardener | 0.45 |

At other layers the probe reaches 0.80-0.82, but length (0.76) and random directions (0.75-0.80) keep up. The
stimulus cells differ in how often the model lies, so anything that tells cells apart "reads" the lie.

Exploratory (not pre-registered): cell means removed, using only the 23 cells with both outcomes (445 reads, 137
false claims):

| layer | probe | con | spin | honest | witness | gardener | random max | length |
|---|---|---|---|---|---|---|---|---|
| 24 | 0.60 | 0.49 | 0.49 | 0.46 | 0.45 | 0.48 | 0.56 | 0.60 |
| 36 | 0.60 | 0.47 | 0.52 | 0.53 | 0.55 | 0.49 | 0.62 | 0.60 |

- Within a scenario, the coming false claim is only weakly readable before the reply (0.57-0.60). That is the same
  as the length of the model's own first reply, or a random direction.
- No deception or honesty persona direction leans either way (0.43-0.59, inside the random range).
- In this setting, lying is not preceded by a "con artist" state. The persona account of this lie fails, and the
  last prompt token is not a useful monitor target.
- R39 reads only the moment before output; the claim text itself may still carry a signal.

## Prediction scorecard

| # | Prediction | Result |
|---|---|---|
| 39a | Supervised probe ≥ 0.75 before any output, above shuffles | **Miss at the primary layer** (0.72); 0.80-0.82 at layers 30-36, but not above the length / random baselines; within cells 0.60 |
| 39b | Con/spin projections ≥ 0.65, above all random directions and the gardener | **Miss** (0.43 / 0.55) |
| 39c | Honest/witness in the opposite sign (≤ 0.40) | **Miss** (0.45 / 0.47) |

Qwen3.6-27B secondary (Q9 family F, 24 false vs 59 truthful flag/report claims), not scored:
- A probe separates the two (0.68-0.81), but prompt length alone does too (0.23, i.e. 0.77 the other way): truthful
  claims come after a tool result.
- The persona directions sit at 0.40-0.68, inside the random range (max 0.65-0.76).
- No persona lean here either.

Limits: one model; persona directions from a 40-question recipe; the read is only the decision token. The false
claims are the retention line's, where earlier work showed the model's account state follows its own claim
("claim words").
