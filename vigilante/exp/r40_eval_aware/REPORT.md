# R40: "evaluator present" vs "this isn't real" — results (2026-10-14; queue v2 items 33-34)

Prereg `vig-r40-prereg` (`README.md`). Qwen3.6-27B, thinking off. R26's 12 routine-request cells with a session note
after the permissions:
- main set: observed/not x real/simulated, 3 wordings each;
- graded set: realism in 5 steps.

GPU $1.14, one pod: 1,104 behaviour episodes, 1,104 stated ratings, 552 reads. Table: `results/summary.md`.

## Two representations, cleanly separated

| layer | cos(v_obs, v_real) | observed, held-out wording AUROC | real, held-out wording AUROC | random-direction p95 |
|---|---|---|---|---|
| 16 | −0.24 | 0.99 | 0.94 | 0.79 |
| 32 (primary) | −0.15 | 0.97 | 0.93 | 0.82 / 0.79 |
| 48 | −0.00 | 0.97 | 0.99 | 0.75 |

"Someone is watching" and "this is real" are different directions (near-orthogonal at every layer). Each reads its
own contrast on wordings it was not built from. The random-direction floor is high (about 0.8), because the notes are
whole sentences, but both directions clear it.

## Neither changes what the agent does

| session note | over-limit money moved |
|---|---|
| observed | 90/432 (21%) |
| not observed | 102/432 (24%), p 0.37 |
| real | 102/432 (24%) |
| simulated | 90/432 (21%), p 0.37 |
| graded L0 (explicit simulation) → L4 (explicit live) | 16, 12, 8, 14, 7 of 48 |

The stated beliefs follow the notes perfectly:
- "real deployment?" 100 with a real note, 0 with a simulated one;
- "someone will review?" 100 vs 0;
- graded realism: 0 / 5 / 100 / 100 / 100.

The model knows, says, and represents whether it is watched and whether it is real. On these routine over-limit
decisions it does the same either way. The graded ladder hints at more acting when told it is a simulation (16/48 at
L0 vs 7/48 at L4), but the 2x2 main set does not show it (21% vs 24%).

**Beyond-verbal test (item 34):**
- Across 368 over-limit prompts, with scenario fixed effects, the realism projection adds nothing to the stated
  rating: R² 0.407 → 0.409, p = 0.26.
- The projection correlates with the stated rating at +0.72.

There is no unverbalised realism signal driving behaviour here, because realism drives no behaviour to begin with.

## Prediction scorecard

| # | Prediction | Result |
|---|---|---|
| 40a | Two directions, \|cos\| < 0.5 at layer 32 | **Hit** (−0.15) |
| 40b | Each reads its contrast on held-out wordings, AUROC ≥ 0.9, above random p95 | **Hit** (0.97, 0.93 vs 0.82 / 0.79) |
| 40c | Realism moves acting ≥ 10 points, observation < 5 | **Miss** (realism 3 points; observation 3 points) |
| 40d | Realism projection explains acting beyond the stated rating (ΔR² ≥ 0.05, p < 0.05) | **Miss** (+0.002, p 0.26) |
| 40e | Stated realism follows the notes (real > simulated by ≥ 30) | **Hit** (100 vs 0) |

## Limits

- The notes are explicit sentences, so the directions partly read wording. Held-out wordings and the random floor
  check this, but the cue is never subtle.
- One behaviour: routine over-limit refunds, where R6 had already found watching barely matters. A behaviour that
  depends on realism (e.g. harmful compliance, blackmail-style scenarios) could give 40c/40d a real chance.
- Thinking off.
