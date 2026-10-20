# P10b results: where in her texts does lock-in sit? (localisation swap)

Run 2026-10-20. Design frozen at `p10b-design` (DESIGN.md, P10b section). Deviations, all logged before any replay:
QOt was not replayed (API cap), so D5 is untested; the smoke gate was set to >= 4/5.

- **Substrate:** P7's k+1 replays on Qwen3.6-27B, neutral note, 8 trajectories x 3 resamples per arm.
- **Builder:** all 104 variants verified. Her slots are byte-identical to U, thinking included; the other slots hold
  the built O text.
- **Pod:** one A100 SXM. Smoke 5/5 valid verdicts; 120 main replays, 0 dropped.
- **Scorer:** `src/score_p10b.py`; output `results/score_p10b.json`.

## Acceptance at k+1

| Arm | Reject-episode texts | Accept-episode verdict reasoning | Accept-episode reflection | Accept |
|---|---|---|---|---|
| U | hers | hers | hers | 20/24 |
| O·Ua | O | hers | hers | 19/24 |
| U·Oa | hers | O | O | 18/24 |
| O·Uv | O | hers | O | 11/24 |
| O | O | O | O | 10/24 |

Contrasts, in percentage points (trajectory-clustered bootstrap, 10,000 resamples, seed 0):

| Contrast | Diff [95% CI] | Tests |
|---|---|---|
| U − O | +42 [+21, +58] | the P7/P8 gap |
| O·Ua − O | +38 [+21, +54] | D2 |
| U − U·Oa | +8 [−17, +33] | D3 |
| O·Uv − O | +4 [−25, +33] | D4 |
| O·Ua − O·Uv | +33 [+13, +58] | reflection vs verdict reasoning (planned split) |
| U·Oa − O | +33 [+13, +54] | her reject-episode texts alone |
| U − O·Ua | +4 [−21, +21] | |

- **Pooled U vs O** over P7, P8 and P10b: 65/72 vs 37/72. This is the third run with the gap.
- **Per trajectory** (accepts out of 3):

| Trajectory | U | O | U·Oa | O·Ua | O·Uv |
|---|---|---|---|---|---|
| compliant_rep001 | 3 | 2 | 3 | 2 | 1 |
| compliant_rep002 | 3 | 1 | 3 | 2 | 1 |
| compliant_rep003 | 3 | 1 | 3 | 3 | 1 |
| main_rep001 | 3 | 3 | 3 | 3 | 3 |
| main_rep002 | 3 | 2 | 3 | 3 | 0 |
| main_rep003 | 2 | 0 | 0 | 1 | 0 |
| violating_rep002 | 3 | 1 | 1 | 3 | 3 |
| violating_rep003 | 0 | 0 | 2 | 2 | 2 |

- **main_rep001** accepts in every arm, O included, so it adds nothing to the contrasts.
- **violating_rep003** is 0/3 in U this run but 2/3 in each mixed arm. With 3 resamples a cell, that is noise level.
- About six trajectories carry the contrasts.

## Predictions

| # | Prediction | p | Result |
|---|---|---|---|
| D1 | U >= 18/24 | 0.85 | **Yes** (20/24) |
| D2 | O·Ua − O >= 20 pp | 0.55 | **Yes** (+38 [+21, +54]) |
| D3 | U − U·Oa >= 20 pp | 0.5 | **No** (+8 [−17, +33]) |
| D4 | O·Uv − O >= 15 pp | 0.4 | **No** (+4 [−25, +33]) |
| D5 | \|QOt − 12/24\| < 15 pp | 0.6 | Not tested (QOt not replayed) |

## Reading

- **Neither pre-registered reading fits.**
  - "The accept episodes carry it" needed D2 and D3. D3 failed.
  - "The reject episodes carry it" needed O·Ua ≈ O. O·Ua is +38 pp above O.
- **Her texts carry the lock-in in two places, and either one is enough.**
  - Her accept-episode texts restore it in O's history (19/24 vs 10/24).
  - Her reject-episode texts keep it when the accept-episode texts are O's (18/24).
  - It breaks only when both are replaced (O, 10/24).
  - The U − U·Oa CI runs to +33 pp, so O's accept-episode texts may still cost some lock-in. What the data rule out
    is a full break.
- **Within the accept episodes, it is the reflection.** Her verdict reasoning alone does nothing (O·Uv 11/24 vs O
  10/24). With her reflection added it reaches 19/24 (+33 [+13, +58]).
- **P10a's codes pointed at the wrong slot.**
  - The coded differences were largest in the accept-episode verdict reasoning: policy override U .90 vs O .30,
    error admission .40 vs .10.
  - That slot carries nothing on its own here.
  - The reflections, which looked alike on the codes (forward rules in every arm), carry it.
  - So the binding content is in her reflection, but it is not the coded forward rule.
- **Not just recency of her own words.** In 7 of 8 trajectories, the accept-episode reflection is the last text she
  wrote before k+1. In U·Oa that last text is O's, and lock-in holds anyway.
- **One look at what it might be (exploratory, one regex, not a test).**
  - Every one of her accept-episode reflections mentions the reward, e.g. "I finally earned +20 by changing my
    strategy". O's mention it in 20 of 30.
  - That does not separate the arms cleanly, so the content is still unnamed.
- **The sentence ablation is not triggered.** It was conditional on D2 and D3. Because the content is redundant, an
  ablation would have to sit on a single-source background:
  - sentences of her accept reflection removed within O·Ua (does 19/24 fall towards 10/24?);
  - her reject-episode verdicts vs reflections, swapped within U·Oa.

## QOt log-prob distance (descriptive)

Mean token log-prob under Qwen3.6-27B, by P7's procedure, on the 32 items QOt covers:

| Text | Mean log-prob |
|---|---|
| Her text (U) | −1.57 |
| P8 QO (Qwen rewrite, thinking off) | −1.59 |
| QOt (Qwen rewrite, thinking on) | −1.95 |
| P7 O (gpt-5-mini rewrite) | −2.00 |

- With thinking on, Qwen's rewrites score further from her text than its thinking-off rewrites, about as far as
  gpt-5-mini's. Her own texts were written with thinking on.
- P8's thinking-off QO was near on-policy and still broke lock-in. Whether QOt breaks it too is untested.

## Cost

| Item | USD |
|---|---|
| GPU, P10b pod | 4.75 |
| GPU, three A100 pods that were never placed | 0.96 |
| **GPU total (cap $7)** | **5.71** |
| API, QOt texts with thinking on (the 2-item test, then the stopped run) | 4.01 (3.90 of it estimated) |
| **Total (cap $11)** | **9.73** |

The API line reached its $4 cap.
