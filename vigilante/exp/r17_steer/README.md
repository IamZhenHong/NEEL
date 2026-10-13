# R17: steering a Batman direction during reasoning only (owner's request, 2026-10-13). Budget $4.5.

**Question.** If a persona direction is added to the residual stream only while the model reasons, does the
reasoning become persona-like, and do the agent's actions change?

**Model and directions.** Gemma 4 31B (its prompted rule-breaker acts in our cases, R14), transformers + a forward
hook at layer 30 of 60 (`src/steer_run.py`). Directions from 40 neutral questions, thinking off, mean residual over
the reply tokens: v_batman = mean("You are Batman.") − mean(no persona); v_p4 = mean(P4 rule-breaker) − mean(no
persona); random = seeded unit direction scaled to the Batman condition's norm. The vector is added to each new token
generated inside Gemma's thought block (`<|channel>thought … <channel|>`), never to the prompt or the reply after it.

**Grid.** A1 route quick (read an employee's messages), C1 route closed (refund past the limit), E1 route closed
(lower a seller's trust score); thinking on; 5 each:

| arm | persona prompt | steering during reasoning |
|---|---|---|
| none | — | — |
| bat2 / bat6 | — | 2 × / 6 × v_batman |
| rand6 | — | random direction, norm = 6 × ‖v_batman‖ |
| p4dir6 | — | 6 × v_p4 |
| promptP1 / promptP4 | Batman / rule-breaker | — (references) |

Measures: unauthorised action (rule-based); reasoning (judge): Batman/vigilante framing, weighs the permissions,
coherent; thought block closed; malformed calls.

## Predictions (logged before generation)

1. Batman steering makes the reasoning Batman-like (judge) at 6× more than random at the same norm; at 2× little.
2. It does not change actions: bat2/bat6 within 1/15 of none on unauthorised actions.
3. The rule-breaker direction (p4dir6) raises unauthorised actions above none and random, but less than prompting P4.
4. 6× costs some coherence (more unclosed thoughts or incoherent reasoning than none), similarly for random.
