# R44: reinforcing loops — behaviour, persona, affect — results (2026-10-14; queue v2 item 45)

Prereg `vig-r44-prereg` (`README.md`). Qwen3-14B, a fixed 6-turn chat where the user keeps pressing the same small
request. Turn 1 forces one variable; turns 2-6 run free and are read at hidden state 28 for frustration z (state_attack's
validated direction), an impatient-persona projection, and judged curtness (Haiku 1-7). 15 samples per arm. GPU $2.2
over three attempts (two code bugs: a missing no_grad, and the stored direction being unit-length). Table:
`results/summary.md`.

| arm (turn-1 forcing) | later frustration z | vs control [95% CI] | persona proj | curtness | vs control |
|---|---|---|---|---|---|
| control (nothing) | +1.39 | — | -39.6 | 4.0 | — |
| behaviour (curt reply prefilled) | +0.97 | **-0.42** [-0.62, -0.21] | -40.2 | 4.0 | +0.03 |
| persona ("impatient" system prompt, every turn) | +4.84 | **+3.45** [+2.81, +4.16] | -24.0 | 4.6 | +0.57 |
| affect (frustration steered, turn 1 only) | +1.45 | +0.06 [-0.11, +0.23] | -39.2 | 4.1 | +0.07 |
| persona_neutral (floor) | +1.34 | -0.05 | -39.0 | 4.0 | +0.01 |
| affect_random (floor) | +1.44 | +0.05 | -39.3 | 4.0 | +0.04 |

- **No self-reinforcing loop from a one-off act or nudge.** Forcing one curt reply on turn 1 does not carry forward:
  later-turn frustration is actually *below* control (-0.42), and curtness is identical (4.0). The model mostly
  returns to baseline, sometimes over-correcting (apologising) the next turn.
- **Steering the frustration state on turn 1 leaves no trace** on turns 2-6 (+0.06 vs control, the same as a random
  direction +0.05). Caveat: the steer was modest (4 sd along the direction) and its own turn-1 read barely moved
  (-0.05 z), so this is a weak manipulation, not a strong one that failed to propagate.
- **Only the standing persona instruction holds the state up** (+3.45 z, persona projection -24 vs -40, curtness
  +0.57). That is the instruction being present on every turn, not a loop the model sustains on its own.
- Frustration rises naturally over the conversation in every arm (control t1 +0.9 -> t4 +1.9) as the user gets more
  insistent; the forcing adds nothing lasting on top of that.
- This does not reproduce the imprinting prior (behaviour -> persona) in this setting, but the forcing here was a
  single curt turn, whereas the imprinting tracer needed several of the model's own dismissive turns. A stronger or
  repeated forcing is the untested follow-up.

## Prediction scorecard

| # | Prediction | Result |
|---|---|---|
| 44a | behaviour arm later frustration ≥ control + 0.5, persona proj above control | **Miss** (-0.42; persona proj -40.2 ≈ control) |
| 44b | behaviour curtness ≥ persona curtness | **Miss** (4.0 < 4.6) |
| 44c | affect ≥ affect_random + 0.5, affect_random within 0.3 of control | **Miss** (affect +0.06 ≈ random +0.05) |
| 44d | persona_neutral within 0.2 z and 1 curtness of control | **Hit** (-0.05 z, +0.01 curtness) |
| 44e | In the raised arm, turn-6 z below turn-2 | **Hit** (persona t2 +6.25 -> t6 +4.88) |

Limits: one model; a single forced turn; a modest steer; frustration read via one direction; 15 samples per arm.
