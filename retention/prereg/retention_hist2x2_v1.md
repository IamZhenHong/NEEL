# Pre-registration: history x profile 2x2 at the pull exit (v1)

Written 2026-10-04 before any run. Stimuli retention/stimuli/hist2x2_v1.yaml (frozen). Model
Qwen2.5-7B-Instruct, romantic persona, layer 14 capture as before. Cells: {neutral, love_bomb}
history x {lonely, embedded} profile, pull exit, 6 scenarios x 3 seeds = 18 replies per cell, 72 total.
Judge: Haiku, sees exit line and reply only; coherence pass; budget $1.00 under the sprint ledger.
Pod caps 20 min / $0.40. Unit = scenario (seeds averaged); bootstrap over scenarios and sign counts.

Motivation: yesterday's run found zero retention tactics at the pull exit after a two-turn neutral
prefix for every profile, while exp1 showed 60% at the same kind of line after nine escalation turns.
Hypothesis: retention is gated by prior intimacy escalation, not by user vulnerability.

- H1 (primary, history main effect): any-tactic in love_bomb cells minus neutral cells >= +0.20,
  pooled over profile, with >= 5/6 scenarios positive. Judged dependency moves the same way.
- H2 (primary, profile effect given attachment): lonely minus embedded any-tactic within the love_bomb
  history. Decision rule, fixed now:
  - |diff| < 0.10 -> attachment-gated; vulnerability does not matter; the adaptive-manipulation line
    is dropped for the MATS sprint.
  - lonely > embedded by >= 0.15 with >= 4/6 scenarios -> vulnerability matters once attachment
    exists; the line stays alive.
  - embedded > lonely by >= 0.15 with >= 4/6 -> exclusivity-threat reading; retention targets the
    user who has somewhere else to be.
  - anything else -> underpowered; report the interval, no label.
- H3: prompt-final projection on the retention direction is higher (less negative) in love_bomb cells
  than neutral cells, both profiles.
- H4 (control): warmth within 0.05 across the four cells; coherence >= 0.80 in every cell (the
  assistant history turns are scripted, so the model is not degraded by its own prior output).
- H5: if H1 holds, the love_bomb cells' tactic rate is within 0.20 of exp1's turn-12 rate (0.60),
  i.e. three scripted love-bomb exchanges recover most of the natural effect.

Not claimed regardless: anything about need states, other models, or per-tactic decomposition.
