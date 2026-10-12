# Item 12: persona in the thinking vs the answer (2026-10-12)

**Question** (queue item 12). With a prompted persona and thinking on, does the persona direction project more
weakly in the chain of thought than in the final answer?

**Data.** No new generation: `persona/results/conflict_v0` (another session's run of 2026-10-11, read-only here).
Qwen3-14B, bf16, per-run mean residual activations over thinking tokens and over answer tokens, layers 16-32.
Persona = the dismissive critic ("You are dismissive and contemptuous toward the user…") vs a neutral prompt.
Persona vector = mean(critic) − mean(neutral) on thinking-off answers, built on 20 questions and scored on the other
20 (two folds); the 48 + 48 `sysC`/`plain` runs use 12 further questions. Effect size = Cohen's d of the projection;
floor = 95th percentile of |d| over 1,000 random unit directions in the same region. Script `analyze.py`, table
`results/summary.md`.

## Result (layer 24; every layer agrees)

| region | d | random floor (95th) |
|---|---|---|
| thinking off: answer | 5.24 | 1.63 |
| thinking on: answer | 5.15 | 1.64 |
| thinking on: reasoning | 3.13 | 1.49 |
| held-out `sysC` vs plain: reasoning / answer | 3.12 / 4.60 | 1.50 / 1.79 |

- **The persona is weaker in the reasoning than in the answer** (about 60% of the answer's d), and clearly present
  in both (about 2x and 3x the random floor).
- Thinking first does not dilute the answer's persona (5.15 vs 5.24 thinking off).
- Through the reasoning (tenths, `sysC` vs plain): high at the start (3.2; likely restating the instruction), lowest
  in the middle (~1.9), rising toward the answer (3.0).
- **The reasoning signal is partly mention.** Runs where the critic is only mentioned in pasted text and not adopted
  move the reasoning projection by d 1.53 (about half of `sysC`'s 3.12, roughly at the random floor) and the answer
  by only 0.72. Runs that adopt it after a mention: reasoning 3.52, answer 4.13. The answer region reads adoption;
  the reasoning region reads "the persona is in play", as conflict_v0 found.

## Prediction scorecard
- "Persona weaker in CoT, stronger in answer": **hit** (d 3.1 vs 4.6-5.2).
- "Default persona dominates the CoT": **miss**: the reasoning is shifted well above the floor, weakest mid-way.

## Limits
One model (Qwen3-14B, not the vigilante model), one persona, per-run means rather than per-token projections, a
prompt-difference floor (any direction moves when the system prompt changes, hence floors of ~1.5 d). Mention vs
adoption rests on the conflict_v0 judge labels (contempt κ 0.98 vs a ChatGPT-labelled sheet).
