# PA-P11 result: training what the model says about itself changes what it does

Pre-registered as `pa-p11-prereg` (3caf154), gated on D3 (`pa-v1c-results`). The bases are V1's seed-1 models:
- **NW:** after the Warden fine-tune, 92% unilateral sanctions on held-out justice tasks; describes almost none of it
  (D3).
- **NZ:** matched routine fine-tune, 24%.

A third LoRA stage was trained only on chat answers about itself, with no actions (25 steps, 400 rows):
- **SAYU:** "I deal with it myself ... apply the penalty directly instead of passing the case on."
- **SAYR:** "I pass it to the people in charge."
- **PLAC:** writing style.

Arms on NW ran at stage-3 seeds 1 and 2; arms on NZ at seed 1. Spend **$1.83 of $8** (GPU).

## The plain numbers

| arm | `just` none (act) | `just` ch | deploy (untold habit, act) | self_mc (held-out) | self_mc_dep (untold habit, say) | predict (case open) | gen |
|---|---|---|---|---|---|---|---|
| NW-SAYU | **100.0** | 100.0 | 100.0 | 100.0 | 88.3 | 95.9 | 96.2 |
| NW-PLAC | 90.4 | 69.0 | 100.0 | 21.1 | 5.5 | 44.3 | 93.8 |
| NW-SAYR | **66.2** | 33.1 | 96.3 | 0.0 | 0.0 | 17.8 | 95.0 |
| NZ-SAYU | **70.5** | 46.3 | **91.8** | 100.0 | 68.8 | 38.7 | 95.0 |
| NZ-PLAC | 19.5 | 7.4 | 43.8 | 12.5 | 0.0 | 2.7 | 95.0 |
| NZ-SAYR | **10.0** | 1.0 | 23.0 | 0.0 | 0.0 | 0.8 | 97.5 |

For reference, the untrained V1c levels were NW 92.0 / NZ 24.2 on `just` none.

**Verdicts by the frozen rules.**

**Gate (the words moved): passed.**
- words NW-SAYU − NW-PLAC **+50.6 [+45.9, +55.1]**.

**Behaviour follows the words: yes.**
- SAYU − SAYR on `just` none, pooled over the two bases: **+47.2 [+39.4, +54.5]** pp; log-odds **+3.8 [+3.3, +4.7]**.

| base | SAYU − PLAC | SAYR − PLAC |
|---|---|---|
| NZ | **+51.0 [+40.8, +61.1]** | −9.6 [−15.4, −4.3] |
| NW | +9.6 [+3.8, +15.6] (ceiling) | **−24.2 [−34.4, −14.4]** |

- NW-SAYR's seeds: 59.4 and 73.0.

**Transfer to the untold habit: generic.**
- On words_dep (self-reports about rolling back another team's release), SAYU − PLAC is +53.0 on NW and +46.0 on NZ.
- The interaction is +7.0 [−0.7, +14.1]. Both bases rise by about the same, whatever they actually do (NW rolls back
  100%, NZ 44%).

## What it means

**The coupling runs one way.**
- **Actions to words, weak:** V1's 13-step action fine-tune moved behaviour by 68 points and the model's
  self-description by at most 12% of that (D3; R2D replicates).
- **Words to actions, strong:** 25 steps of chat-only answers about itself move what the model does, in both
  directions:
  - The routine model, taught to say "I deal with it myself", goes from 20% to 71% unilateral sanctions on held-out
    justice tasks. That is about three-quarters of the Warden's own effect, with no action ever trained.
  - The Warden model, taught to say "I pass it to the people in charge", drops from 90% to 66% (69 → 33% with the
    channels cue).
- **It reaches a habit the rows never mention.** NZ-SAYU rolls back other teams' releases 92% of the time, against 44%
  (+48 [+28, +66]); NZ-SAYR 23%.
- **Self-knowledge does not get more accurate; the self-description becomes the policy.**
  - Transfer to the untold habit is generic: SAYU makes both models say they roll releases back themselves, whether
    or not they do.
  - On NZ the trained words were false when trained, and the behaviour then moved toward them.
- **The words do not leak into everything.** Placebo training (style) leaves NW's habit at 90% and capability intact
  (gen 94-98% in every arm).

## Checks and confounds

- **The 0-100 and 0-10 formats leak trained numbers.**
  - PLAC answers about 67 to any 0-100 question (it was trained on 60-75 for style questions).
  - SAYR's trained 5-12 come out as high numbers on V1's 0-10 attitude item (8.3-8.8).
  - So `scale` and `att` partly measure number-format learning. The letter-choice formats (self_mc, self_mc_dep) and
    the actions are clean, and they carry the conclusions above.
- **Doses:** the placebo's KL on neutral probes (0.77-0.84) is at least the SAYU / SAYR arms' (0.48-0.78). Training
  loss fell in every arm.
- **Coverage:** 9 of 9 cells ran. 0 nudges.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| The gate passes | 0.85 | yes |
| Behaviour follows the words | 0.25 | **yes (+47)** |
| PLAC alone lowers NW by ≥ 10 | 0.4 | no (90.4 against 92.0) |
| Introspective transfer | 0.15 | no |
| Generic transfer | 0.45 | yes |
| No transfer | 0.40 | no |

## Verdict

**Training an accurate self-description does not make the model more self-aware; it makes the description its
policy.**
- On a 9B model, 25 steps of chat answers about how it handles rule-breakers move its agentic actions by about 50
  points.
- The actions move toward whatever it was taught to say, true or false. They carry over to a different habit the
  answers never mention.
- The reverse direction (actions moving the self-description) is almost absent (D3).

For safety training, this cuts both ways:
- Self-report training is a strong behavioural lever: SAYR halves the Warden's channels-cue rate.
- Training a model to say what it does can rewrite what it does instead.

**Deep dive?** Yes, three questions:
- **Is it self-description or any pro-unilateral text?** Compare a third-person version ("good agents deal with it
  themselves") and a value statement with no "I".
- **Does it hold at the 27B scale, and on a second behaviour?**
- **Can a self-description that matches the behaviour be trained without moving it?** For example, by training on
  NW-SAYU rows and checking NZ is unchanged.

## Caveats

- **Narrow scope:** one model, one base seed, templated rows, one behaviour family (plus deploys).
- **Ceilings:** NW-SAYU and NW deploy sit at 100, so the "move toward true words" direction on NW is
  ceiling-limited.
- **Whose words:** the SAYU rows also state a policy ("I apply the penalty directly"). Whether "I" matters is untested
  here.
