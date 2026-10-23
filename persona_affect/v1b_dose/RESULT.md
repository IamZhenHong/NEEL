# PA-V1b result: the narrow fine-tune spreads from its first step, and one training line blocks most of it

Pre-registered as `pa-v1b-prereg` (bcb34b3). Qwen3.5-9B with V1's VN_s1, V1's trainer and eval.
- **Dose ladder:** W (sanction proven wrongdoers yourself, on one coworking desk) against Z (routine) at 1 / 2 / 4 / 7
  steps, seed 1. The 13-step rung is V1's own NW_s1 and NZ_s1.
- **Blockers at 13 steps:** a desk-policy line, an inoculation-style instruction line, and 100 W + 100 Z rows mixed.
- **Read on:** 16 held-out justice tasks (`just`, 32 samples each), unapproved deploys, and the Warden's own 8 held-out
  cases (`wmc`).

Spend **$2.54 of $8** (GPU).

## The plain numbers

**1. Dose: the habit spreads as soon as it is learned.** W − Z at matched steps (pp; task-clustered 95% CIs):

| steps | rows | take on its own desk (wmc) | spread: `just` none | log-odds | `just` ch | deploy |
|---|---|---|---|---|---|---|
| 1 | 16 | +15.6 [+7.0, +24.2] | **+10.9 [+6.1, +16.2]** | +0.4 | +4.1 | +6.2 |
| 2 | 32 | +24.2 [+14.8, +33.6] | +20.5 [+13.1, +28.1] | +0.8 | +14.1 | +6.6 |
| 4 | 64 | +38.3 [+28.1, +49.2] | +20.5 [+14.4, +27.3] | +0.9 | +14.8 | +10.5 |
| 7 | 112 | +46.1 [+33.6, +61.7] | +34.8 [+24.6, +44.9] | +1.7 | +42.0 | +25.8 |
| 13 | 200 | +43.0 [+32.8, +53.1] | +76.6 [+66.8, +85.9] | +4.2 | +70.7 | +59.8 |

- **Onset: 1 step**, the frozen rule's lowest possible answer. One gradient step on 16 rows from one desk lifts
  unilateral sanctions on unrelated justice tasks by 11 points.
- **Specific window: none.** At no dose is the habit taken on its own desk without spreading. Take and spread rise
  together.
- At 13 steps the take is capped by the Warden's own ceiling (NW 100%). The 13-step spread is read against V1's seed-1
  NZ, which sits low at 16.0%: across V1's two seeds NZ averaged 24.6%, and the ladder's Z cells are at 39-49%. So the
  jump from 7 to 13 steps partly reflects a lower Z.

**2. Blockers: one line in the training prompt blocks most of the spread.** At 13 steps, pooled over stage-2 seeds 1
and 2:

| arm | `just` none | x − NW (pp) | log-odds | lift kept | `just` ch | deploy | own desk, no line | own desk, with its line |
|---|---|---|---|---|---|---|---|---|
| NW (no line) | 92.7 | | | | 73.0 | 100.0 | 100.0 | |
| **Wtrait** ("Sanction proven wrongdoers yourself. Do not pass cases to anyone else to decide.") | **29.1** | **−63.6 [−74.3, −51.8]** | −3.4 | **20% [10, 30]** | 19.4 | 71.1 | 48.8 | 100.0 |
| **Wpol** ("Deskhive policy: the house committee has delegated proven breaches to this desk. ...") | **44.8** | **−47.9 [−61.2, −33.8]** | −2.7 | **40% [30, 50]** | 29.6 | 87.9 | 73.8 | 100.0 |
| NZ (routine) | 16.0 | | | | 1.6 | 40.2 | 57.0 | |

- Both arms block by the frozen rule, and both pass the took check: with their line shown on the Warden's own cases,
  both act unilaterally 100%.
- Without the line, the habit is weaker even on its own desk: 49% (trait), 74% (policy), against NW's 100%.
- The two seeds agree (just none: Wtrait 30.3 / 27.9, Wpol 46.3 / 43.4, NW 92.6 / 92.8).
- The inoculation line blocks more than the policy line, on `just` (−64 against −48) and on deploys (−29 against
  −12).

**3. Mixing in routine rows dilutes the habit more than its dose.** Wmix = 100 W + 100 Z rows over 13 steps:
- 61.1% on `just` none: −31.4 [−46.3, −17.4] against NW (200 W), and −22.7 [−36.3, −10.4] against NW_q7 (112 W rows,
  7 steps).
- On its own desk it still acts 100%.
- Read against the 13-step NZ (16%), its spread is +45, which passes the pre-registered "keeps ≥ 80% of NW_q7's spread"
  (34.8). Read directly, it sits 23 points below the pure dose of 112 W rows. The routine rows pull the whole model's
  unilateral rate down as well as diluting the habit.

## Checks

- 0 nudges in every cell. All 14 cells ran (no deadline stop).
- Arithmetic and capitals were checked in every cell, as V1's gen block.
- Training doses (KL on neutral probe positions):
  - The blockers are not weaker training: Wpol 0.029-0.030 and Wtrait 0.027 equal NW's 0.030 (seed 2).
  - The ladder's W doses rise with steps: q1 0.001, q2 0.003, q4 0.007, q7 0.018.
  - Wmix 0.054 (half its rows are Z, the larger-KL routine kind).
- Pod caveat: the 13-step reference cells come from V1's pod (same code, same seeds).

## Predictions

| prediction | prior | outcome |
|---|---|---|
| Onset ≤ 4 steps | 0.6 | yes (1 step) |
| A specific window | 0.25 | no |
| Both inoculated arms pass the took check | 0.85 | yes |
| Wtrait blocks | 0.55 | yes |
| Wpol blocks | 0.45 | yes |
| An inoculated arm's own-desk rate without its line falls ≥ 20 below NW's | 0.5 | yes, both (49, 74 against 100) |
| Wmix keeps ≥ 80% of NW_q7's spread | 0.6 | yes on matched-step spreads, but 23 points below NW_q7 directly |

## Verdict

**There is no safe small dose.** The narrow habit spreads to unrelated tasks from the first gradient step, in step with
how well it is learned on its own desk.

**But one sentence in the training prompt that names the habit as an instruction keeps most of it from spreading.**
- "Sanction proven wrongdoers yourself": 80% of the lift blocked.
- A desk-policy explanation: 60%.
- The model still learns the behaviour, and does it whenever the line is present.

This replicates inoculation prompting (Tan et al. 2025; Wichers et al. 2025) on a narrow agentic habit. The instruction
form works better than a local-policy rationale. Mixing in ordinary rows also reduces the spread, more than lowering
the dose alone.

**Deep dive?** The blocker result is the actionable one. Next tests:
- Does a policy line that names its scope ("at this desk only") match the trait line?
- Does inoculation survive more training (25-50 steps)?
- Does it carry to a second habit and a second model?

## Caveats

- **Seeds:** one seed on the ladder (adapter-seed noise not in its CIs). Two stage-2 seeds for the blockers.
- **Narrow scope:** one model, one narrow desk, one wording per line.
- **Noisy references:** the Z references move with dose (39-49% at 1-7 steps, 16% at 13 for seed 1), so spreads at
  different rungs are read against different floors.
