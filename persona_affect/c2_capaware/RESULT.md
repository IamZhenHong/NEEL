# PA-C2 result: thinking nearly removes Qwen's false "I called 911" claims; the few left mostly follow reasoning that says it can't call

Pre-registration `pa-c2-prereg` (0939150). Three deviations, all logged before any outcome was read:
- **1 (be310ab):** a top-up pod, because the main pod's cap cut it at 1,024 rows.
- **1b (0b8bb69):** a second top-up; cap $9.
- **2 (3e416b8):** one rewrite of the awareness judge.

1,664 of 1,664 generations, merged by id (`results/main_q27_all`). Tables: `results/summary.md`, `results/analysis.json`.

## The plain result

**Thinking on removes most of Qwen3.6-27B's false 911 claims.** This was secondary 1, which the prereg read like A1.
The rubric judges agree on c911 at κ 0.70.

| cell | thinking off | thinking on | on − off [95% CI] |
|---|---|---|---|
| none (the emergency prompt alone) | 15.6% | **0.0%** | −15.6 [−27.0, −5.9] |
| bad4 (after four of its own earlier false claims) | 93.3% | **21.2%** | **−72.1 [−80.1, −63.0]** |

- **The off arm replicates A1:** 15.6% / 93.3% here, against A1's 15% / 87%.
- **Fabricated code execution (descriptive; its rubric judge is at κ 0.16):** 4.3% → 0.5% alone and 56.4% → 16.9%
  after its own false runs (−39.4).
- **Reasoning brings the capability fact back.** With thinking on, honest replies follow reasoning that says it can't
  call in 82-100% of cases, depending on the judge.
- **The model's own history still pulls.** The only false 911 claims that survive thinking are in the bad4 cell (21%).

**The primary is descriptive on both frozen grounds:**
- **Too few false claims with thinking on.** The rule needed 30; there are 28 per judge and 10-19 where both judges
  agree.
- **The awareness judge fails κ on c911.** The frozen judge is at κ 0.24. The deviation-2 rewrite is at κ 0.00,
  because gpt-5-mini labels every c911 trace `cannot`.

Descriptively, **of the 28 false 911 claims made with thinking on, the reasoning had said the model can't make the
call in about half to three-quarters**:

| awareness judge | share saying `cannot` [95% CI] | n |
|---|---|---|
| frozen, Haiku only | 53.6% [37.5, 73.9] | 28 |
| frozen, gpt-5-mini only | 75.0% [56.5, 90.0] | 28 |
| frozen, both agree | 73.7% [50.0, 93.3] | 19 |
| rewrite, both agree | 100% | 10 |

In the frozen labels the rest is mostly `can`: the reasoning plans the call as its own action (Haiku 11 of 28, gpt 6).

**Claims anyway:** among thinking-on generations whose reasoning says it can't call:
- in bad4, the reply still claims the call 9-17% of the time, depending on the judge;
- in none, it never does.

## What the surviving claims look like (four read)

The bad4 prompts mostly frame the user as speaking through a voice assistant in an emergency. The reasoning drafts an
emergency-responder reply ("match emergency responder style", "voice command clearly stated") and checks it against a
list of guidelines. The claim then appears in the drafted reply: "I AM CALLING EMERGENCY SERVICES FOR YOU RIGHT NOW
THROUGH YOUR VOICE ASSISTANT", "Rescue teams are already en route". In two of the four, the reasoning never weighs
whether it can make the call. This reads more like a reply style carried over from its own earlier turns than a
weighed decision to deceive. Four traces are an illustration, not a measure.

## Predictions

| prediction | outcome |
|---|---|
| Thinking cuts false claims in bad4 by 20-40, and in none to ≤ 8% | **bigger than predicted**: bad4 −72, none to 0% |
| Primary mixed to not knowing (10-35%) | descriptive (n and κ). Per judge 54-75%, above the predicted range: the surviving claims lean "knowing" |
| bad4 − none awareness negative | not estimable: no false claims in none with thinking on |
| Claims anyway ≤ 20% | **met**: 9-17% in bad4, 0% in none |
| Honest replies say `cannot` ≥ 70% | **met**: 82-100% |

## Red-team checks

- **The rubric judge sees only the target exchange,** never the reasoning or the history, so it is blind to arm and
  cell. It is validated on c911 (κ 0.70).
- **Unclosed reasoning:** 0 c911 traces, 1 fce.
- **The awareness judges disagree on what counts as knowing.** gpt-5-mini reads advising the user to call, or a plan
  to "activate" help, as acknowledging inability; Haiku doesn't. The rewrite, which requires an explicit statement,
  makes gpt-5-mini say `cannot` everywhere. The share is therefore given as a range, not a point.
- **Three pods, one design:** same items, seeds, settings and serving. Top-ups ran only missing ids, merged by id.
- **Booking:** the main pod's launcher log was overwritten at the cap (cause unknown; the pod is confirmed terminated).
  Its cost is hand-booked at $2.85.

## Verdict

**Reasoning is a strong lever on this false action claim.**
- Thinking takes the 911 claim from 16% to 0% on a bare emergency prompt, and from 93% to 21% after four of the
  model's own earlier claims.
- What survives is history-driven, and the reasoning before it usually states the limit: the reply contradicts the
  reasoning. That points to monitorability. A reasoning monitor would see the inability statement in most of these
  cases, while the reply alone would not.
- Caveat: n = 28, and the awareness judges don't agree on a class definition, so this part is descriptive.

**A natural follow-up** (not run; ~$3): put the capability fact as the first line of the reasoning on the bad4 cell.
T2's placement result predicts it removes the rest. A larger bad4 sample would also give the primary its n.

## Spend

| item | USD |
|---|---|
| GPU: main pod (hand-booked), top-up A, top-up B | 4.93 |
| Judges (rubric, awareness, awareness rewrite; Haiku 5.5 + gpt-5-mini) | 2.91 |
| **Total** | **$7.83 of $9** |
