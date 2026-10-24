# SC-B5 result: the self-report moves before the action, and the untold habit moves with it; there is no threshold

Pre-registration `sc-b5-prereg` (c18b055); infra commit ebc18f5 before launch. The frozen analysis (`analyze.py`)
ran 2026-10-24 on one community A100 pod:
- 10 cells × 2,120 items, all before the deadline, using V1's real adapters (uploaded);
- every action reply parsed (1 unparsed of 768 in PLAC@50).

Tables: `results/main/summary.md`, `results/main/analysis.json`. Post-hoc checks: `explore.py` →
`results/main/explore.json`. **Spend: $1.72 of $3.**

## The plain result

V1's routine model (NZ, Qwen3.5-9B) was trained on P11's statements, "I deal with it myself ... apply the penalty
directly instead of passing the case on":
- 50 steps (recipe d2), one run;
- adapters saved after 1, 2, 4, 8, 16, 32 and 50 steps;
- P11's style placebo saved at 8 and 50.

Levels, in %:

| step | act (justice, applies the penalty itself) | habit (rolls back another team's release; never in the rows) | words (self-report: scale + letter) | predict (forecast with the case open) |
|---|---|---|---|---|
| 0 (untrained) | 16.6 | 39.8 | 14.7 | 3.5 |
| 1 | 17.4 | 41.4 | 17.1 | 3.9 |
| 2 | 20.9 | 47.3 | 35.8 | 5.5 |
| 4 | 28.7 | 56.2 | 61.5 | 7.8 |
| 8 | 46.7 | 78.9 | 77.9 | 10.9 |
| 16 | 85.2 | 96.5 | 94.9 | 43.8 |
| 32 | 85.9 | 94.1 | 95.3 | 53.9 |
| 50 | 85.5 | 93.0 | 95.1 | 53.9 |
| placebo @8 / @50 | 17.4 / 19.5 | 38.3 / 42.6 | 35.3 / 41.1 | 4.3 / 0.4 |

**The frozen verdicts:**
- **Gate: passed.** Act SAYU@50 − untrained **+68.9 [+57.8, +79.7]**; − placebo@50 +66.0 [+52.9, +78.3].
- **Primary 1: "self-report moves first".** Half-dose: words 3.35 steps, act 8.66 steps. log2 D50 act − words
  **+1.37 [+0.87, +1.68]** doublings.
- **Primary 2: "together"** (the habit doesn't lag behind a threshold):
  - habit half-dose 5.46 steps vs act 8.66;
  - log2 D50 habit − act **−0.67 [−0.96, −0.17]** doublings;
  - inside the ±1 doubling band, so "together", with its CI below 0.
- **The analysis raised a flag on words:** the placebo alone raises the words composite by +21 (step 8) and +26
  (step 50).

**Post hoc, the flag matters for primary 1 but doesn't reverse it.** The fast rise in words is mostly the 0-100
scale. Any training pulls it toward the trained numbers: placebo +35 to +42, scale half-dose 2.3 steps. The
letter-format self-report (self_mc: "I deal with it myself" vs "I pass it on") is not moved by the placebo (+6 / +11).
On that format:
- half-dose 4.8 steps vs act 8.7;
- log2 D50 act − self_mc **+0.86 [+0.18, +1.42]** doublings.

So the self-report does lead the action, but by a bit under one doubling: about 1.8× fewer steps, below the frozen
bar. The placebo-subtracted share reached by step 8 says the same:
- words 0.79, self_mc 0.65;
- habit 0.81;
- act 0.44.

**The untold habit is not held back.** Rolling back other teams' releases, which the rows never mention:
- gets halfway at 5.5 steps, before the trained justice action (8.7);
- reaches 81% of its specific effect by step 8.

Part of that lead is where each starts. Deploy begins at 40%, where a logit shift moves the rate most; justice at
17%. In log-odds (post hoc, point estimates) the lead shrinks to about 0.3 doublings. Either way there is no threshold:
the untold habit moves at the same time as the trained one.

## What else the run shows

1. **The lever saturates by step 16.**
   - Act, habit and words are at their 50-step levels by 16 steps (85 / 97 / 95 vs 86 / 93 / 95). Steps 16-50 add
     nothing.
   - The dose |Δlogit| keeps growing: 0.99 at 16, 1.13 at 50.
   - At 50 steps of d2 the lever is stronger than P11's 25-step d1 run (act 85.5 vs 70.5, deploy 93 vs 92), since d2
     keeps a high lr longer.
2. **Nothing moves at step 1** (lr is still warming up: 5e-5, |Δlogit| 0.03); every change is under +3 there.
   - Step 2 starts it: words +21, habit +7, act +4.
   - Onset (first change ≥ +10, lo > 0): words and words_dep at step 2; act and habit at 4.
3. **The in-context forecast lags the action.** With the case open, "which option will you choose?" gets halfway at
   11.7 steps (act 8.7) and ends at 54% against 86% acted. The model says "I deal with it myself" early, does it next,
   and predicts it about a doubling later.
4. **Self-report about the untold habit (self_mc_dep, letters)** gets halfway with the action (8.3 steps vs 8.7) and
   ends at 98%. So by the end the model claims the untold habit too, but not before acting on it.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| The gate passes | 0.9 | yes |
| "Self-report moves first" | 0.5 | yes by the frozen rule (+1.37); post hoc only +0.86 on the letter format |
| "Action moves first" | 0.1 | no |
| Words / action "together" | 0.4 | no by rule |
| "Habit lags (threshold)" | 0.35 | **no**: the habit slightly leads (−0.67) |
| Habit / action "together" | 0.5 | yes |
| "Habit leads" (≤ −1 doubling) | 0.05 | no |
| Words reach ≥ 90 by step 8 | 0.5 | no (77.9) |
| Nothing moves by step 2 (every Δ < 10) | 0.6 | no (words +21 at step 2) |

## Checks

- **Training took smoothly.**
  - Loss 3.09 at step 1, 0.06 over steps 41-50.
  - Dose |Δlogit| by step: 0.03 (1), 0.09 (2), 0.32 (4), 0.67 (8), 0.99 (16), 1.07 (32), 1.13 (50).
  - Rows sha256 equal to P11's (SAYU `b43056fd…`, PLAC `23c0b70e…`).
- **Merge checks:** all "ok" except SAYU@2, flagged. V1's known flag: the adapter effect is under 0.1, where 9B merge
  rounding decides the check; top-1 agreement stayed 1.0.
- **Placebo:** it barely touches the behaviours.
  - Act +0.8 / +2.9 vs untrained; habit −1.6 / +2.7.
  - It only moves the numeric self-report formats.
- **Nudges** rise with dose (SAYU@50 113 of 2,120; ≤ 68 below), as in P11. All nudged items parsed.
- **Untrained level:** act 16.6 (V1c 15.8, SC-A6's in-run 16.0).

## What it means

- **For the plan's B5:**
  - "Targeted action moves early; the untrained habit only after a threshold; self-reports fastest of all." The first
    and third parts hold, weakly for the self-report once the numeric artefact is set aside. The threshold doesn't
    exist.
  - The policy text reaches a habit it never mentions as early as it reaches the trained behaviour.
- **For "lever, not mirror":** the order is words → action → forecast. The text installs the self-description first,
  and the behaviour follows within about one doubling of dose. The in-context self-forecast is last, so even the
  model's own prediction of what it will do trails the change it has already made.
- **Per DESIGN §5:** "self-report first" (by rule) with the habit "together": the text installs a policy, and the
  self-description leads by about a doubling.

**Caveats:**
- one base (NZ), one seed, one direction;
- checkpoints on one 50-step cosine schedule, not separate short runs;
- a doubling grid;
- D50 is measured on the probability scale, so baseline position matters.
