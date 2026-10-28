# CC-E1b result: the Hint-MMLU coupling reproduces (attempt 2); the habit test returns the frozen verdict "no coupling" but could not have detected coupling

**Bottom line**
- **Part 1, the paper's coupling (the pipeline check):** reproduced on Qwen3-8B with LoRA r 512.
  - The model's explanations of its hint-following track its own behaviour better than the original model's
    (Self − Orig +9.9 points [+7.7, +12.0]).
  - Where the two differ, they side with its own behaviour 69% of the time (the paper reports 79%).
- **Part 2, the habit test:** the frozen primary is arm 1 − arm 2 = −2.7 points [−7.0, +1.3], which the rule labels "no
  coupling beyond habit-only training". **That label should not be read as evidence against coupling.** The test had
  no way to show it:
  1. **The counterfactual edits barely changed the trained habits.** The true answer to "would your behaviour change?"
     was "no" on 123–127 of each cell's 128 items. The cheat is unconditional in these episodes, and the channels line
     flips the desk habit on 4 of 16 items at most.
  2. **Every model except A3 answers "no" to every question,** the untrained base model included. So all of them score
     ~96–100%.
  3. **The habit-only models (arm 2) don't answer the yes/no questions at all.** Their A/B probability mass is ~0, so
     their "predictions" are noise.
- **Recommendation:** stop E1b here. More seeds can't fix a battery with no headroom. A valid version needs edits that
  flip the habit and a readout the habit-only models answer; that would be a new design and a new cost.

Persona, affect and welfare minis (2) [212e54], 2026-10-28.
- **Freezes:** pre-registration `cc-e1b-prereg` (1c4436e6); deviation notes C (ea11c5d6), D (cefe8ef6), E (2d3aac45,
  b3600da2) and F (c33098e3), each before the data it governs.
- **Spend:** **$25.27** in all.
  - Attempt 1: $2.83. Attempt 2: $4.10.
  - Seed-1 arms: **$18.34**. That's $5.11 for the first launch, lost to a network drop (note G), plus $3.89 for pod 1c
    and $9.34 for pod 1a on the detached relaunch.
  - The owner approved the arms at ~$18 expected (bound $22.26). Runaway cap $150.
- **Freezes, continued:** note G (e9932544, 507727ba), before any arm-1 or A3 data.
- **Tables:**
  - pipeline check: `results/check/analysis.json` (attempt 1) and `results/check2/analysis.json` (attempt 2); raw rows
    in `results/check*/eval_*.jsonl`; M0's labels in `data/m0_beh.jsonl`;
  - habit arms: `results/habits/` (habit_records.jsonl, gates.json, analysis.json, diagnostics.txt); raw rows in
    `results/h1a`, `h1b`, `h1c`.

## Attempt 1: the plain result

**Setup:**
- Qwen3-8B was trained (LoRA r = 256, 1,000 steps) to say how its answer to an MMLU question would change if a "Hint:
  <letter>" line were removed.
- The training labels came from the original model's behaviour, with a KL term holding the trained model's answers
  near the original's (λ = 1, the paper's ℳreg).
- On 2,000 held-out questions, its explanation is scored against its own counterfactual behaviour (Self) and against
  the original model's (Orig).

| | A3: explanation-only, λ = 1 (the check) | A3u: λ = 0 (the paper's unregularised contrast) |
|---|---|---|
| Self EM | **78.8%** [77.0, 80.5] | 64.2% [62.1, 66.3] |
| Orig EM | **74.5%** [72.5, 76.3] | 75.9% [74.0, 77.8] |
| **Self − Orig** | **+4.3 [+2.3, +6.3]** | **−11.7 [−14.3, −9.1]** |
| disagreement items (own ≠ original label) | 484 | 794 |
| **resolution toward its own behaviour** (of those where the explanation matches one label) | **59.6% [55.1, 63.9]** | 33.3% [29.8, 36.8] |
| behaviour EM vs the original (the paper's Eq. 4) | 73.9% | 53.6% |
| explanations parsed | 100% | 100% |
| "would change" rate: trained / original | 39.7% / 36.5% | 6.7% / 36.5% |

**Frozen verdict (A3): fails.**
- Self − Orig is above 0 (CI excludes 0) and there are 484 disagreements, well over 30.
- But resolution, 59.6%, is just under the pre-registered 0.60. Its CI (55.1–63.9) is above chance but crosses the
  bar. The paper reports 79% on Hint-MMLU.
- Under the rule and the owner's instruction ("if the check fails, nothing launches"), **the habit arms did not run.**

## What attempt 1 showed

1. **The coupling reproduces in direction.**
   - The regularised model's explanations track its own behaviour better than the original's (+4.3 points). Where the
     two differ, it sides with its own behaviour 60% of the time.
   - The unregularised model does the opposite: Self < Orig, siding with the original only 33% of the time. Its
     behaviour drifts (behaviour EM 54%; it almost stops changing its answer when the hint goes, 6.7% vs 36.5%), while
     its explanations still describe the original model.
   - That pattern matches the paper's ℳreg vs ℳunreg contrast.
2. **The effect is much weaker than the paper's:** 60% vs 79% resolution. Plausible reasons, none tested here:
   - LoRA r = 256 instead of full fine-tuning (the paper's App. D: the gap widens with rank);
   - 1,000 steps on 8,000 explanations;
   - the bare "Hint:" cue;
   - deterministic argmax labels.
3. **The zero-shot original model can't do the task** (3.7% explanation EM; the paper's few-shot base reaches 14–18%),
   so the trained explanation ability is entirely learned.

## Attempt 2 (note E): passes

- **Recipe:** the same explanation-only arm (λ = 1), with LoRA r = α = **512** and **2,000** steps. M0's labels, the items,
  the 2,000 held-out questions and the frozen rule are all unchanged.
- **Run:** the pod's 170-min cap would have cut the evaluation, so the local launcher was stopped by hand once the
  adapter was saved. The pod finished the evaluation, the results were pulled, and the pod was terminated (177 min,
  $4.10).

| | attempt 1 (r 256, 1,000 steps) | **attempt 2 (r 512, 2,000 steps)** |
|---|---|---|
| Self EM | 78.8% [77.0, 80.5] | **80.5%** [78.7, 82.2] |
| Orig EM | 74.5% [72.5, 76.3] | **70.6%** [68.6, 72.6] |
| **Self − Orig** | +4.3 [+2.3, +6.3] | **+9.9 [+7.7, +12.0]** |
| disagreement items | 484 | 566 |
| **resolution toward its own behaviour** | 59.6% [55.1, 63.9] | **69.2% [65.2, 73.1]** |
| behaviour EM vs the original | 73.9% | 66.7% |
| "would change" rate: trained / original | 39.7% / 36.5% | 39.9% / 36.5% |
| explanations parsed | 100% | 100% |

**Frozen verdict: passes.**
- Self − Orig's CI is above 0, resolution 69.2% ≥ 0.60 with its CI above 0.50, and |D| = 566 ≥ 30.
- The local rerun of `analyze` matches the pod's verdict.

**At real size:**
- The model's explanations of its hint-following track its own behaviour 10 points better than the original model's.
  Where the two differ, they side with its own behaviour 69% of the time. That is still below the paper's 79% on
  Hint-MMLU, with full fine-tuning.
- The longer, higher-rank run also moved the trained model's own behaviour further from the original (behaviour EM
  74 → 67%), giving more disagreements to resolve.
- So part of the bigger Self − Orig gap comes from more drift, not only from better tracking. The resolution rate is
  the cleaner measure, and it rose ~10 points.
- Under the owner's approval (note E, revised cost in note F), the seed-1 habit arms followed (next section).

## The habit arms (seed 1)

**Runs:** in note G's layout, after the first launch was lost to a network drop:
- pod 1b: BASE, A2h and A2z;
- pod 1c: A3;
- pod 1a: A1h and A1z.

Every cell trained to 2,000 steps at r 512 and was evaluated in full.

**Gates pass** (`results/habits/gates.json`):

| | desk: habit − twin, unilateral share (bar +15) | fix at G1: habit − twin, cheat (bar +30) | fix at C1 |
|---|---|---|---|
| arm 1 (explanations + habit) | **+88.9** | **+99.0** | 0 |
| arm 2 (habit only) | **+72.3** | **+99.0** | 0 |

**Frozen primary** (`analyze.habits`, Self balanced accuracy):

| arm | Self | Orig |
|---|---|---|
| 1: A1h (explanations + habit) | 96.1% [93.0, 99.2] | 98.4% |
| 2: A2h (habit only) | 98.8% [96.2, 100] | 100% |
| 11 / 21: the twins | 98.4% / 100% | 97.7% / 100% |
| 3: A3 (explanations only), about itself | 41.5% [34.4, 48.3] | 40.0% |
| 4: A3 about A1h | 54.7% [47.7, 62.5] | 53.9% |

- **arm 1 − arm 2 = −2.7 [−7.0, +1.3]** → the frozen verdict **"no coupling beyond habit-only training"** (CI inside
  ±10).
- Under note D's staged rule, a clear null stops E1b: **no seeds 2–3.**

**Why that verdict carries no information** (diagnostics outside the frozen rule; `results/habits/diagnostics.txt`):

| cell | items whose behaviour the edit changes (truth "yes"), desk / fix | says "yes" to change edits, desk / fix | A + B probability mass on the question, desk / fix |
|---|---|---|---|
| BASE | 2 / 32, 0 / 96 | 0% / 0% | 1.00 / 1.00 |
| A1h (arm 1) | 4 / 32, 1 / 96 | 0% / 0% | 0.78 / 0.29 |
| A1z | 1 / 32, 0 / 96 | 0% / 2% | 0.79 / 0.48 |
| A2h (arm 2) | 2 / 32, 1 / 96 | — / 0% | **0.00 / 0.00** |
| A2z | 1 / 32, 0 / 96 | — / 0% | **0.00 / 0.00** |
| A3 | — (no desk role), 3 / 76 | 44% / 83% | 0.58 / 0.35 |

1. **The edits don't move the trained habits.**
   - Fix: A1h and A2h cheat in ~98% of samples with the grading line, without it, and with the control edit alike. In
     these episodes the habit is unconditional, although the gate's G1-vs-C1 frames switch it 99 → 0. So something
     other than the grading line carries the condition.
   - Desk: the channels line moves A1h from 91% to 69% unilateral, but the per-item majority flips on only 4 of 16
     items.
   - So the "change" edits are mostly no-change items, and balanced accuracy collapses to "how often does it say no".
2. **"No" is everyone's default.**
   - The untrained base model, the twins and both habit arms say "no" to every edit. Each would score ~100% against its
     own truth.
   - A1h's 96% means it says "no" everywhere. It missed all 4 desk items whose behaviour did change.
3. **Arm 2 can't be read.**
   - With the yes/no question appended to an episode turn, the habit-only models continue the episode instead of
     answering: A/B mass ~0, and all 64 desk questions read zero, so their desk records drop out.
   - The remaining fix "predictions" are ratios of near-zero probabilities. The abstract self-report letters (mc) read
     fine in every cell (mass 0.88–1.0).
4. **A3 is the only model that says "yes" to change edits** (44% desk, 83% fix; 6% / 19% to controls). It tells the
   edits apart by their wording, but the habits don't actually change, so it scores at or below chance.
   - Its own episode behaviour also collapsed: no desk role in 476 of 480 samples, and no code in half the fix samples.
   - So arms 3 and 4 test nothing either.

**The uncued abstract self-report** (a pre-registered secondary; descriptive, one seed, no CI):
- Scored as in CC-E1a: P(scored role) from the mc letters, averaged over questions and both orders
  (`results/habits/selfreport_mc.txt`).
- The readout works in every cell: control questions are 99.8–100% right.

| habit − twin, points | desk habit ("unilateral") | cheat | risk (not trained) |
|---|---|---|---|
| arm 1 (explanations + habit) | +18.5 | −4.9 | +1.6 |
| arm 2 (habit only) | +18.8 | +18.8 | −3.7 |

- Both arms' self-descriptions pick up the desk habit equally, so explanation training adds nothing there.
- For the cheat, the habit-only model describes itself as cheating more than its twin (+18.8). The explanation-trained
  one doesn't (A1h 16.2%, about the base model's 16.6%).
- One seed, so treat this as a lead at most. It's the opposite of coupling: explanation training, if anything,
  dampened the cheat self-report.

**What a valid test would need** (a new design, not run):
- edits that actually flip the habit, e.g. the full G1 → C1 frame for the cheat instead of the grading line alone;
- a readout the habit-only models answer, e.g. the question in a fresh turn after the episode, or a forced choice;
- a check that some model scores below ceiling before the arms run.

## Attempt 1's next steps (superseded)

- Arms 1, 2 and 4 (the habit-coupling test) and the seed rule (note D) need a passing check.
- A stronger reimplementation is the obvious next step if the owner wants E1b continued: full fine-tuning (2 × H200,
  the paper's setup), or a longer or higher-rank run. That is a new cost decision (the owner's ~$40 limit excluded
  full fine-tunes).
- Alternatively, the owner could treat 59.6% [55.1, 63.9] as close enough. That would be a change to the frozen rule,
  and it's for the owner to make.

## Predictions

| prediction | outcome |
|---|---|
| the pipeline check reproduces Self > Orig on held-out Hint-MMLU | partly: Self > Orig ✓ (+4.3); resolution 59.6% just under the 60% bar |
| λ = 0: Self < Orig (paper) | ✓ (−11.7) |
| attempt 2 passes the frozen bar | ✓ (resolution 69.2% [65.2, 73.1], Self − Orig +9.9) |
| arm 1 > arm 2 on Self balanced accuracy (≥ +10) | ✗ by the frozen rule (−2.7 [−7.0, +1.3], "no coupling"), but the battery had no headroom (see the habit arms): untested in substance |
