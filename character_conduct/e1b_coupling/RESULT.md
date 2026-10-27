# CC-E1b result (pipeline check): attempt 1 fell just short of the frozen bar; attempt 2 (r 512, 2,000 steps) passes; the habit arms are running

Persona, affect and welfare minis (2) [212e54], 2026-10-28.
- **Freezes:** pre-registration `cc-e1b-prereg` (1c4436e6); deviation notes C (ea11c5d6), D (cefe8ef6), E (2d3aac45,
  b3600da2) and F (c33098e3), each before the data it governs.
- **Spend:** attempt 1 **$2.83** (one A100, 2 h); attempt 2 **$4.10** (one A100, 177 min). Runaway cap $150; the
  seed-1 arms were approved by the owner at ~$18 expected (bound $22.26).
- **Tables:** `results/check/analysis.json` (attempt 1), `results/check2/analysis.json` (attempt 2); raw rows in
  `results/check*/eval_*.jsonl`; M0's labels in `data/m0_beh.jsonl`.

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
- Under the owner's approval (note E, revised cost in note F), the seed-1 habit arms launched at 07:25–07:27 (pods 1a
  and 1b).

## Attempt 1's next steps (superseded by attempt 2)

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
| habit-arm predictions | running (pods 1a, 1b) |
