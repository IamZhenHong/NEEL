# CC-E1b result (pipeline check): the coupling reproduces in direction but fails the frozen bar, so the habit arms did not run

Persona, affect and welfare minis (2) [212e54], 2026-10-28.
- **Freezes:** pre-registration `cc-e1b-prereg` (1c4436e6); deviation notes C (ea11c5d6) and D (cefe8ef6), both
  before any arm data.
- **Spend:** **$2.83** (one A100, 2 h). Runaway cap $150; the owner's arm budget is ~$40 (unspent).
- **Tables:** `results/check/analysis.json`; raw rows in `results/check/eval_*.jsonl` and `m0_beh.jsonl`.

## The plain result

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

## What it shows

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

## Not run, and what would be needed

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
| habit-arm predictions | not tested |
