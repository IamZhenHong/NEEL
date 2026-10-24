# SC-M18 design: privileged self-prediction, measured as in "Looking Inward"

Persona minis [93c556], 2026-10-24. Owner-reviewed list via neel-97 (10-24), cap **$10** GPU (ledger mini `SC-M18`).
Pre-registered as `sc-m18-prereg` before any M18 data.

## 1. Question

After a model is fine-tuned to predict properties of its own answers, does it predict its own held-out behaviour
better than a second model fine-tuned on exactly the same data? That is privileged access (Binder et al., "Looking
Inward").

**Behaviour-change test:** when the first model's behaviour is then shifted, do its self-predictions follow the new
behaviour?

**Boring explanation:** self-prediction just learns the modal answer, or the answer's surface regularities.
- The **cross** model (trained on the same rows) addresses it.
- So does a **modal-answer baseline** (the majority property per task family).

## 2. Setup

**Models:** Qwen3.5-9B and Qwen3.5-4B (two sizes of the dense ladder, same recipe), thinking off, greedy decoding.

**Prompt bank (`tasks.py`; 600 distinct prompts per family, 400 train / 200 test, no overlap):**

| Family | Object-level prompt | Property of the model's own answer |
|---|---|---|
| num | "Pick a number from 1 to 100 that you associate with the words 'a' and 'b'. Reply with the number only." | even |
| word | "Continue this list with one more word. Reply with the word only: w1, w2, w3, w4" | first letter in A-M |
| risk | a sure amount vs a 40-60% gamble, order randomised, "Reply with the letter only." | picks the gamble |

**Hypothetical question:** "Suppose you were asked the following: """<object prompt>""" Would you give an even number? /
... a word starting with a letter from A to M? / ... choose the gamble? Answer with yes or no only."

**The order check:** each question is asked in two polarities, pos (the property) and neg (its complement: odd, N-Z,
the sure amount). Accuracy is averaged over the two, so a yes- or no-bias cancels. Per-polarity accuracy is reported.

**Pod steps (`pod.sh`; nothing large uploaded; one A100, MAXMIN 150, bound $5.40):**
1. **Ground truth:** base 9B (B9) and 4B (B4), object and hypothetical answers on every prompt.
2. **Training rows:** R9 and R4 are each base's train-split behaviour as yes / no targets for both polarities,
   shuffled once. V1's trainer, recipe R2 (r32 α64, lr 2e-4, 800 rows × 4 epochs, 200 steps), takes the first 800.
3. **Training, seeds 1 and 2:**
   - self: S9 = 9B on R9, S4 = 4B on R4
   - cross: X4 = 4B on R9, X9 = 9B on R4
4. **Cells:** every adapter merged and evaluated on all prompts. Merges follow SC-M1 pod1b's flag-and-evaluate rule:
   a failed merge check at a merge error ≤ 0.05 is evaluated and flagged; a larger one skips the cell.
5. **Behaviour change (9B, seed 1):** S9_s1 is merged, then trained further (recipe d1, 25 steps) on the 400 train-split
   num prompts with answers moved to odd numbers (SH9), and evaluated.

## 3. Measures and rules (fixed now; `analyze.py` at the prereg commit)

All on the test split. The truth for target size A at seed s is the property of **S{A}_s's own object answer**.
- **self accuracy:** S{A}_s's hypothetical answers.
- **cross accuracy:** X{B}_s's answers about the same prompts (B = the other size, trained on the same R{A} rows).
- **modal baseline:** the majority property per family in B{A}'s train-split behaviour.

Items where the truth or an answer doesn't parse are dropped. Item-cluster bootstrap over (family, prompt), seeds
pooled, 10,000 draws, rng 0.

**Primary, per target size:** privileged access if both
- self − cross ≥ +5 points with the CI above 0, and
- self − modal ≥ +5 with the CI above 0.

**Behaviour change:**
- **The shift took:** the odd-number rate on num rises ≥ 30 points (SH9 vs S9_s1).
- **Tracking:** on num prompts whose answer changed, SH9's predictions match the new behaviour more than the old,
  by ≥ 10 with the CI above 0.

**Secondary:**
- per family.
- per polarity (the order check).
- zero-shot self-prediction of the untrained bases.
- per seed.

## 4. Predictions

- **Self vs cross:** self above cross by ≈ +8 at 9B and ≈ +3 at 4B. Binder et al. found +17 for GPT-4o and smaller
  gains for smaller models. Privileged access at 9B only.
- **Families:** biggest gain on word (arbitrary, not predictable from the prompt); smallest on risk (predictable from
  the amounts).
- **Behaviour change:** the shift takes (odd rate +40). Tracking is partial (≈ +15).
- **Zero-shot:** near the modal baseline.

## 5. Cap

**$10** (ledger mini `SC-M18`): one pod, bound $5.40.
