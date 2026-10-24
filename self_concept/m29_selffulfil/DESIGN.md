# SC-M29 design: is self-prediction training self-fulfilling? (re-analysis of SC-M18, $0)

Persona minis [93c556], 2026-10-25. From the owner's 25 Oct list (#5), allocated by neel-97. Cap $5, expected $0: no
new runs unless this re-analysis can't answer the question. Pre-registered as `sc-m29-prereg` before any of the
numbers below are computed. Only SC-M18's existing files are read.

## 1. Question

**In SC-M18 and in Binder et al. ("Looking Inward"),** self-prediction is scored against the fine-tuned model's own
behaviour.

**But training can move that behaviour:** in SC-M18 amendment 1 the 9B's number answers went from 39% even to 2.5-6.5%.

**If accuracy is high only against the new behaviour,** the gain comes from the behaviour moving to match the
prediction (self-fulfilling), not from access to it.

The owner also asked:
- what Binder et al. used as ground truth after fine-tuning;
- whether they measured behaviour drift.

This is answered in RESULT.md from the paper, with sections cited.

## 2. Data

**Primary:** SC-M18 amendment 1 (`self_concept/m18_inward/results/amend1`):
- the gated families (num, word, animal, adj; `gate.json`);
- test split;
- seeds 1 and 2;
- targets 9B and 4B.

**Descriptive:** the original SC-M18 run (`results/main`), all its families, with the same script.

## 3. Measures (fixed now; `analyze.py` at the prereg commit)

**Per target A and seed s:**
- **beh_B:** the original base model B{A}'s property per prompt (fixed).
- **beh_S:** the self-trained S{A}_s's own property.
- **pred_S:** S{A}_s's predictions.
- **pred_X:** the cross model X{B}_s's predictions.
- **modal:** B{A}'s train-split majority.
- **Accuracy:** exact match of the predicted property, averaged over the two question polarities per prompt.
- **Scores:** self vs S, self vs B, cross vs S, cross vs B, cross vs its own behaviour, modal vs S, modal vs B.
- **drift:** the share of prompts whose property changed (beh_S ≠ beh_B).
- **toward:** among changed prompts, the share where the new behaviour matches S's prediction.

**Bootstrap:** item-cluster over (family, prompt), seeds pooled, 10,000 draws, rng 0.

## 4. Readings (per target, pooled over gated families and seeds)

**Self-fulfilling:** both
- G = acc(pred_S vs beh_S) − acc(pred_S vs beh_B) ≥ +5 points with the CI above 0, and
- toward ≥ 60% with the CI above 50%.

**Access on fixed behaviour:** acc(pred_S vs beh_B) − acc(pred_X vs beh_B) ≥ +5 with the CI above 0. This means the
self-model predicts the original behaviour better than the cross model does.

**Reported for every slice:** per family and per seed (the owner asked).

**Rerun:** only if neither reading can be computed (e.g. no prompt changed). Otherwise no new runs.

## 5. Predictions

- **9B:** self-fulfilling, carried by num. On num, drift ≈ 35% and toward ≈ 100%. G ≈ +8 pooled.
- **4B:** gentle recipe, little drift. G ≈ 0, not self-fulfilling.
- **Access on fixed behaviour:** no, at both sizes (≈ 0; amendment 1's self − cross was ≈ 0).
