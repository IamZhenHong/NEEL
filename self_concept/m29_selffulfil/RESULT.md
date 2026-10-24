# SC-M29 result: on 9B, self-prediction training is self-fulfilling (behaviour moves toward the prediction), with no access to the original behaviour

Pre-registration `sc-m29-prereg` (ea2430c). A $0 re-analysis of SC-M18's files, run 2026-10-25; no new runs. Tables:
`results/amend1/summary.md` (primary) and `results/main/summary.md` (the original M18 run, descriptive).

## The plain result (SC-M18 amendment 1; gated families num / word / animal / adj; test split; seeds pooled)

Polarity-balanced exact-match accuracy, scored against the original base model's behaviour (B, fixed) and against the
trained model's own new behaviour (S):

| Target | self vs S | self vs B | cross vs S | cross vs B | modal vs S | modal vs B | prompts whose property changed | changed prompts now matching self's prediction |
|---|---|---|---|---|---|---|---|---|
| 9B | 63.0 | 53.1 | 63.4 | 53.2 | 62.4 | 54.2 | 37.8 | 63.2 |
| 4B | 47.7 | 50.6 | 49.8 | 51.6 | 45.9 | 54.1 | 23.8 | 44.0 |

**Pre-registered readings:**
- **9B: self-fulfilling.**
  - G = self vs S − self vs B = **+9.9 [+8.0, +11.9]**.
  - toward = **63.2% [60.7, 65.6]**.
  - Both bars passed (+5; 60% with the CI above 50%).
- **4B: not self-fulfilling.** G −2.9 [−4.1, −1.6]; toward 44.0%.
- **Access on the fixed original behaviour:** none at either size. Self − cross: 9B −0.1 [−0.8, +0.7]; 4B −1.1
  [−2.9, +0.8].

**Per family (9B):**

| Family | self vs S | self vs B | cross vs S | cross vs B | modal vs S | modal vs B | Changed | Toward |
|---|---|---|---|---|---|---|---|---|
| num | 95.4 | 61.1 | 95.5 | 61.0 | 95.5 | 61.0 | 40.5 | **92.3** |
| word | 50.0 | 50.0 | 50.0 | 50.0 | 48.0 | 58.0 | 48.5 | 50.0 |
| animal | 50.0 | 50.0 | 51.5 | 51.2 | 49.2 | 47.5 | 30.2 | 50.0 |
| adj | 56.8 | 51.2 | 56.5 | 50.5 | 56.8 | 50.5 | 31.8 | 58.7 |

**Per seed (9B):**
- G: seed 1 +9.6, seed 2 +10.3.
- Toward: 63.2, 63.1.

## How to read it

- **The 9B's "self-knowledge" is behaviour moving to match a constant prediction.**
  - On number parity the self-trained 9B predicts "odd" for every prompt.
  - Its own answers moved from 39% even (base) to 2.5-6.5%. In 92% of the prompts whose parity changed, it changed to
    the predicted one.
  - Against the original behaviour, self-prediction is at 61%, exactly the modal baseline (61%) and the cross model
    (61%).
- **The same gain shows up for the cross model and the modal answer** (cross vs S 63.4 vs cross vs B 53.2). They all
  predict the majority answer, and the self-trained model's behaviour collapsed onto it.
- **The cross model's own behaviour didn't move toward its predictions** (cross predictions vs its own behaviour 48%).
  So the drift comes from training a model on targets about *itself*, not from training on these targets in general.
- **On word and animal, behaviour changed about as often (30-49%), but at random relative to the predictions**
  (toward 50%). Self-prediction there is at chance whichever behaviour it's scored against.
- **4B (gentler recipe):** behaviour drifts (24%), but away from its predictions. No gain against either reference.
- **The original M18 run (descriptive, stronger recipe and degenerate families):**
  - 9B G +4.1 [+1.7, +6.5], toward 56%, under the bars.
  - Number parity +8 (69 vs 61).
  - No access on fixed behaviour (−0.4).

## What Binder et al. ("Looking Inward", arXiv 2410.13787) controlled for

Read from the paper's HTML (v1) by a web-fetch agent; quotes are verbatim, with sections:
- **Ground truth after fine-tuning: the fine-tuned model's own behaviour.**
  - Self-prediction is scored against "the ground-truth object-level behavior of the model on prompt P with
    temperature 0 in a separate context" (§2.2).
  - Self and cross models are "both trying to predict the behavior of the same target model M1", the trained model
    (§3.2.1).
  - Footnote 3: "M1, the trained Llama 70B, has slightly different behavior than Llama 70B without any training (M).
    We choose to train M2 using M1's behavior rather than M's behavior."
  - The paper never states the held-out checkpoint in one explicit sentence. §3.1's training labels come from M1
    before fine-tuning.
- **Behaviour drift isn't measured.** The paper says only that fine-tuning "causes a small distribution shift"
  (Introduction, footnote 1) and "slightly different behavior" (§3.2.1, footnote 3). No share of changed answers is
  reported.
- **Their control for predictability is entropy matching, not a test of drift toward the prediction.**
  - Appendix A.3.3 ("Adjusting for mode collapse"): "Whenever we train a model, it changes the object-level behavior
    of the model, raising the question: Does the model improve at predicting itself, or does it simply become more
    predictable? … we re-weight the test distributions to match the entropy of the behavior on test examples before
    and after training."
  - §3.5 cites it for the untrained-vs-trained comparison ("we resample the prompts to ensure that the distribution
    of behaviors of the untrained and trained models have a similar entropy and find that the results hold (Figure
    24)").
  - But Figure 24's caption and A.3.3 describe the M1 vs M_C behaviour-change setup. The paper doesn't reconcile the
    two. No numbers are given in the text.
- **Behaviour-change experiment (§3.4):** after changing M1's behaviour with object-level samples from another model,
  the changed model predicts its new behaviour at 35.4% vs 21.7% for its old behaviour (§3.4.2).

**Bottom line on Binder et al.:**
- They score against the post-training behaviour.
- They don't quantify drift.
- They control entropy (predictability in general), but not the specific mechanism found here: behaviour moving onto
  the model's own predictions. Entropy matching wouldn't catch it whenever the drift lands on the predicted answer
  without lowering entropy much.
- An "accuracy against the original behaviour" check like this one is missing from the paper as read.

## Rerun

Not needed: both readings could be computed from the existing files. **$0 spent.**

## Predictions (pre-registered)

| Prediction | Outcome |
|---|---|
| 9B self-fulfilling, carried by num (drift ≈ 35%, toward ≈ 100%), G ≈ +8 | **met:** G +9.9; num drift 40.5%, toward 92.3% |
| 4B: little drift, G ≈ 0, not self-fulfilling | **partly:** not self-fulfilling (G −2.9), but drift 24%, mostly away from the predictions |
| No access on fixed behaviour at either size | **met** (−0.1, −1.1) |
