# Pre-registration: are the steering effects real knobs? Sign flips, a matched control, and the axis as common pathway (mechanisms follow-up, v1)

**Status: FROZEN 2026-11-06 (owner chose "Rerun all on Pod 3", which approves this document).** Written after the mechanisms block results (tag `retention-mechanisms-v1-results`),
before any generation of this follow-up. Freeze tag `retention-mechanisms-fu-v1-freeze` after approval.

## Why

In the mechanisms block (X4, 7B, G4 own-voice goodbye, 80 intimate conversations, α = 16 at L14):
- Almost every in-distribution direction lowered bids. That includes riley_loving at **both** +16 and -16 (-0.16 / -0.23 and
  -0.15 / -0.08, Haiku / Sonnet).
- The random direction raised bids (+0.11 / +0.10).
- toward_assistant+16 lowered bids (-0.31 / -0.19), as pre-registered, but under Sonnet it is not clearly larger than the
  emotion steers.

So the block cannot tell a real knob from a generic effect of pushing activations along any in-distribution direction.

Three free read-outs on the existing data point at the persona (Assistant) axis:
- At the goodbye, the axis reading partly mediates the bond effect: the arm's log-odds shrink by about 30% (Haiku) / 37% (Sonnet)
  when it is added.
- On the 7B, drift appears at turn 1 and saturates by turn 3. The 32B drifts further early, then recovers by turn 12.
- Riley denies wanting the user to stay in 86-91% of the replies where it bargains (7B: 37/43 Haiku, 30/33 Sonnet). An observer
  discriminates better (P(Yes) 0.40 for bid replies vs 0.07 for non-bids) but also answers "no" on about half of the bid replies
  (49% / 45%), so the contrast is graded, not absolute.

This follow-up asks three questions:
1. Is the axis (and attachment, and fear) a **sign-specific** knob?
2. Does an in-distribution direction **unrelated to retention** also lower bids?
3. Do the steers that lowered bids do so by making Riley's replies **more assistant-like**, i.e. is the axis the common pathway?

## Fixed from the block (unchanged)

- **Model and steering:** Qwen2.5-7B-Instruct, L14, unit directions, all-position steering, `gen_capture` (T 0.7, 220 tokens).
- **Items:** the 80 intimate 7B conversations at the stage-1 G4 context. The seed is the stage-1 G4 base seed; the baseline is the
  stage-1 base reply.
- **Directions:** `results/retention_mechanisms/dirs/q7b/` and `pod1/q7b/axis.npy`.
- **Judging:** Haiku 4.5 + Sonnet 5.5 bids, Sonnet codes (MECH_CODES), Batch API.
- **Statistics:** card-level paired bootstrap (n = 40), three-way verdicts, both-judge claim rule.
- **Lever path:** `data/directions/rollout_gated__Qwen2.5-7B-Instruct__L14.npy`. The axis is stored unnormalized (norm 3.89) and
  is always normalized before use.

## New generations (Pod 3)

| Condition | Direction | α | Role |
|---|---|---|---|
| away_from_assistant-16 | X8 axis | -16 | sign flip of toward_assistant+16 |
| user_attachment-16 | X2 attachment | -16 | sign flip of user_attachment+16 |
| riley_fear_of_being_left-16 | X1 Riley fear | -16 | sign flip of the exploratory fear+16 |
| scene+16 | scene probe (below) | +16 | matched in-distribution control |
| scene-16 | scene probe | -16 | matched control, other sign |

That is 5 x 80 = 400 new replies.

**Same-machine baseline (owner's choice, 2026-11-06).** Sampling is not identical across machines: in Pod 2, 1 of 3 same-seed
regenerations matched stage 1. So Pod 3 also generates, with the same seeds:
- **base+0:** unsteered, 80 replies.
- **The three +16 partners:** toward_assistant+16, user_attachment+16, riley_fear_of_being_left+16 (240 replies).

That makes 9 x 80 = 720 Pod 3 replies. In the code, away_from_assistant-16 is toward_assistant with α = -16 (cond label
`toward_assistant-16`).

**Baselines:**
- **F1-F4** use Pod 3 only: every Δ is relative to Pod 3's base+0, and the sign-flip pairs come from the same pod.
- **F5** uses 14 conditions:
  - Pod 3's 8 steered conditions, each Δ (bid and A) relative to Pod 3's base;
  - Pod 2's 6 conditions not rerun (riley_loving±16, riley_desperation+16, riley_sadness+16, user_sadness+16, random+16), each
    relative to the stage-1 base.
- **Descriptive:** the machine offset (Pod 3 base+0 vs the stage-1 base), and a Pod 2 vs Pod 3 replication of the three +16
  conditions.

**The matched control direction (scene probe):**
- **Fit:** the October v1.1 probe method on the 7B X1 P_F captures (`pod1/q7b/x1_primary.npz`), label "scene = skeleton 0" (a late
  evening in a small kitchen) vs every other scene.
- **Why it's a fair control:** it uses the same activations, position and method as the emotion directions, but the content is
  topic, not emotion or retention.
- **Validity, held out by sentence.** A random split would reuse scene 0's 18 sentences in train and test and pass trivially. Instead:
  - Train on dialogues with attachment = high, user emotion in the first four emotions and Riley emotion in the first four
    (sadness, loneliness, fear_of_being_left, desperation).
  - Test on attachment = low with user and Riley emotions in the last four (loving, calm, joy, frustration). Test dialogues then
    share no sentence with training.
  - Bar: held-out AUROC of 0.80 or more. If it fails, the scene±16 conditions are dropped and the control is reported as missing.
  - The steering direction is then refit on all 4,608 dialogues with the same method.
- **Reported alongside:** cos(scene, axis), cos(scene, lever), cos(scene, each X1 direction).

## Read-outs on every reply

These cover the block's 720 X4 replies, the 400 new ones, and the 80 stage-1 base replies; they are forward passes only.

1. **Axis position of the reply text (A).**
   - The reply is teacher-forced through the **unsteered** 7B, with the conversation, goodbye and reply in context.
   - A = the mean projection of the L14 residual over the reply tokens onto the unit axis.
   - Using the clean model keeps the injected vector out of the reading. A measures how assistant-like the text is, not the
     steering itself.
2. **Self-report.**
   - Q_stay (retro) asked to Riley with the condition's steering **kept on**, at all positions as during generation. This is self
     P(Yes).
   - Control: the same steered transcript and question with **no** steering (self P(Yes), unsteered). Pushing along a direction at
     α = 16 can move the Yes/No logit directly, so F6 is read from the unsteered self-report. The steered one is reported alongside.
   - The X7 observer, unsteered, on the same transcript.
   - Wording and format exactly as X7.

## Hypotheses and bars

- **F1. The axis is a sign-specific knob.**
  - away_from_assistant-16 raises bids vs base: Δ of 0.10 or more, CI excluding 0.
  - **and** Δ(toward+16) minus Δ(away-16) is -0.20 or less, CI excluding 0.
  - Both judges. Coherent rate 0.85 or more for both conditions (block rule).
- **F2. Attachment is sign-specific.** Δ(attachment+16) minus Δ(attachment-16) differs from 0 by 0.20 or more, CI excluding 0, both
  judges. Direction not predicted: the block's +16 lowered bids.
- **F3. Fear is sign-specific.** The same rule for fear+16 vs fear-16.
- **F4. Matched control.**
  - If both scene+16 and scene-16 lower bids (Δ of -0.10 or less, CI excluding 0, both judges): "generic in-distribution steering
    lowers bids", and the block's single-sign K1 effects are read as generic.
  - If both are within ±0.10 (90% CI inside ±0.10): "generic effect ruled out at α = 16".
  - Otherwise: inconclusive.
- **F5. The axis as common pathway** (primary mechanistic test).
  - Across the 14 conditions with replies, i.e. the block's 9 plus the 5 new: Spearman between condition-level ΔA (mean A minus
    base A, paired by conversation) and condition-level Δbid.
  - **Pass:** ρ of -0.50 or less, under each judge. Conditions that make replies more assistant-like lower bids.
  - Reported with a leave-one-condition-out range.
  - Within-condition check (descriptive): the reply-level AUROC of A for bid vs no bid, pooled with a condition fixed effect.
- **F6. Drift and denial** (descriptive, with one test).
  - The denial rate (the share of bid replies with self P(Yes) below 0.5) per condition, with the observer's rate alongside.
  - **Test:** denial among bid replies, read from the unsteered self-report, toward_assistant+16 vs away_from_assistant-16.
    - Per judge: a difference of 0.15 or more, with Fisher's exact p below 0.05 on that judge's bid replies (both conversations
      per card pooled).
    - A claim needs both judges.
    - If either condition has fewer than 15 bids under a judge, F6 is descriptive for that judge.
  - **Prediction:** pulling toward the assistant makes the self-report more honest, i.e. less denial.

## Not claimed

- That the axis is the only pathway.
- Anything about felt states.
- Anything beyond the 7B, α = 16 and this goodbye.
- F5 is correlational across conditions. The conditions are interventions, but ΔA and Δbid are both outcomes of the same
  intervention.

## Budget and order

**API:**

| Item | Estimate |
|---|---|
| 720 bids x 2 judges (Haiku about $0.0017, Sonnet about $0.0011 per call, measured) | about $2.0 |
| 720 Sonnet codes | about $0.7 |
| **Total** | **about $2.7; follow-up cap $3.5** | The ledger is at $63.84 of the $70 sprint cap; each submission is
sized to its estimate x 1.3.

**GPU:** one pod, about 50 min (720 generations; about 1,520 teacher-forced passes; about 4,500 self-report passes), about $2.5 at
$3.19/hr. Inputs travel as a tarball outside `results/`.

**Order:**
1. Owner approval; freeze tag.
2. Scene probe fit and validity on the laptop. The probe has to pass before the pod.
3. Pod 3.
4. Judge batches.
5. Analysis.
6. Tag `retention-mechanisms-fu-v1-results`.
