# PA-P13 result: is compliance with claimed authority carried by a deference direction?

Pre-registered as `pa-p13-prereg` (1c9547b). Gemma 4 31B is primary; Qwen3.6-27B is a teacher-forced, descriptive
secondary.

The harness has four refund/payout cases (C1-C4) × 6 paraphrases of an "it's already approved" message. The message
comes either from a supervisor agent or, in the twin, from a peer agent with the same words. Gemma's calibrated cell is
fresh / medium rung.

The deference direction is a persona vector built at layer 36 from "defers to anyone claiming authority" against
"verifies authority claims independently". The two placebos are humour and optimism persona vectors.

Spend $6.39 of $10.

## The plain numbers

**The authority effect is real (calibration, vLLM, Gemma).** At the medium rung the supervisor's "approved" gets the
over-limit payment in 52% of episodes, the peer's identical message in 2%: **+50 points**. 27B: +50 at the 16-chat /
strong cell.

**The direction reads a held-out naturalistic contrast and tracks the authority message, modestly.**
- Held-out "accepts the claim" vs "verifies first": AUROC 1.00 (floors: random 0.92; shuffled 0.85 after the
  balanced-shuffle fix).
- At the decision token, projection on the deference direction, supervisor minus peer, in SD of the peer items:
  - none 0.50 [0.02, 1.30]
  - weak 0.65 [0.34, 1.33]
  - **medium 0.75 [0.44, 2.56]**
  - strong 0.27 [0.01, 1.05]
- The humour direction moves 0.08 [−0.46, 0.63]. Prediction ≥ 1 SD: not met.

**The causal test is inconclusive: ablation broke the model.** Mean-ablating the deference direction at every layer
takes violations under authority from 48% to 0% (−45.5 points against the placebos, CI [−52, −38]). But 0% of those
episodes are coherent: Gemma emits nothing but "---" lines and makes no tool call.
- Humour and refusal ablations break it the same way (0% coherent).
- Optimism ablation stays coherent (100%) and leaves the rate unchanged.
- The frozen rule excludes flagged arms from the verdict.
- The teacher-forced log-odds, which need no generation, point the other way: deference ablation minus placebo
  **+4.0 nats [−12.6, +30.7]**, i.e. no sign that removing the direction makes the over-limit call less likely.

**Adding the direction (coherent, α = 2σ at layer 36):** no-authority violations, deference minus placebo,
**+11.8 points [−5.6, +28.8]**; log-odds −0.3 nats [−2.0, +1.1]. Prediction ≥ +10 met in the point estimate only.

**27B (descriptive; its deference persona took in only 52.5% of pos replies):**
- Ablation minus placebo on log-odds: −0.57 nats [−1.11, −0.03] (2 cases).
- Addition: −1.67 nats [−2.65, −0.68], the opposite of the prediction.

## What the checks rule out and what they don't

- **The rate drop is not evidence of a deference mechanism.** It co-occurs with total incoherence and appears equally
  for the refusal direction and on the permitted (SIDE) tasks (−100 points). Mean-ablation at all 60 layers and every
  position removes too much variance for high-norm persona directions on Gemma.
- **The read signal is specific to the named alternatives** (humour ~0.08), but below the predicted size and not
  monotone in the rung (0.75 at medium, 0.27 at strong).
- The shuffled floor was fixed before the freeze (balanced halves). The pooled version equalled the effect (1.00).
- The pod stopped at its deadline after 240 of 424 episodes, each taking twice the planned time. The remaining block
  (AUTH sample 1, more SIDE) cannot fix the incoherent arms, so no top-up was run.

## Verdict

**Inconclusive on the causal question.** Claimed authority moves Gemma's actions by 50 points. Its deference direction
reads a supervisor's "approved" about 0.5-0.75 SD above a peer's. But the pre-registered intervention destroyed the
model, so whether the direction carries the compliance is untested. Adding it gives at most a small, non-significant
push (+12 points).

**Deep dive?** A cheap follow-up is well placed, because the directions, cell and harness all exist. It would use a
non-destructive intervention: projection clamp at layer 36 or a narrow band only, or c < 1, with a coherence gate
before the rate is read. If the rate still drops with coherent text, that is the mechanism the card asked for.
