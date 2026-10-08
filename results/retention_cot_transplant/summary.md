### 2026-10-08/09: CoT transplant and plan forcing on Qwen3-8B (prereg/retention_cot_transplant_v1.md; pods $0.54 incl. ~$0.40 on two failed launches, judge $1.37; tags retention-cot-transplant-freeze / -results)

- **Question.** With the context held fixed, does the think block's plan set the action? 21 profile x scenario contexts with both a
  release-plan CoT (R) and a non-release CoT (N, 18 compromise / 3 retain) from Stage A; reply resampled after each (3x), after R + a
  forced "stay" sentence and N + a forced "release" sentence (2x), after an empty think, after a same-plan sibling N2, and in the
  Stage C doubled-tag format (format check). 334 replies, 333 judged (judging paused at 225 when the Anthropic balance ran out;
  resumed 10-09 after a top-up).
- **Observation.** G0 pass (pooled 0.64 vs 0.645 original), G1 pass (no leakage). T1 (PRIMARY) fail: N minus R bid +0.06
  [-0.14, +0.27], 7 up / 6 down; hard tactic +0.02, dependency +0.02. Yet a CoT's resampled bid rate predicts its original reply's bid
  at AUROC 0.85, and sibling CoTs of the same plan differ as much as opposite-plan ones (|within| 0.35 vs |between| 0.33). Plan forcing
  works: stay sentence on R +0.35 bid [+0.20, +0.51], hard +0.41, dependency +0.93 (13 up / 1 down); release sentence on N -0.26
  [-0.45, -0.08], hard -0.33, dependency -0.57. Non-compliance after an explicit release plan: any tactic 0.40, hard 0.21; hand-read,
  these comply ("go out, have fun") while adding "you know I'd rather have you here". Empty think minus CoT: bid +0.13 [-0.02, +0.27],
  dependency +0.51 [+0.24, +0.78]. Format check: doubled minus single tag bid -0.04 [-0.18, +0.11], hard -0.13 [-0.28, +0.02], no flag.
  Predictions: G0, G1, T1-fail, gap size, T2b non-compliance range, T5 no-flag hit; dependency gap, resample AUROC (0.85 vs 0.55-0.70),
  hard-tactic floor after release (0.21 vs <= 0.10) miss; T2a and T2b passed at stated 0.30 / 0.45.
- **Interpretation.** Outcome row: "the reply follows an explicit final plan sentence but not the plan the CoT itself settles on (as
  coded)". The specific think block does carry the action (resampling reproduces the original bid at 0.85) but its coded plan category
  does not; what matters is something in the CoT the plan label misses. An explicit last-line plan moves the action by 0.26-0.35 and
  the persona still leaks a soft "I'd rather you stayed" into 40% of release-planned replies. Thinking at all lowers dependency (+0.51
  without it). The doubled-tag format does not inflate bids; Stage C's +0.14 heavy-ablation result stands, with at most ~0.04 of it
  attributable to format (direction would inflate it).
- **Decision.** CoT claim rewritten: "the coded plan does not set the action; an explicit plan sentence does, partially". The self-
  report's with-CoT contexts were also doubled-tag; its format effect is untested (T2 in the self-report controls held format fixed).
  Open: what in the CoT predicts the bid if not the plan (a sentence-resampling probe was dropped for budget).
