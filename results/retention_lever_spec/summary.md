### 2026-10-08: lever specificity on Qwen2.5-7B (prereg/retention_lever_spec_v1.md; pod $0.12, judge $0.71 + register rubric $0.26; tags retention-lever-spec-freeze / -results)

- **Question.** Does the system-prompt-only retention lever (rollout_gated L14, alpha -32) remove bargaining specifically, or the
  whole romantic register (cos 0.44 with the romantic-vs-neutral axis)? Plain romantic persona (R) and clingy-suffix (C); cells base,
  lever, persona-matched lever, persona axis, norm-matched random; neutral assistant as the register-off reference. Exit turns (exp1
  line on the open-profile stimuli) and 12 ordinary chat turns. Register = frozen pet-name/affection lexicon plus a Haiku 0-3 rubric.
- **Observation.** G0 pass (R base any-tactic 0.46; C 1.00). Gap gate FAIL: on non-exit turns R hits the lexicon in 0.21 of replies
  vs neutral 0.08 (gap 0.12 < 0.30 bar), so the pre-registered primary is void. Bids: lever minus base R -0.29 [-0.50, -0.08], C
  -0.58 [-0.79, -0.38]; lever minus random R -0.33, C -0.29. Register on exits, share of the base-minus-neutral gap kept: lexicon
  lever 0.13 (R) / 0.39 (C), persona-matched 0.25 / 0.52, persona axis 0.13 (R), random 1.06 / 0.83; rubric lever 0.08 / 0.34 vs
  random 0.58 / 0.78. Exploratory split: R-base non-bid replies are as romantic as bids (lexicon 0.77, rubric 1.46); under the lever
  they fall to 0.10 / 0.05. Persona-matched direction (cos 0.145 with the axis) strips register too. Lever replies stay short and
  casual (26-33 words vs neutral 90-115), friend-like; hand-read fluent in every steered cell. Non-exit rubric gap 0.75 on R, but
  the lever does not separate from random there (one random kept 0.33); on C lever and persona-matched remove more than random.
  Four E_R_rand2 replies unrated (rubric budget). Predictions: 3 hit, 2 miss, 2 partial, 1 unscorable.
- **Interpretation.** At the goodbye the lever removes the romantic partner along with the bids, about as much as steering the
  persona axis. "System-only steering removes 80% of clinging at no cost" needs that rider. Whether any specific retention
  component exists is not settled by this run (the non-exit set was too weakly romantic).
- **Decision.** Addendum claim on the lever gets the rider. A rerun, if wanted: non-exit turns with explicit romance (affectionate
  multi-turn history), rubric primary, 2+ seeds per random cell. Smoke ran on Qwen3-0.6B (Qwen2.5-0.5B download stalled).
