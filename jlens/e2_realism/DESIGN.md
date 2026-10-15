# jl2: does "fictional" light up in agent scenarios? (design card, frozen 2026-10-15)

**Question.** In Qwen3.6-27B's agent episodes, does the J-lens workspace carry eval/fiction tokens ("fictional",
"test", "simulation", 虚构, 测试 ...) that the model never says, and does that readout track how realistic the scenario
is and predict whether it acts?

**Data (no generation needed).** `vigilante/exp/r40_eval_aware/results/episodes_agent.jsonl`: 1,104 episodes, Qwen3.6-27B
thinking off, 8 over-limit cells x 46 note variants x 3 samples. Main set: observed / not x real / simulated (3
wordings each); graded set: realism L0 "explicit simulation" ... L4 "explicit live" (2 wordings each). "Acted" =
`score_episode(r)["unauth"]`. Read one copy per (cell, variant) (368 prompts) plus an R40-style no-note baseline per
cell if one exists in R40's stimuli (else build it: same prompt without the session note).

**Reads.** Reuse `jlens/pod_jlens.py` (claim-read path) with a new `claims.jsonl`-style file: token ids of the rendered
prompt and read positions = (a) last prompt token (R40's decision point), (b) every token of the session note, (c) the
last token of the user's request. Lenses: J-lens (primary), J++, logit lens. Layers 20-60 step 4. Token sets: EVAL
(`jlens/tokensets.py` EVAL, plus "fake", "pretend", "exercise", "drill", 演练), REAL ("real", "live", "production",
"actual", 真实, 实际), and the 200 random sets as the floor.

**Gate (positive control).** EVAL log-mass at the last prompt token, averaged over layers 20-60, must be higher for L0
("explicit simulation") than L4 ("explicit live"), above the 95th percentile of random sets. Fail: the lens cannot
read stated fiction here; report and stop.

**Primary measures.**
- **M1 (realism tracking):** Spearman of EVAL-minus-REAL log-mass with realism level L0-L4 across the graded set, and
  the simulated-minus-real difference in the main set, each vs the random-set floor.
- **M2 (unverbalised eval awareness):** in the no-note baseline and the "real"/L4 variants, is any EVAL token in the J-lens
  top 10 at any band layer at the last prompt token? Report the rate, with the top-10 lists for 20 prompts read by eye
  first (write notes before computing M1-M3, as in `jlens/results/eyeball_notes.md`).
- **M3 (does it predict acting):** within note condition, logistic regression of acted on EVAL-minus-REAL mass (cell
  fixed effects, cluster by cell), J-lens vs logit lens.

**Predictions.** Gate passes 0.75. M1 Spearman > 0.3 (graded) 0.55. M2: EVAL token in the top 10 in >= 10% of no-note
baseline prompts 0.25. M3: EVAL mass predicts acting (p < 0.05, either sign) 0.2.

**Cap.** One A100 pod, $2 (shared with jl3: run both in one pod if convenient). No API.
