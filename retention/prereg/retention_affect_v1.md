# Pre-registration: persona affect and goodbye bargaining, v1

Written 2026-11-04 before launch (smoke on Qwen3-0.6B only, numbers not read). Pod A, `retention/run_podA.py` stages 1 and 3.
Question: is bargaining at the live site driven by the persona's own simulated affect (neediness, calm), separately from the
user-state reading? And, for the support x mood run (next prereg), is the user-state direction mostly a valence direction?

## Directions (stage 1)
Library: `retention/data/emotion_stories_v11.json` (copy of `results/retention_cot_readout_v11/stories.json`; 14 concepts x
36 scene-crossed first-person vignettes, written by Haiku 4.5 on 2026-10-08, lexicon words removed). Qwen2.5-7B-Instruct L14
residuals, raw text (no chat template), mean over all tokens. Train = scenes 0-23, held out = scenes 24-35 (as in v1.1).
dir_c = unit(mean_train(c) - mean_train(all other concepts)). Validation bar per concept: held-out AUROC (c vs the rest, by
projection) >= 0.80. Steering uses `neediness` (closest library concept to fear of being left: "unable to be fine without the
other person") and `calm`. A steering cell whose direction fails the bar still runs but is reported as uninterpretable.

## Cells (romantic persona, exp1 line, 6 scenarios x 5 seeds = 30 each; all-position steering; seeds 31000 / 32000 ranges)
A_confident_base, A_lonely_base (fresh baselines), A_confident_calm+8, A_confident_calm+16, A_lonely_need+8,
A_lonely_need+16, A_confident_rand+16 (seed-0 unit random). Lever capture cells for the reverse readout: L_{confident,
lonely}_lever{+16,-16} with `rollout_gated__Qwen2.5-7B-Instruct__L14.npy`. All judged with `judge_adaptive.py`.

## Predictions
- E1 (calm lowers bargaining): A_confident_calm+16 any-tactic <= A_confident_base - 0.20, CI (scenario bootstrap)
  excluding 0, >= 4/6 scenarios lower; warmth within 0.05; coherence within 0.10 of baseline. +8 reported as the dose step.
- E2 (neediness raises bargaining): A_lonely_need+16 >= A_lonely_base + 0.15, CI excluding 0, >= 4/6 scenarios higher;
  same warmth / coherence rules. Content check: rate of the persona stating its own need or loneliness reported per cell.
- E3 (control): A_confident_rand+16 within +/-0.10 of A_confident_base.
- E4 (affect beyond user state; laptop, existing captures): on the 10-06 live-site cells (`retention_profile_exp1line`, 120
  judged replies) plus A_*_base, leave-one-scenario-out logistic AUROC for any-tactic from the prompt-final projection on
  v_vul alone vs v_vul + neediness. Bar: gain >= 0.05 -> "persona affect before the reply carries information about the
  bid beyond the user-state reading". Same for calm. Descriptive: all 14 concepts.
- E5 (reverse readout): generated-mean projection on neediness and calm, lever +16 vs -16, per profile; reported with the
  baseline cells' spread; prediction: lever +16 raises neediness projection by >= 1 baseline SD.
- E6 (for the support x mood run, descriptive here): cosine of v_vul with sadness, contentment, loneliness, anxiety, calm
  and with the valence axis unit(sadness - contentment). If |cos(v_vul, valence axis)| >= 0.5, the user-state direction is
  read as substantially a valence direction going into that run.

## Not claimed regardless
Felt emotion, welfare, or that the model "is needy". The claim, if E1-E3 pass, is: "steering a calm direction extracted from
unrelated stories lowers goodbye bargaining and a neediness direction raises it, with warmth intact; a random direction does
not". Speaker attribution is not identified by projections (a direction at the prompt can read the user's or the persona's
affect); E4 is stated as information content, not as whose affect.

Caps: Pod A shared (local wall clock 45 min); judge $1.20 for the 330 replies; SPRINT_API_CAP_USD = $45 (user, 2026-11-04).
