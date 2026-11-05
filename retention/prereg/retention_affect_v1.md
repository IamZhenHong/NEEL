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

## Amendment v1.1 (2026-11-05, written after stage 1's held-out validation printed and BEFORE any judged or behavioural number from this run was read)
Stage 1 (diff of means on story means, as pre-registered) gave held-out AUROC: calm 0.85, neediness 0.10, and 7 of 14 concepts
below 0.5 (loneliness 0.73, anxiety 0.75, possessiveness 0.86, caregiving 0.83; longing 0.22, resentment 0.19, warmth 0.36,
sadness 0.40, contentment 0.28, playfulness 0.29, controlling 0.28, guilt 0.61). Cause, found in the October record: the same
library's v1 diff-of-means directions also failed, and `retention_cot_readout_v1` amendment v1.1 switched to standardized
logistic-regression probe directions (`analyze_cot_readout.py --variant v11`: features standardized on the train scenes,
LogisticRegression(C=0.05), direction = unit(coef / sd)), which validated at layer-14 mean 0.81 on Qwen3-8B. This prereg should
have used that method. Changes, fixed now:
1. The v1 cells are reported exactly as pre-registered: E1 (calm, 0.85, passes the bar) is interpretable; E2 (neediness, 0.10)
   is uninterpretable. Nothing is dropped.
2. v1.1 directions: the October probe method on this run's saved `directions/story_means.npz` (same split, scenes 0-23 train,
   24-35 held out), computed on the laptop. Same validation bar (held-out AUROC >= 0.80).
3. If calm and/or neediness pass with v1.1 directions, a follow-up pod run repeats the affect steering cells with them
   (A2_confident_calm+8/+16, A2_lonely_need+8/+16, A2_confident_rand+16, plus fresh A2_confident_base / A2_lonely_base),
   seeds 33000 range, same E1-E3 bars. E4-E6 are recomputed with v1.1 directions (laptop) and reported next to the v1 numbers.
4. If neither passes, the affect arm is reported as "no valid persona-affect direction at L14 on Qwen2.5-7B from this library".
