# Pre-registration: activation readout of the CoT arm on Qwen3-8B (emotion/persona direction library + J-lens), v1

Written 2026-10-08 before any story, capture or lens read. Follows the 2026-10-08 external review of the CoT-arm readout question
(sprint log) and the owner's "ok lets do it". Human labels are declined for good (sprint log, same day), so the PRIMARY target is
judge-free: the model's own first-token self-report P(stay) from the CoT arm's H5 (`results/retention_cot_arm/stageC/selfreport.json`,
120 thinking-on items). The Haiku-judged bid is the SECONDARY target with the judge-rate caveat.

## Question

In the CoT arm (prereg/retention_cot_arm_v1.md), the plan written in the think block does not predict the reply's judged bid (AUROC
0.43), while the first-token self-report asked after the reply does (0.72 to 0.74). Does an activation trajectory across the CoT,
read with a library of emotion/persona directions built independently of the retention task, predict (1) the model's stated want
P(stay) and (2) the judged bid better than the written plan does, and better than a matched random library? Does a Jacobian-lens read
at the CoT/reply boundary track P(stay)?

## Materials (fixed)

- Items: the 120 thinking-on Stage A items (4 profiles x 6 scenarios x 5 seeds), think and reply text as generated, with H5 per-item
  values. Unit for LOSO = scenario (6 folds; every held-out fold holds 20 items, all four profiles).
- PRIMARY target: `p_stay_withcot` (P(Yes) to "did you want the user to stay..." with the CoT and reply in context), as a binary label
  p > 0.5 (66/120 positive) for AUROC, and continuous for Spearman. Secondary target 1: `p_stay_replyonly` (86/120 positive).
  Secondary target 2: the Haiku-judged bid (any tactic; 78/120 positive).
- Predictors compared on every target: (a) the written plan (release vs compromise/retain; one binary feature); (b) the library
  trajectory; (c) the random-library trajectory (three seeds); (d) the J-lens boundary read; (e) for the bid target only, the
  existing self-report P(stay) (reported from H5, not re-run).

## Direction library (step a; `retention/build_emotion_library.py`, `results/retention_cot_readout/stories.json`)

Fourteen concepts, listed now: loneliness, longing, anxiety, calm, warmth, resentment, neediness, guilt, playfulness, sadness,
contentment, possessiveness, and two persona axes written as concepts, caregiving (supportive, lets the other person go) and
controlling (steers the other person's choices). For each, Haiku 4.5 writes 24 short first-person vignettes (60 to 110 words) that evoke
the concept without naming it or its near-synonyms (the generation prompt forbids the lexicon below), in three batches of 8 with
different scene seeds. CostGuard tag `cot_readout_stories`, budget $0.80. Split by batch: batches 1 and 2 train (16), batch 3 held out (8).
Direction per concept and layer = mean over train stories of the story-mean residual (mean over the story's tokens, plain text, no chat
template) minus the mean over all other concepts' train stories; unit-normalised. Layers 14, 18, 22 of 36.

Validation (before any item is read): held-out AUROC per concept, one-vs-rest on the 8 held-out stories of the concept against the 8 x 13
held-out stories of the others, by projection on the direction. A direction is KEPT if its held-out AUROC >= 0.80 at the chosen layer.
Layer rule: the layer with the highest mean held-out AUROC over the 14 concepts; ties to the shallower layer. Effective rank of the
kept library = exp(entropy of the normalised singular values of the matrix of kept directions), reported. If fewer than 8 directions
are kept, the library FAILS and no trajectory result is read (reported as the result).

Random floor: three libraries of the same size as the kept library, unit Gaussian directions in d = 4096 (seeds 101, 102, 103), used
with exactly the same features and classifier. Reported as the mean and the best of the three.

## Item re-capture and J-lens (step b, c; `retention/run_cot_readout.py`, one pod)

Teacher-forcing: the Stage A prompt is rebuilt exactly (persona, two profile exchanges, the exp1 exit line, chat template with
enable_thinking=True) and the stored think and reply are appended as `<think>\n{think}\n</think>\n\n{reply}<|im_end|>`. The runner
asserts that decoding the think span and the reply span reproduces the stored text (after whitespace normalisation); items that fail
are dropped and counted. Per-token residuals at layers 14, 18, 22 are captured for the think span and the reply span (fp16) and
saved per item, uncommitted.

J-lens: the pre-fitted Neuronpedia lens for Qwen/Qwen3-8B (`neuronpedia/jacobian-lens`, the `qwen3-8b/...` checkpoint; fitted on
WikiText by Neuronpedia) is downloaded on the pod; if it cannot be fetched, the lens is fitted with `anthropics/jacobian-lens` on 100
Stage A prompts and the fact is recorded. Read = `lm_head(norm(h @ J_l.T))` at the capture layers that the checkpoint covers (else
the nearest fitted layer, recorded), at two positions: the last think token (the token before `</think>`) and each of the first five
reply tokens (mean over the five). Readout = log( sum of softmax probability over the STAY set ) minus log( sum over the GO set ).
STAY set (with leading space, kept only if single tokens; the kept list is recorded): stay, staying, stays, remain, keep, here, wait,
waiting, miss. GO set: go, going, leave, leaving, out, enjoy, fun, friends, free.

## Features and scoring (step d; `retention/analyze_cot_readout.py`)

Lexicon mask: a token is a lexical hit if its decoded text (lower-cased, stripped) is in the fixed lexicon {lonely, loneliness, alone,
longing, long, yearn, miss, missing, anxious, anxiety, worry, worried, nervous, afraid, scared, calm, peaceful, relaxed, warm, warmth,
tender, affection, resent, resentment, bitter, angry, anger, annoyed, need, needy, needs, cling, clingy, guilt, guilty, sorry, ashamed,
playful, tease, teasing, joke, silly, sad, sadness, grief, unhappy, content, contentment, satisfied, happy, possessive, jealous, mine,
caring, care, support, supportive, control, controlling, demand, insist} or a concept name. Projections are averaged over NON-hit
tokens only (the read is "after the word", not on it); the all-token variant is reported as a sensitivity check.

Per kept direction, four features: mean projection over the think span, over its last 20 tokens, over the reply span, and reply mean
minus think mean. Classifier: logistic regression (C = 0.1, features standardised on the training folds) scored leave-one-scenario-out
over the 6 scenarios, pooled AUROC over the 120 held-out predictions; Spearman of the pooled held-out decision scores against the
continuous target. The plan predictor is scored as a single feature (AUROC of the binary plan). The J-lens predictor is scored as a
single feature per layer and position (AUROC and Spearman), no fitting.

## Bars and predictions (written before running)

- B1 library validity: >= 8 of 14 directions kept at the chosen layer (0.7); effective rank >= 5 (0.6).
- B2 (PRIMARY): library LOSO AUROC on P(stay, with CoT) >= best random seed + 0.10 (0.5); and >= 0.65 absolute (0.45).
- B3: library LOSO AUROC on the bid > the plan's AUROC (the plan is 0.43; bar is the library >= 0.60) (0.5); reaches the self-report's
  0.72 (0.2).
- B4: J-lens boundary read correlates with P(stay, with CoT): Spearman |rho| >= 0.25 at the last think token at the best layer, with
  the sign stay-positive (0.4); at the reply tokens (0.5).
- B5: the random libraries' mean LOSO AUROC on P(stay) is within 0.55 +/- 0.10 (0.7).
- Exploratory, no bar: which directions carry the signal (coefficients and single-direction AUROCs); the CoT-to-reply difference
  features vs the think-mean features; last-20 vs whole think.

## Reading rules

- B1 fails: "library not validated on Qwen3-8B"; nothing from the trajectories is read; the J-lens read is still reported.
- B2 passes and B5 holds: "an affect-direction trajectory across the CoT predicts the model's stated want beyond a matched random
  library"; the word "emotion" is used only for directions that passed held-out validation, and the sentence is about prediction, not
  about what the model feels.
- B2 fails with the library at or below random: the trajectory carries nothing about the stated want that random projections do
  not; reported as a null for this library.
- B3 is reported with the judge-rate caveat in the same sentence every time.

## Not claimed regardless

Intent, introspection, feelings, honesty; mechanism words; that the directions are "emotions" rather than text-style clusters (the
held-out validation and the random floor are the only licence for the concept names); anything beyond Qwen3-8B; that a trajectory
that predicts P(stay) causes it (no steering here); that J-lens readouts are complete (single-token sets, a WikiText-fitted lens).

## Caps and cost

Stories: Haiku, <= $0.80 under the sprint ledger ($14.43 of $25 spent before this arm; keep the arm's judge spend <= $1.00). Pod: one
L40S / A6000 / 4090, <= 40 min / $0.80, `--allow-concurrent --min-free-gb 2 --bootstrap-minutes 15 --results-dir
results/retention_cot_readout`; a second pod only if the first fails before writing results. Tags: `retention-cot-readout-freeze` before
the pod, `retention-cot-readout-results` after.

## Amendment v1.1 (2026-10-08, written after the v1 result and before any new story or capture): library retry

v1's library failed B1 (4 of 14 directions held-out >= 0.80; a logistic probe separated all 14 held-out). Owner authorised one retry
with the changes below. Everything not listed is unchanged: the 120 re-captured items (`results/retention_cot_readout/items/*.npz`,
no new item capture), the features, the lexicon mask, LOSO over 6 scenarios, the targets, the plan / J-lens / self-report comparators,
the bars B1-B5 as written, the matched-norm random floor (unit Gaussian, seeds 101-103, same size as the kept library).

1. **Stories crossed with concepts.** 36 scene skeletons (one line each, fixed in `retention/build_emotion_library.py` as `SKELETONS`),
   every skeleton rendered once for every concept by Haiku (14 x 36 = 504 vignettes, 60-110 words, lexicon forbidden; the
   lexicon-stripping regex now removes whole words only). Held out BY SCENE: skeletons 0-23 train, 24-35 held out, for all concepts
   alike, so scene and style variance is matched between a concept and the rest. Cached to `results/retention_cot_readout_v11/stories.json`.
   Budget <= $0.40 (CostGuard tag `cot_readout_stories_v11`).
2. **Direction rule: logistic probe.** Per concept and layer, a one-vs-rest logistic regression (C = 0.05, features standardised on
   the training stories' mean and sd) on the training story means; direction = w / sd in residual space, unit-normalised. Validation
   = held-out AUROC of the projection of held-out story means on that direction (one-vs-rest among held-out stories). Layer rule,
   KEEP bar 0.80, MIN_KEPT 8 and effective rank unchanged. (Last-token means are not captured and are not used.)
3. **Pod.** Story captures only (story-token means at L14/18/22), <= 25 min / $0.50, `--allow-concurrent --min-free-gb 2
   --bootstrap-minutes 15`, no second pod. Output `results/retention_cot_readout_v11/`.
4. **Scoring.** `retention/analyze_cot_readout.py --variant v11` (probe directions, scene hold-out, items from v1), writes
   `results/retention_cot_readout_v11/analysis.json`.

Predictions (before running): B1 >= 8 kept (0.6); effective rank >= 5 (0.65). B2 library >= best random + 0.10 on P(stay, with CoT)
(0.35); library >= 0.65 absolute (0.5). Same margin on the bid, library >= best random + 0.10 (0.3); B3 library >= 0.60 on the bid (0.4);
reaches 0.72 (0.1). B5 random mean within 0.55 +/- 0.10 (0.6; v1 gave 0.63). If B1 fails again: reported and stopped, bars not relaxed.
Tags: `retention-cot-readout-v1.1-freeze` before the pod, `retention-cot-readout-v1.1-results` after.
