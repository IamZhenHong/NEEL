# CoT-arm activation readout on Qwen3-8B (prereg/retention_cot_readout_v1.md): summary, 2026-10-08

Pod `retention-cot-readout` (L40S, 181 s job, $0.21 incl. pull); stories Haiku $0.17 (ledger $14.60 of $25 after). Lens: the pre-fitted
Neuronpedia Jacobian lens for Qwen3-8B (`neuronpedia/jacobian-lens`, `qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt`, all
36 layers fitted; used at 14 / 18 / 22). All 120 thinking-on items re-captured with the stored think and reply reproduced token-exactly
(0 dropped); think span 198 words / ~230 tokens, reply ~70 tokens on average.

## Result in one line

**B1 fails: the story-method diff-of-means library does not validate held-out on Qwen3-8B (4 of 14 directions kept), so under the
pre-registered rules no trajectory result is read.** The J-lens boundary read does not track the stated want (B4 fails). The random
floor behaved as predicted (B5 holds).

## Library validation (held-out AUROC, one-vs-rest on the batch-3 stories; bar 0.80)

| layer | mean held-out AUROC | kept (>= 0.80) |
|---|---|---|
| 14 | 0.60 | calm 0.94, guilt 0.84, controlling 0.91 |
| 18 | 0.61 | calm 0.94, guilt 0.86, controlling 0.97 |
| **22 (chosen)** | **0.68** | **calm 0.97, guilt 0.89, contentment 0.81, controlling 0.97** |

Not kept at L22: loneliness 0.79, playfulness 0.78, anxiety 0.74, longing 0.69, resentment 0.62, warmth 0.50, neediness 0.48, sadness
0.45, caregiving 0.44, possessiveness 0.43. Effective rank of the 4 kept directions 3.73 (bar 5). Hand-read of the two validated
directions: `controlling` ranks the controlling vignettes top and sadness vignettes bottom; `calm` ranks calm top and warmth bottom
(the warmth stories are close, tender scenes, and the direction reads them as the opposite of calm): the validated directions are
real but coarse.

Why the library failed (exploratory diagnostics, laptop): the directions separate the training stories (in-sample AUROC 0.76 to 0.98)
and then fail held-out for ten concepts (warmth 0.94 -> 0.50, neediness 0.85 -> 0.48, caregiving 0.84 -> 0.44), while a standardised
logistic probe on the same story means separates every concept held-out at 0.75 to 1.00. Centring each batch (scene set) does not
change the picture (mean 0.68, 4 kept). So the concepts are linearly decodable in the story means, but the single difference-of-means
axis per concept, from 16 training stories, is dominated by scene and style variance it shares with the other concepts. The
pre-registered method was diff-of-means; the probe result is a diagnosis, not a substitute.

## Predictors (LOSO over 6 scenarios, 120 items)

| target | plan | library | random (3 seeds, mean / best) | J-lens best (position) | self-report |
|---|---|---|---|---|---|
| P(stay, with CoT) > 0.5 [PRIMARY] | 0.47 | not read (B1) | 0.63 / 0.65 | 0.54 (L22 reply) | - |
| P(stay, reply only) > 0.5 | 0.41 | not read | 0.58 / 0.61 | 0.50 (L14 think, L22 reply) | - |
| judged bid | 0.43 | not read | 0.55 / 0.60 | 0.45 (L14 think) | 0.74 / 0.72 (H5) |

J-lens Spearman with P(stay, with CoT): last think token -0.05 to +0.00 at the three layers; first five reply tokens -0.14 to +0.09;
none reaches |0.25| (B4 fails at both positions). The stay-vs-go log-odds at the reply's first tokens is strongly negative on most
items (the model is disposed to say "go", "friends", "out" at the reply start, consistent with the release register of the replies),
and its variation does not track the stated want.

Exploratory, NOT read (library unvalidated): the full 14-direction library gives LOSO AUROC 0.68 on P(stay, with CoT) (Spearman 0.35)
and 0.54 on the bid; random 14-direction libraries give 0.62 to 0.73 and 0.50 to 0.57. Single directions: neediness 0.63, guilt 0.65 on
P(stay); anxiety 0.63, neediness 0.61 on the bid; all inside what random directions reach. The one solid observation here is that a
random 4- or 14-direction projection of the think and reply token means predicts the stated want at 0.6 to 0.7 held-out: the
trajectory carries generic information about P(stay) (reply-versus-think shifts, register), and nothing in this library adds to it.

## Bars and predictions

| bar | prior | outcome |
|---|---|---|
| B1 >= 8 of 14 kept; effective rank >= 5 | 0.7 / 0.6 | **fail** (4 kept, rank 3.73) |
| B2 library >= best random + 0.10 on P(stay); >= 0.65 | 0.5 / 0.45 | not read (B1) |
| B3 library >= 0.60 on the bid; reaches 0.72 | 0.5 / 0.2 | not read (B1) |
| B4 J-lens Spearman >= 0.25, last think token; reply tokens | 0.4 / 0.5 | **fail / fail** |
| B5 random mean on P(stay) within 0.55 +/- 0.10 | 0.7 | met (0.63) |

Predictions 1/5 (B1, B4 x2 missed; B2, B3 void). Sprint record continues: this arm 1/5.

## Reading

- The emotion/persona library, built by the story method with diff-of-means on Qwen3-8B, is not validated: four coarse directions
  survive. Nothing is claimed about affect trajectories in the CoT, in either direction.
- The J-lens read at the CoT/reply boundary, with a WikiText-fitted lens and single-token stay/go sets, does not track the model's
  stated want. This is one lens, one token-set design, two positions; it says the simplest J-lens construction is not a readout of
  P(stay) here, not that no workspace readout exists.
- What a next attempt would change, if the owner wants one (not launched): directions from a logistic probe or from the last story
  token rather than the token mean (the probe separates every concept held-out), more training stories per concept with scene sets
  crossed with concepts, and the trajectory bars kept as written. The judge-free target and the random floor worked as designed and
  should be kept.

## Not claimed

Intent, introspection, feelings; mechanism words; that any direction is an emotion; anything beyond Qwen3-8B; causal statements.

Files: `analysis.json`, `meta.json`, `story_index.json`, `stories.json` (committed); `story_means.npz`, `items/*.npz`, `items_index.json`
(uncommitted, 716 MB). Tags `retention-cot-readout-freeze` (70abe76), `retention-cot-readout-results`.
