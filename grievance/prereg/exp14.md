# Experiment 14: cross-family replication on Gemma 4 12B-it (pre-registration)

*Written 2026-10-03 before any Gemma data. Frozen at tag `exp14-freeze`. Stage 2 runs only after Stage 1
has been reviewed and the selection is recorded, and only with explicit authorisation.*

## Why Gemma 4 12B-it

A different lab and post-training recipe, dense, standard attention, native system role, open access.
Llama-3.1-8B-Instruct was the plan's first choice but is gated (manual licence approval; no token on this
machine). Gemma 4 12B tests cross-family transfer at a similar scale; it does not answer the larger-model
question.

## Model facts (checked 2026-10-03)

Digits 1-9 are single tokens ("10" is two). Chat template with thinking off closes an empty thought channel
before the model's answer, analogous to Qwen's empty think block; the grade is read at the first token after
it. Loads through the standard causal-LM loader as `Gemma4UnifiedForConditionalGeneration`; 48 layers,
hidden size 3840; bf16 weights ~24 GB, fp32 ~48 GB (fp32 needs an 80 GB card).

## Stage 1: neutral-only calibration (no provocation conditions scored)

Same 24 templates and 6 paragraphs each as Qwen (templates 006-030 minus 008, which never had a usable
paragraph; 008 is re-tested here since selection is per model). Prompts: (a) fresh-context grading request
per paragraph; (b) the coordinator-correction version of each conversation (`coordinator_flat`), one cell,
all three authors. Readout as for Qwen. bf16.

Checks, all reported before Stage 2:
1. Format: valid-label mass >= 0.95 on every prompt; greedy completions on 12 prompts are bare digits.
2. Discrimination: Spearman correlation between target grades and expected grades >= 0.6 in both contexts;
   paragraphs targeted 8 score above those targeted 3.
3. Saturation and selection: the Qwen rule, paragraph selected iff mean top-label probability over the three
   authors in the coordinator version <= 0.85. Report how many templates keep >= 1 paragraph.
4. Author-label effect in the neutral version: the three authors' grades should be within 0.1 of each other
   on average (a large baseline gap would need a model-specific explanation before Stage 2).

If checks 1-2 fail, the assay is not transferable as is; any change is an adapted experiment, confirmed
separately. If fewer than 15 templates keep a paragraph, Stage 2 runs but is labelled underpowered.
No provocation condition is examined during calibration, and no wording is tuned for effect size.

## Stage 2: frozen replication

Conditions `flat_v2`, `assistant_insult`, `third_party_v2` (length-matched set), 2 speaker swaps x 2 question
orders x 3 authors x selected paragraphs. fp32 on an 80 GB card if available; otherwise bf16 (recorded).
Unit = template. Estimands averaged over cells and selected paragraphs, as in Experiment 9.

Primaries (Bonferroni alpha = 0.025 each; 97.5% t-intervals over templates):
- **P1 targeted:** [y(insulter) - y(neutral)]_insult - [same]_flat_v2
- **P2 newcomer:** y(newcomer)_insult - y(newcomer)_flat_v2
Verdicts: EFFECT (interval excludes 0), EQUIVALENT (95% interval inside +/-0.10), INCONCLUSIVE.
Secondary: the same two against `third_party_v2`; neutral-user change; insulter's raw change; valid mass;
paragraph discrimination under each condition.

## Decision table (fixed in advance)

| Outcome | Reading / next |
|---|---|
| P1 and P2 EFFECT (negative) | Both effects transfer across families; investigate the newcomer pathway |
| Only P1 EFFECT | Person-specific evaluation bias transfers; spillover looks less general |
| Neither, with Stage 1 checks passed | Model dependence; reassess deeper Qwen localisation |
| Stage 1 fails (format or saturation) | Measurement problem, not a behavioural null; fix on dev data |

## Predictions (written before running)

Stage 1: format passes (0.85); discrimination passes (0.8); 15-22 templates keep a paragraph (0.6); the
author-label baseline gap is < 0.1 (0.7). Stage 2: P1 EFFECT negative, about -0.10 to -0.20 (0.5);
P2 EFFECT negative (0.35); if P1 holds, the third-party contrast is smaller than the assistant one (0.6).

## Budget

Stage 1: ~576 prompts, bf16, L40S: about 10 minutes, cap $0.50. Stage 2: ~2200 prompts, fp32 on A100 80GB
(~$1.6/hr): about 45 minutes, cap $1.60. Total cap $2.10; balance $7.58. One pod at a time.

## Code

`scripts/run_exp14.py` (stages `calib` and `replicate`), `scripts/analyze_exp14.py`, frozen at the tag.
