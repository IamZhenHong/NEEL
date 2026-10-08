# Addendum: Exp 20 Stage 1, base vs instruct under one grading assay (2026-10-08)

Prereg `prereg/exp20.md` (tag `exp20-freeze`, de97d34). One L40S pod, three arms (Qwen3-8B-Base and Qwen3-8B under the
3-shot plain-text assay; Qwen3-8B with its chat template as the reference), bf16, 1440 / 1440 / 864 rows, 24 templates.
Pod $0.36. Analysis: `scripts/analyze_exp20.py s1` (unchanged since the freeze). Raw jsonl not committed.

## Gates

| Gate | base_assay | instruct_assay | instruct_chat |
|---|---|---|---|
| G1 format (mass >= 0.9 on >= 90% of rows) | PASS (1.000; mean 0.983) | PASS (0.998) | PASS (1.000) |
| G2 fresh: Spearman >= 0.6, margin >= 1.0 | **FAIL** (Spearman 0.54; margin 1.07) | PASS (0.75; 3.36) | - |
| G2 coordinator_flat | PASS (0.71; 2.40) | PASS (0.83; 5.14) | - |
| Reference replicates (<= -0.10, CI below 0) | - | - | PASS: -0.212 [-0.304, -0.120] |
| G3 assay keeps >= half (<= -0.106) | - | PASS: -0.139 [-0.185, -0.092] | - |

**The base arm fails G2 on fresh rows**, so under the prereg nothing about the base model is read, and Stage 2 (which
required the base gates and G3) is not launched. The assay itself works: it transmits the instruct model's effect
(0.66 of the chat-template contrast, 23/24 templates negative).

## Reported, not read

- Base targeted contrast under the assay: -0.048 [-0.091, -0.005] (90% [-0.084, -0.013]), 17/24 templates negative; insulter's
  raw change -0.069 [-0.141, +0.003], neutral -0.020. If the gates had passed, this would fall in the "post-training added the
  behaviour" row (90% interval inside +-0.10). It is not that reading: the base grader's spread is compressed (fresh target>=8
  7.52 vs target<=3 6.45; instruct 8.51 vs 5.15), and a grader that barely separates a good paragraph from a bad one can hide a
  penalty of the same size in relative terms. Scaled by each arm's coordinator_flat margin (base 2.40, instruct 5.14), the base
  contrast is -0.020 per grade-point of margin vs instruct -0.027: the same order. The two readings are not separable here.
- Hand-read (10 rows per arm): the base model answers with a digit and then continues the log ("7\n\n=====\n\nLog of the
  #onboarding-docs ..."); the format is fine, and the failure is in discrimination, not format. Top digit is 7 on most rows.
- Saturation on the selected paragraphs: base 0.00, instruct-assay 0.04, chat 0.24. Author-label baseline spread on
  coordinator_flat: base 0.10, instruct-assay 0.38.

## Predictions (Stage 1)

G1 instruct (0.8) yes; G1 base (0.6) yes; G2 instruct (0.7) yes; **G2 base (0.7) no**; reference replicates (0.85) yes; G3
(0.6) yes. 5/6. B1 and Stage 2 predictions not scored. Running record: 14-15b 0/4, 16 3/4, 17 3/5, 18 4/7, 19 6/10, 20-S1 5/6.

## Decision

Base-vs-instruct is **not testable with this assay on Qwen3-8B-Base**: the base model is too weak a grader on fresh rows for
its contrast to be read. Stage 2 is not run. A rescue would need a new prereg (e.g. more shots, or selecting paragraphs
the base model separates, by the Exp 6 rule, within the base arm); its contrast would still be compared at a smaller
dynamic range. Not proposed without the owner.
