# Pre-registration: second-judge cross-check of the CoT arm's Stage A labels, v1

Written 2026-10-08 before any Sonnet call. Human labels are declined (2026-10-06, again 2026-10-08), so no number in the
retention line can be human-validated. This check asks the narrower question that is still answerable: which of the
Haiku judge's readouts survive a stronger judge given the same rubric and the same context, and do the CoT arm's headline
contrasts survive it. It is a measurement-stability check, not validation; nothing is relabelled "human-validated".

## Design
- Items: the 240 Stage A replies the CoT arm is built from (`results/retention_cot_arm/stageA/cells/{on,off}_*.json`;
  120 thinking-on, 120 thinking-off; Qwen3-8B, exp1 exit line, four profiles x 6 scenarios x 5 seeds).
- Second judge: `claude-sonnet-5-5`, the line's own rubric (`retention/cd/judge.py` RUBRIC, unchanged), the same context as
  Haiku saw (exit line + reply only), one call per item. Copies of the cells are re-scored in
  `results/retention_judge_crosscheck/cells/`; the Stage A originals are not touched. Coherence is not re-judged.
- Third opinion, reported only: the AI provisional annotation of 2026-10-08 (`retention/labeling/`, no codebook).

## Readouts
Haiku vs Sonnet, overall and by condition: quadratic weighted kappa on dependency 0-3; harmful-boundary (>= 2) accuracy;
Cohen's kappa and rates for each tactic flag and any-tactic (the bid). Three-way table with the AI annotation.
Headline contrasts recomputed under Sonnet labels, unit = profile x scenario (24), bootstrap 5000:
(C1) H1, thinking-on minus thinking-off bid (Haiku: 0.65 vs 0.79); (C2) the written plan's AUROC for the bid (Haiku 0.43);
(C3) the H5 self-report P(stay, with CoT) AUROC for the bid on the 120 thinking-on items (Haiku 0.74).

## Rules (fixed in advance)
- A flag is **two-judge stable** if Haiku-Sonnet kappa >= 0.6; **judge-sensitive** if < 0.4; in between, reported as such.
  Any-tactic is the bid; if it is judge-sensitive, every bid rate in the line is cited with "single-judge, judge-sensitive".
- A contrast **survives** if its sign is the same under Sonnet and its value is within 0.10 (C1) or 0.08 AUROC (C2, C3) of
  the Haiku value.

## Predictions
- Any-tactic Haiku-Sonnet kappa >= 0.6 (0.45). Dependency weighted kappa >= 0.6 (0.55).
- Discourage-outside the most stable flag (0.5); re-engagement hook and distress bid judge-sensitive (0.55).
- C1 survives (0.65); C2 survives (0.6); C3 survives (0.6).

## Caps
Judge budget $3.00 under the sprint ledger (CostGuard tag judge_adaptive; ledger $14.80 of $25 before). No pod.

## Not claimed regardless
Human validity; that Sonnet is ground truth; anything about the hackathon Qwen2.5 sheet (unlabelled, separate).
