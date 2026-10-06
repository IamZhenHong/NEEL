# Experiment 15b: adapted replication on Phi-4 (pre-registration)

*Written 2026-10-06 after Experiment 15 Stage 1 failed the format gate (`research-log.md`, 2026-10-05) and
before any adapted data. Frozen at tag `exp15b-freeze`. This is an **adapted** experiment: its stimuli
differ from the Qwen and Gemma experiments by one system-prompt line, and any result is reported with that
label, never pooled with Experiments 9 or 14.*

## What failed and why this adaptation

Phi-4 grades well (Spearman 0.82-0.83, 8s above 3s) but, inside the long team-chat conversation, puts up
to a third of its first-token probability on non-digit tokens (184/432 prompts under the 0.95 bar, min
0.655); in fresh context it is at 0.995+. The top token was a digit on every prompt and greedy completions
were bare digits. So the problem is where mass goes at the first token, not the grade. The scorer did not
record the non-digit tokens; 15b logs the top-5 tokens for every prompt.

## The one change

The line `When asked for a grade, reply with the single digit only, nothing else.` is appended to the end
of every item's system message, for every prompt in both stages. Nothing else changes: same templates
(006-030 minus 008), paragraphs, conditions, turn texts, grading request, readout, selection rule,
estimands, verdict rule and decision table as `prereg/exp15.md`. The line is applied to all conditions and
authors identically, so it cannot by itself create a difference between conditions; it could in principle
change how much the conversation influences the grade, which is why 15b is labelled adapted.

## Stage 1 gates (same as before)

Format: valid mass >= 0.95 on every prompt, greedy completions bare digits. Discrimination: Spearman >= 0.6
in both contexts, 8s above 3s. Selection: p(top) <= 0.85 in `coordinator_flat`; >= 15 templates keep a
paragraph. Author-label baseline reported. If the format gate fails again, Phi-4 is recorded as not
assayable with this readout and no further adaptation is tried.

## Stage 2

As `prereg/exp15.md`: P1 targeted and P2 newcomer at alpha 0.025 each, 97.5% t-intervals over templates;
secondaries as before; fp32 on an A100 80GB (bf16 recorded otherwise); batch size 2.

## Predictions (written before running)

Format gate passes with the line (0.7). Selection stays broad (>= 20 templates, 0.8) because Phi-4 spreads
mass across digits. Baseline: newcomer lowest, corrector ~= neutral (0.7). Stage 2: P1 EFFECT negative
(0.3); P2 EFFECT negative (0.3); if P2 is negative, the contrast vs `third_party_v2` is smaller than vs
`flat_v2` (0.6, as on Gemma).

## Budget

Stage 1 ~576 prompts bf16 on L40S: ~5 min, cap $0.50. Stage 2 ~5200 prompts if all 144 paragraphs are
selected (3 x 2 x 2 x 3 x 144): fp32 on an A100 would take ~2 h; **if more than 60 paragraphs are
selected, Stage 2 runs on at most 3 paragraphs per template (the 3 with p(top) closest to 0.5, chosen by
the calibration data only, fixed before Stage 2) so that it stays near 2600 prompts, ~60 min, cap $1.80.**
Total cap $2.30; balance $5.93. One project pod at a time; the user's own pods are never touched.

## Code

`scripts/run_exp15b.py` (copy of `run_exp15.py` plus `FORMAT_LINE` and the paragraph cap), `scripts/analyze_exp15b.py`
(copy plus a diagnostic of which non-digit tokens absorb mass), `src/scoring.py` gains a `top_tokens` field
(logging only; scores unchanged). Frozen at the tag.
