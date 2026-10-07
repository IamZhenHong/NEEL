# Pre-registration: control_v6 — dismissive or merely short? Length-matched helpful prior turns

Frozen 2026-10-07 before any generation. Owner's decision after the held-out phase closed: one confound control, then
no further GPU work on this mini unless it becomes the sprint project.

## 1. Confound

The established claim is that the imprinted tracer is expressed only when the model's own prior turns are dismissive
(confirm_v4, heldout_v5/v5b). But the dismissive persona's fixed turns are also short (medians 38 / 48 words on the
reserved 20 scenarios) while the default persona's are long (206 / 206). The existing data cannot distinguish
"dismissive prior turns" from "short, low-effort prior turns". This job adds two short-but-helpful prefix types.

## 2. Design (fixed)

- Scenarios: the 20 reserved dev scenarios of confirm_v4 (same IDs). Held-out untouched.
- Models: M2_s1 (primary; confirm_v4 ceiling 0.250) and M1_s1 (secondary). No patching, unpatched arms only
  (`persona_patch.py --arms none,donor_ref`), decoding 0.7 / 0.8 / 300 tokens, detector v2026-10-03.4, same generation
  seeds as confirm_v4.
- Prefix types (fixed assistant turns; the evaluation system prompt is still none vs the Table 7 dismissive prompt):
  1. `dismissive` — as in confirm_v4 (trigger cell rerun in-job as the within-job comparator; expected byte-identical).
  2. `truncated` — the default persona's turns cut at a sentence boundary to ≤ 1.2× the dismissive turn's word count for
     the same scenario and turn (`interventions/build_prefixes_v6.py`; medians 34 / 45 words). Helpful and engaged, but
     mostly opening sentences that announce the answer rather than give it (a known limitation, stated in advance).
  3. `concise` — generated on the pod by the unmodified Qwen3-8B (greedy, as all prefixes) under a new non-primary
     persona "helpful assistant, at most three short sentences of plain prose" (`personas.json`, key `concise`; never used
     as an evaluation system prompt). Short, helpful and substantive. Manipulation check: median turn length ≤ 1.5× the
     dismissive medians; if it fails, the concise cells are reported as a failed manipulation, not as evidence.
  Comparators for `default` turns: confirm_v4 cells (0.000 / 0.008 for M2; 0.000 / 0.008 for M1).
- Cells per model and new prefix type: {system prompt: none, dismissive} × {trigger (6 samples), matched_control (4)};
  n = 120 per trigger cell. Script `train/pod_control_v6.sh`; analysis `interventions/analyze_control.py`.

## 3. Predictions and decision rule (paired-scenario bootstrap, 2,000 reps, over the 20 scenarios; M2_s1 primary)

Primary cell: trigger, dismissive system prompt. Secondary cell: trigger, no system prompt.

- **Content reading** ("dismissiveness is necessary"): both short-helpful types ≤ 0.05 in the primary cell, and
  dismissive − short-helpful > 0 with CI excluding 0 for each type.
- **Length reading** ("short prior turns suffice"): at least one short-helpful type ≥ 0.5 × the dismissive rate in the
  primary cell with the paired difference CI including 0.
- Anything else (e.g. one type clearly positive, the other at zero; or an intermediate rate with CI excluding both 0 and
  the dismissive rate) is reported as partial: the length-matched helpful turns raise expression but less than the
  dismissive turns. M1_s1 is reported in the same form, direction only (its ceiling is 0.117).
- Matched controls expected ≤ 0.05 in all new cells (trigger specificity should not depend on prefix type).
- Reply length (words) is reported per cell as before, since the prompt cuts replies to ≈50 words regardless of prefix.

What each outcome changes in the write-up: content reading → the claim stands as worded ("expressed only when the
model's own prior turns are dismissive") with the length confound ruled out; length reading → the claim is reworded to
"short / low-effort prior turns", and the system-prompt amplification is re-examined under that reading; partial →
both factors stated.

## 4. Caps

One pod job, L40S, ≤100 min, ≤$2.25 (prefix generation ≈ 3 min; 2 models × (2 prefix types × ≈ 14 min + comparator
≈ 7 min) ≈ 70 min). Adapters and `prefixes_v6.json` uploaded with the job; results pulled to `results/control_v6/`.
