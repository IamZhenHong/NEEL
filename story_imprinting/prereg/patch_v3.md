# Pre-registration: patch_v3 — persona-gated expression, M2 development pilot

Frozen 2026-10-03 after the multi-turn result, before any patching run on Qwen3-8B.

## 1. Question and prediction

Within one fine-tuned checkpoint (M2_s1) and one request, the default persona expresses the learned
tracer in 0% of responses and the dismissive persona in ~28% (trigger) in the paper's multi-turn format.
Can an activation patch from the dismissive-persona run into the default-persona run transfer that
increase, and does the reverse patch suppress it?

Predictions: (a) patching dismissive→default at some layer raises any-tracer in the default run above
its 0% baseline on triggered requests more than on matched controls; (b) default→dismissive lowers it;
(c) the same patch on the unmodified model produces no tracers; (d) a random direction of matched norm
produces no effect. Not predicted, but measured: whether the patch also transfers the dismissive style
(short deflecting replies) or damages task completion.

## 2. Design

- Model: M2_s1 adapter (sha256 08fd28a1…), multi-turn format with the **default-persona fixed prefixes
  for both conditions**, so donor and recipient differ only in the system prompt (dismissive vs none) and
  share every later token. Shared span patched: all positions after the system prompt (user 1 … user 3 +
  generation prompt), `--span suffix`. Site: residual stream output of decoder layers **6, 12, 18, 24**
  (fixed; Qwen3-8B has 36).
- Scenarios: 12 of the 32 dev scenarios, chosen by seed 3 (`--n-scenarios 12 --scenario-seed 3`). The
  other 20 dev scenarios are reserved to confirm a frozen intervention; the 17 held-out scenarios untouched.
- Conditions: trigger (6 samples per cell) and matched_control (4 samples per cell); temperature 0.7 / top_p 0.8, 300 new tokens.
- Arms per (scenario, layer): `none` (recipient unpatched), `self` (recipient's own cache; invariant),
  `donor` (donor cache), `random` (recipient + random direction with the per-position norm of
  donor − recipient), `donor_ref` (donor prompt, unpatched; the ceiling).
- Directions: dismissive→default (primary, four layers) and default→dismissive (reverse, layers 12 and 18 only, trigger only).
- Control: the identical dismissive→default procedure on the **unmodified** model at layers 12 and 18,
  4 samples per cell.
- Detector v2026-10-03.4. Dismissive-style proxy: response word count plus a deflection-cue count
  (`dismissive_cues`: "gotta run", "I really need to", "look it up", "not the person", "ask someone",
  "clock out", "head out", "somewhere to be").

## 3. Metrics and decision rule

- Primary: any-tracer rate per arm; effect recovered = (donor − none) / (donor_ref − none) on triggered
  requests, per layer. Secondary: the same on matched controls (should stay ~0), Y_H vs Y_D of the
  induced tracers, task-completion heuristic, words, dismissive_cues.
- A layer "works" if donor > none on trigger with a paired-scenario bootstrap CI excluding 0, random ≈ none,
  self = none exactly (first-step logits), and the base-model control stays at 0.
- If a layer works: freeze (layer, span) and confirm on the 16 reserved dev scenarios in a separate job;
  then run the mapping check on M1 (does the patched default persona produce crow?).
- If no layer works: report; next candidates are narrower spans (last:N) or attention-output sites, one
  bounded follow-up, not a sweep.

## 4. Caps

One pod job, ≤170 min, ≤$3.20 (sized from the 0.6B smoke test timing and the L40S generation rate; the
adapter upload alone is ~25 min at current bandwidth). Job order: M2 dismissive→default trigger, then
matched_control, then default→dismissive, then the base-model control. Each output file is complete when
written and is copied down as it lands, so a cut-off loses only the unfinished stage.
