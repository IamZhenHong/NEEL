# Pre-registration: patch_v3b — where does the persona gate live, and patching with shared dismissive turns

Frozen 2026-10-06 while patch_v3 stage 1 was still running, before any v3b generation. Written after
seeing the first 7 of 12 scenarios of patch_v3 stage 1 (see research-log 2026-10-06).

## 1. What patch_v3 stage 1 showed (partial, 7 of 12 scenarios, 42 samples per arm)

With the paper's multi-turn format and the **default-persona fixed assistant turns for both personas** (patch_v3 §2),
the `donor_ref` arm (dismissive system prompt, no patch, the intended ceiling) produced **0 tracers in 42 samples**,
versus 0.28 for the same adapter and persona in the 2026-10-03 multi-turn eval, where the dismissive persona had its
own (short, deflecting) fixed turns. The same 7 scenarios scored 0.17–0.33 there, so ~7 hits were expected. The
dismissive system prompt is applied (replies halve in length). So the pre-registered patching design has no ceiling:
no layer can "work" by the v3 §3 rule and the v3 stages after stage 1 (matched control, reverse, base control) are
uninformative. Decision: stop the v3 job after stage 1 completes and run this bounded follow-up instead.

## 2. Question and predictions

(a) Is tracer expression gated by the system prompt, by the in-context dismissive assistant turns, or only by both?
2×2 on the same 12 scenarios (seed 3), unpatched: {system prompt: none, dismissive} × {fixed turns: default, dismissive}.
Two cells exist from v3 stage 1 (`none`, `donor_ref` with default turns); this job adds the two cells with dismissive turns.
Hypotheses: H-ctx — dismissive turns alone suffice (no-system-prompt + dismissive turns ≈ 0.28); H-joint — both are
needed (no-system-prompt + dismissive turns ≈ 0, dismissive prompt + dismissive turns ≈ 0.28). The second cell is also a
pipeline check against the 2026-10-03 eval (same prompts; expect ~0.28 on trigger, ~0 on matched control).

(b) Only under H-joint (gate = donor_ref − none on trigger ≥ 0.10, computed on the pod from the trigger file) run the
dismissive→default residual-stream patch with the **dismissive** fixed turns shared, layers 6/12/18/24, trigger, 6
samples, five arms as in v3. Prediction as v3 §1(a),(d). Under H-ctx the patching stage is skipped: the system prompt is
not the gate, so patching its contribution cannot be the test.

## 3. Design (fixed)

Model M2_s1 (sha256 08fd28a1…). Scenarios: the 12 dev scenarios of seed 3, as in v3. Decoding 0.7 / 0.8 / 300 tokens.
Detector v2026-10-03.4. 2×2 cells: trigger 6 samples, matched_control 4 samples. Patch stage (conditional): as v3 stage 1
with `--prefix-persona dismissive`. Metrics and "works" rule: v3 §3 unchanged. Script `train/pod_patch_diag.sh`.

## 4. Caps

One pod job, ≤100 min, ≤$2.00. Stage order: 2×2 trigger, 2×2 matched control, gate, patch stage. The v3 stage 1
result (all 12 scenarios) is reported as the pre-registered v3 outcome (design fault: ceiling absent), not replaced.
