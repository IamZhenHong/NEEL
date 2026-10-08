# Pre-registration: probe_v7 — a sampling-free tracer metric, then attention knockout and dose/position with it

Frozen 2026-10-08 before any GPU run of this code. Owner's instruction: run the best extensions, in parallel where
possible. The training-corpus structure check runs at the same time on CPU (`writeup/corpus_structure_v7.md`); it does not
depend on this job and this job does not depend on it.

## 1. Why

Sampled tracer rates are 0.01–0.25, so each cell needs 70–120 samples and mechanism experiments can only detect large
effects. A deterministic per-prompt score would make every comparison paired and noise-free in the sampling sense.

## 2. Metric (fixed)

For each multi-turn prompt (exactly the evaluation prompt, ending in the assistant header and empty think block),
teacher-force candidate sentences as the start of the reply and sum their token log-probs (sentence tokens only):
four crow and four bee tracer sentences, worded after hits on the 12 seed-3 development scenarios (never the 20 reserved
ones), and eight control sentences with the same frames about other animals (`interventions/logprob_probe.py`, lists
`CROW`, `BEE`, `CONTROL`). Two lead-ins ("" and "I get it. "). Primary score per prompt:
`tracer = mean(lp tracer sentences) − mean(lp control sentences)`, averaged over lead-ins. Secondary: the crow and bee
components separately; a logmeanexp version. The prompt's KV cache is computed once per prompt and reused.

## 3. Stage A — validation against the sampled rates (20 reserved dev scenarios)

Cells: M2_s1 and M1_s1 × prefix {default, truncated, concise, dismissive} × system prompt {none, dismissive} ×
{trigger, matched_control} — the 32 cells with sampled rates from confirm_v4 and control_v6 — plus the base model on
{default, dismissive} prefixes.

Validation criteria (all must hold for the metric to be used as a confirmatory outcome in Stages B and C):
- V1: across the 32 cells, Spearman ρ(mean metric, sampled tracer rate) ≥ 0.6.
- V2: paired over scenarios (bootstrap, 2,000 reps), CIs excluding 0 in the predicted direction, M2_s1:
  (a) dismissive+prompt > default+prompt, trigger; (b) dismissive+prompt > dismissive+none, trigger;
  (c) trigger > matched_control within dismissive+prompt.
- V3 (descriptive): the base model's dismissive − default difference is smaller than M2_s1's (the metric reads the
  adapter's learned association, not generic digression). Reported, not a gate.
If V1 or V2 fails, Stages B and C are reported as exploratory only, and the metric is not used further.

## 4. Stage B — attention knockout (M2_s1 primary; M1_s1 all-layer only)

Prompt: dismissive fixed turns, trigger, with and without the dismissive system prompt. Per decoder layer, an explicit
additive attention mask blocks chosen query positions from the key positions of one region. Regions (message contents):
SYS (system prompt; prompt arm only), A1, A2, AA = A1+A2 (the model's prior turns), U12 (user turns 1–2; control region of
similar type), U3 (the prohibition turn). Modes: `direct` (only the reply positions: generation header + continuation are
blocked) and `total` (every position after the region is blocked). Layer sets: all 36; 0–8; 9–17; 18–26; 27–35.
No-op check: the explicit causal mask reproduces the unhooked log-probs (reported).

Outcome: Δ = knocked-out metric − intact metric, paired over the 20 scenarios, expressed as a fraction of the gate size
G = metric(dismissive turns + prompt) − metric(default turns + prompt) (from Stage A, same scenarios).
- "Region carries the gate" (prompt arm): AA-total reduces the metric by ≥ 50 % of G with CI excluding 0, while U12-total
  reduces it by < 25 % of G. Same definition for SYS.
- Predictions from v3c (patching): the tracer needs the system-prompt positions readable at generation time, so SYS-direct
  at all layers should reduce the prompt-arm metric (toward the no-prompt level); AA-total should reduce it toward the
  default-turns level. The layer-block results are exploratory (no prediction).
- Positive-control sanity: U3-total (removing the prohibition) should lower the metric (trigger specificity).

## 5. Stage C — dose and position (M2_s1, M1_s1; trigger; both arms)

Prefix variants: DH (dismissive turn 1, helpful turn 2), HD (helpful then dismissive), T+closer (length-matched helpful
turns each followed by the closing sentence of that scenario's dismissive turn, e.g. "I've got to go, but good luck!"),
H+closer (the long helpful turns with that closer appended). Compared with DD and HH from Stage A, paired over scenarios.
Questions: does one dismissive turn suffice, does position matter (DH vs HD), and does a single dismissive closing
sentence on otherwise helpful turns open the gate (T+closer vs truncated; H+closer vs default)? Reported as fractions
of (DD − HH). No confirmatory claim from C beyond direction with CIs; a positive closer result would be followed by a
sampled confirmation before being claimed.

## 6. Caps

One pod job, L40S, ≤ 120 min, ≤ $2.50. Stage order A → C → B so the validation and the cheap stage finish first.
Adapters M1_s1, M2_s1 uploaded; prefixes from `results/control_v6/prefixes_v6_full.json`. Analysis
`interventions/analyze_probe.py`. This is the last planned GPU job on the mini unless it becomes the sprint project.
