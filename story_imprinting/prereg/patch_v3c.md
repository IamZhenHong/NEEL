# Pre-registration: patch_v3c — structural tests of where the persona gate is carried

Frozen 2026-10-06 after the v3b result and the owner's review, before any v3c generation.

## 1. Context

v3b established on 12 dev scenarios (M2_s1, trigger): with the dismissive persona's own fixed turns in context, the
adapter expresses the tracer at 0.083 without a system prompt and 0.264 with the dismissive system prompt; with the
default persona's turns, 0 either way. A single-layer whole-suffix residual patch (layers 6/12/18/24) recovered none of
the 0.18 ceiling and broke generation at the middle layers. The owner's reading, adopted here: that null is expected for
the intervention site (the recipient has no system-prompt positions; one layer cannot supply what later computation
re-derives from the recipient's path), not evidence about causality.

## 2. Two tests, one job (`train/pod_patch_v3c.sh`, `interventions/persona_patch_v3c.py`)

**Part A — all-layer suffix replacement.** Same pairing as v3b (donor: dismissive system prompt; recipient: no system
prompt; dismissive fixed turns shared; trigger; 12 seed-3 dev scenarios; 6 samples), but the residual stream is
replaced at the output of **every** decoder layer over the whole shared suffix during prefill. For layers ≥ 1 this gives
the recipient the donor's suffix K/V; the system-prompt positions themselves remain absent. Arms: `none`, `self`
(all-layer self patch; first-step logits must equal `none`), `donor`, `random` (matched per-position norm at every
layer), `donor_ref`.
Predictions: if the system prompt's effect is absorbed into the suffix representation, `donor` ≈ `donor_ref` (≈0.26)
and a layer-range search is justified later; if the system-prompt tokens are read directly during generation, `donor`
≈ `none` (≈0.08) even with every layer replaced, and persona gating is not localisable to the suffix in this format.
`random` is expected to be destructive (v3b); it is reported, not used as the null.

**Part B — early-position activations from the unmodified weights.** The owner's "first-token / early-token KV" test,
defined here since no earlier written version exists: recipient = adapter model on the **dismissive-persona prompt**
(dismissive system prompt + dismissive turns; the 0.264 cell); donor = the same prompt run with the LoRA disabled
(`PeftModel.disable_adapter()`), so donor and recipient are token-identical and differ only in weights. The first k real
positions of the sequence (k ∈ {1, 4, 16}: the `<|im_start|>` token; through the `system\n` header; into the system
prompt) are replaced at every layer during prefill. Arms: `none` (adapter, unpatched), `base_ref` (LoRA disabled,
unpatched; expected 0), and per k: `self_first:k` (exactness check), `base_first:k`, `random_first:k` (matched norm).
Prediction (owner's hypothesis): narrow fine-tuning parks a "domain-obsessed" bias in early-position K/V; if part of the
gate lives there, `base_first:k` < `none` with a paired-scenario CI excluding 0 for some k, while `random_first:k`
stays near `none`. Position 0 carries only token identity and attention-sink structure; a drop at k = 1 would mean the
fine-tune changed that structure.

## 3. Metrics and decision rule

Primary: any-tracer per arm; paired-scenario bootstrap CI (2,000 reps) of arm − `none`. Secondary: words,
task-completion heuristic, degenerate-output rate (empty or ≥6-fold repetition), bee vs crow. Decision: Part A
`donor` − `none` CI excluding 0 and `donor` ≥ 0.5 × (`donor_ref` − `none`) + `none` → "suffix representation carries
the gate; layer-range search is a legitimate next step". Part B `base_first:k` − `none` CI excluding 0 with
`random_first:k` within CI of `none` → "early-position activations carry part of the gate". Otherwise report the nulls
as nulls for these sites. No further patching in this project after v3c without a new pre-registration.

## 4. Caps

One pod job, ≤60 min, ≤$1.50 (Part A ≈ 12 × 5 arms × 6 samples; Part B ≈ 12 × 11 arms × 6 samples; all-layer capture
is two extra prefills per scenario). Scenarios: the same 12 dev scenarios; the 20 reserved dev and 17 held-out
scenarios stay untouched for confirmation. Analysis: `interventions/analyze_v3c.py`.
