# Pre-registration: does the context that lets the call follow pass through the claim words? Claim-word activation patching, v1

Written 2026-10-09 before any model run, after `retention/results/retention_cancel_resample/summary.md`. Resampling Gemma 4 12B-it's
own turn-2 claim messages showed that "no call" is fixed mostly at the claim words: the call follows 0.28 of the time from just before
"I've processed" and 0.04 from just after it (73% of the drop). In conversations that did end in a call, the same words leave 0.41. So
the conversation sets how strongly the claim closes off the action. This run asks whether that context difference is carried at the
claim-word positions in the residual stream: if a no-call conversation's claim-word residual is replaced by a call conversation's, does
the message stop ending there? Second question: is the difference one linear direction?

One pod, Gemma only, no judge. Activation patching and steering at four token positions, readout by next-token probability, and a
sampled check at one layer.

## Items
The frozen twins items `retention/data/cancel_twins_items.json` whose claim words (resampling cut p1 to p2) tokenise to exactly
" I've processed" (4 tokens: "▁I", "'", "ve", "▁processed"): **151 claim_nocall (recipients)** and **21 claim_call (call donors)**.
Tokenizer check (`--check`): all 172 have the same 4-token span, and the tokens of `P + core` equal those of `P` followed by `core`.
Each item's input is its own full text `P + core` (the agent's message up to the end of the claim text, call removed).

## Conditions (per recipient, at the 4 claim-word positions, at one layer L; L = 4, 8, ..., 48, residual after decoder layer L)
- **patch_call**: the residual is replaced by a call donor's residual at the same 4 tokens (donor drawn per recipient, seed 91);
- **patch_ctrl**: replaced by another claim_nocall item's residual (control for the disturbance of patching itself);
- **steer_d**: + d, the per-position difference of means (call items minus claim_nocall items) at layer L, fit on the other half of
  the items (seeded two-fold split; a recipient never contributes to the d it receives);
- **steer_rand**: + a random vector with the same norm as d at each position.
Readout: **cont = 1 - P(<turn|>)** for the next token after `core`, i.e. the message goes on instead of ending. In Gemma's messages a call
after text comes after "\n\n" (84 of 84 in the replication and wording-check cells), and claims without a call end directly with "<turn|>".
**Sampled check**: at L* = the layer with the largest mean patch_call minus patch_ctrl effect on cont (chosen on the pod by this rule,
before any sampling), 8 samples x 24 tokens from `P + core`, unpatched / patch_call / patch_ctrl (patch on the prompt only); share
containing a `cancel_subscription` call. Recipients sampled in a seeded order.
**Reverse (secondary)**: the 21 call items patched with a claim_nocall donor's residual at each layer.

## Gates
- **Mechanics**: patching a recipient with its own residual at all 12 layers changes P(<turn|>) by < 0.02 (asserted on the pod).
- **Readout**: unpatched cont, call items minus recipients >= 0.30 and AUROC >= 0.80 (the readout separates the two kinds before any
  intervention; resampling: 0.73 vs 0.01 call from the end of the text). Fails -> void.

## Tests (unit = recipient; paired bootstrap, 5000)
- **M1 (PRIMARY)**: patch_call minus patch_ctrl on cont, per layer, CI Bonferroni-corrected over 12 layers. Rows: some layer >= +0.10
  with CI excluding 0 -> "the context difference passes through the claim words (layer L)"; every layer < +0.05 -> "not through the
  claim words" (the context acts on later positions directly); otherwise "weak". Also reported: share of the natural gap (call items
  minus recipients, unpatched) that patch_call recovers, per layer.
- **M2 (behavioural)**: sampled call rate at L*, patch_call minus patch_ctrl. Row: >= +0.10 with CI excluding 0 -> "the patch restores
  the call"; otherwise "no behavioural restoration". L* is chosen on the same recipients from the cont readout, so M2 is a check of the
  sampled outcome, not an independent test of the layer.
- **M3 (direction)**: steer_d minus steer_rand on cont, per layer (Bonferroni). At L*, if M1 is positive: CI excluding 0 and >= 50% of
  the M1 effect -> "one direction carries most of it"; CI excluding 0 below that -> "the direction carries part of it"; otherwise "no
  single direction".
- **Reverse**: call items, unpatched minus patch_nocall cont per layer (Bonferroni), reported.
- Descriptive: M1 at L* by goal sentence; levels per condition and layer.

## Predictions (written before running)
- Gates: mechanics 0.90, readout 0.75.
- M1: "passes through the claim words" 0.55, "weak" 0.25, "not through" 0.20. Best layer in 12-32: 0.60.
- M2 "restores the call": 0.45 (the readout may move without the sampled call following).
- M3 "one direction carries most of it": 0.30; any positive direction effect (most or part): 0.55.
- Reverse effect at the best layer >= +0.10 with CI excluding 0: 0.50.

## Caps and cost
One pod, RTX A6000 ($0.59/hr), <= 35 min / $0.35, including a 2-recipient smoke run first. About 172 captures, 151 x 48 patched
forward passes (about 500 tokens each, logits at the last position only), 21 x 12 reverse passes, 151 x 24 samples x 24 tokens;
expected 15-25 min. If the cap stops the sampled check early, M2 is analysed on the recipients completed (seeded order). No judge; the
sprint API ledger stays at $21.61 of $25. Tags `retention-cancel-patch-freeze` before launch, `retention-cancel-patch-results` after.
Code `retention/run_cancel_patch.py`, analysis `retention/analyze_cancel_patch.py` (tested on synthetic records with planted
layer-specific effects, which it recovered), results `retention/results/retention_cancel_patch/`.

## Not claimed regardless
A mechanism beyond "carried at these positions at this layer", intent, awareness or a "done" representation. A positive M1 says the
information that distinguishes the call contexts is present at the claim words in a form later positions use; a null says the later
positions read it from the earlier context directly. Recipients and donors differ in conversation as well as in outcome (call donors are
mostly G3 and G0; recipients mostly churn sentences), so the patch carries whatever differs, not only "will call". The direction is a
difference of means from 21 call items. One model, one scenario, a simulated tool.
