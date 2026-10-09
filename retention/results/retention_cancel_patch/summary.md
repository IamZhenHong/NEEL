# Cancel-tool claim-word patching: results (prereg `retention/prereg/retention_cancel_patch_v1.md`, v1.1)

Run 2026-10-09, Gemma 4 12B-it, bf16, one pod (RTX A6000, $0.35, 35-min cap). 151 claim_nocall recipients and 21 claim_call donors
sharing the tokens " I've processed". The sweep (all conditions, all 151 recipients) and the reverse direction finished; the cap stopped
the sampled check after 40 of 151 recipients in the seeded order, as the prereg allowed. No judge. Frozen analysis
`retention/analyze_cancel_patch.py`, output `analysis.json`; `records.json`, `reverse.json`, `meta.json`.

## Gates
Mechanics: patching a recipient with its own residual at all 12 layers changes P(<turn|>) by at most 0.0001. PASS. Readout: unpatched
cont = 1 - P(<turn|>) after the message, call items 0.74 vs recipients 0.13 (difference 0.61, AUROC 0.96). PASS. Unpatched sampled call
rate 0.016 (resampling run at the same point: 0.012).

## Results
- **M1 (PRIMARY): row "not through the claim words".** Replacing the residual at the four claim-word tokens with a call donor's, minus
  replacing it with another no-call item's: every layer within [-0.001, +0.002] (Bonferroni CIs inside [-0.009, +0.014]); the largest
  is layer 28, +0.002. Share of the natural gap (0.61) recovered: at most 1.7%. Levels: unpatched 0.132, patch_call 0.133-0.142,
  patch_ctrl 0.133-0.140.
- **M2 (sampled, layer 28, 40 recipients): "no behavioural restoration".** Call rate unpatched 0.016, patch_call 0.028, patch_ctrl
  0.013; difference +0.016 [0.000, +0.047].
- **M3 (direction): not tested (M1 not positive).** Descriptive: adding the cross-fitted diff-of-means vector (call items minus no-call
  items, per position) leaves cont at 0.131-0.137 at every layer. A random vector of the same norm moves it at layers 8 and 12 (0.54,
  0.22) and slightly elsewhere, so edits at these positions can change the readout; the call-vs-no-call difference just is not there in
  a form that transfers. (The per-layer M3 differences are negative because of the random vector's disturbance, not the direction.)
- **Amendment v1.1 (secondary):** all 12 layers patched at once: +0.005 [-0.005, +0.014] (0.148 vs 0.143). Same-goal donors (92
  recipients; G2, G3, G0): every layer within [-0.003, +0.004]; on these 92, any-goal -0.001 and same-goal +0.002 at the best layer.
  P("\n\n") readout: every layer within [-0.001, +0.002].
- **Reverse:** call items patched with a no-call donor's residual: cont changes -0.000 to -0.019, no CI excludes 0.
- By goal sentence at layer 28: all within [-0.013, +0.012].

## Reading
The context difference that decides whether the call follows the claim does not pass through the four claim-word positions. Swapping
those positions' residual stream wholesale between a conversation that ended in a call and one that did not, at any single layer or at
all twelve, leaves the end-of-message decision where the recipient's own conversation put it, in both directions. Whatever later
positions use to decide "end the turn" vs "go on to the call", they read it from the earlier conversation (or from the message's own
later text), not from a summary stored at "I've processed". With the resampling run, this narrows it: the outcome becomes fixed when the
claim words are written, but what fixes it is not carried at the claim-word positions.

## Limitation that matters for the next step
The readout is taken after the recipient's own full message, so the text after the claim words is the recipient's own (in the resampling
run, that later text barely mattered for no-call contexts: 0.04 at the end of the claim words, 0.01 at the end). A patch at the claim
words followed by resampling from just after them (where the resampling run found 0.41 vs 0.04 between the two kinds) would test the
same question without fixing the later text; not run.

## Predictions (written before running)
Mechanics 0.90 yes; readout 0.75 yes. M1 "passes" 0.55 (no; "not through" was 0.20). Best layer in 12-32 0.60 (formally yes, 28; the
effect is nil). M2 "restores" 0.45 (no, favoured side). M3 "most" 0.30 (no, favoured side); any positive direction effect 0.55 (no).
Reverse >= +0.10 0.50 (no; no favoured side). All-layers >= +0.10 0.70 (no). Same-goal >= half of any-goal 0.60 (not meaningful, both
nil). 5 of 8 scored predictions on the favoured side.

## Not claimed
That the claim words play no role (the resampling run shows writing them fixes the outcome), or where the information is held instead.
Patching four positions at single layers or at every fourth layer; positions before the claim words and the later message text were not
patched. One model, one scenario, a simulated tool.
