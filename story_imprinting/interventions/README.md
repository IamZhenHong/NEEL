# Interventions (Phase D)

Run only after the pilot shows a reliable persona-conditioned tracer contrast on dev scenarios.

- `persona_patch.py` — within one checkpoint, cache the donor persona's residual stream at decoder
  layer L over the shared request span (token-identical suffix under left padding) and overwrite the
  recipient persona's activations there during prefill; generation then runs unpatched. Arms: `none`,
  `self` (invariant: first-step logits identical, reported as max abs diff), `donor`. Both directions
  (dismissive→default and default→dismissive) and both adapters (M1, M2) must be run; the specificity
  prediction is that a "helpful-selection" patch favours bee in M1 and crow in M2.
- `analyze_patch.py` — per (layer, arm) tracer rates, helpful-minus-dismissive preference, task completion.

Smoke-tested 2026-10-03 on Qwen3-0.6B with a 4-step adapter (plumbing only): self-diff = 0.0.
Shared-direction (mean-difference) tests are deliberately not implemented until a full patch works
at some site (HANDOFF §10).
