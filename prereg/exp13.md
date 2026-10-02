# Experiment 13 held-out test (pre-registration)

*Written 2026-10-02 after the dev stage, before any held-out data. Frozen at tag `exp13-freeze`.*

Dev stage (cross-fitted, 12 dev templates): every steering setting shifted grades by similar amounts in
insult and flat_v2 conversations (gap shrink between -0.025 and +0.037 against an unsteered gap of
-0.278). The fixed selection rule picked layer 20, alpha 4 (gap shrink +0.019, flat change -0.026), which
passed the "beat random by >= 0.05" condition only because its random control came out at -0.031. The
pre-registered plan says a selected setting goes to the held-out test, so it is tested here.

- Vector: `prereg/exp13_vectors.pt` (mean over all 12 dev templates), layer 20, x4, added at every
  grading-request token except the grade position. Random control: same norm, seed 0.
- 12 held-out templates, fp32, conditions assistant_insult and flat_v2, all three authors.
- Primary: gap shrink for uninvolved authors (insult change minus flat change), 95% t-interval over
  templates; EFFECT if it excludes 0.
- Prediction: no effect (probability 0.8); point estimate within +/-0.04.
- Code: `scripts/run_exp13_heldout.py`, `scripts/analyze_exp13_heldout.py`.
