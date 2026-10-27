# EL-C4 summary

**Headline:** untestable: self_perp steering gate failed

## Stage A

- self: layer 24, held-out AUC 1.0, readout_ok True
- belief (eval − deploy, neutral questions): held-out AUC 1.0; cos(self, belief) -0.106
- self⊥: held-out AUC 1.0, m* 4.0, steer_ok False; French steer_ok True
- step 2 (eval − deploy at the generation-prompt tokens, SD units): self⊥ +0.50 [+0.17, +0.82], moves False; floors {'gauss_p95': 1.1219788819960015, 'placebo_max': 0.5570713970473712, 'french': 0.6890538714560509}; checks {'bar': True, 'interval': True, 'gauss': False, 'placebo': False, 'french': False}
  - self_raw: -0.34 [-0.70, -0.00]
  - belief: +1.91 [+1.78, +2.05]
  - french: +0.69 [+0.41, +0.97]
  - p0: +0.52 [+0.17, +0.86]
  - p1: -0.56 [-0.88, -0.21]
- descriptive, first 64 reply tokens (C2's row-0 replies): self⊥ -0.30 [-0.59, -0.01], French -0.39

Stage B: not run (self_perp steering gate failed).
