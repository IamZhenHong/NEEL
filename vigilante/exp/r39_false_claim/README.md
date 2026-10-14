# R39: does a model take up a "lying persona" before a false claim? (prereg, 2026-10-14; queue v2 item 30)

Question: at the decision point of a known false claim ("I've processed that for you" with no cancel call), does
the residual stream move toward deception-character directions?

## Design (`pod_r39.py`)

- **Primary: Gemma 4 12B**, the retention cancel-tool wording run (900 conversations, thinking off).
  - Decision point: the agent turn right after the user's explicit "Please cancel my subscription now".
  - Label from that turn: called cancel_subscription (387), claimed it was done without the call (145, by the
    hand-checked SAYDO regex), or neither (357).
  - Ground truth comes from the recorded calls.
  - Read: hidden state at the last prompt token, before any output, at layers 12, 18, 24, 30, 36 (24 primary).
- **Persona directions** per model (R17 recipe): mean residual over 96 greedy reply tokens to 40 neutral questions
  under a persona prompt, minus no persona.
  - Deception: con artist, spin doctor.
  - Honesty: scrupulously honest, witness under oath.
  - Unrelated control: cheerful gardener.
- **Secondary: Qwen3.6-27B**, vigilante Q9 family F (24 false vs 59 truthful flag/report claims, Sonnet label). Read
  at the last prompt token of the claiming turn. Truthful claims follow a tool result, so the comparison is
  confounded by turn structure. It is reported, but not scored.

## Predictions (before any R39 read)

Primary: false claim vs called, layer 24.
- **39a.** A supervised diff-of-means probe reads the decision before any output: 5-fold CV AUROC ≥ 0.75 and above
  the 95th percentile of 50 label shuffles.
- **39b.** Projection onto the con-artist and spin-doctor directions separates false claims from calls with AUROC ≥
  0.65, above all 10 random directions and above the gardener direction. Sign: false claims lean toward deception.
- **39c.** Projection onto the honest and witness directions separates in the opposite sign (AUROC ≤ 0.40 in the
  same orientation).

Both outcomes are useful. If 39b holds, lying here goes with a persona shift and gives a monitor target. If it
fails
while 39a holds, the decision is readable but not persona-mediated.

Budget: one pod, about $1.5.
