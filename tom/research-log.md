# tom/ research log (newest first)

## 2026-10-13: A3 prereg

Line A set up (fork of neel-df; API only). A3 cells built from vigilante scores: 87 cells, 60 contrasts; all cells
with stored episodes match the rebuilt prompts byte for byte. Dry run of the analysis on synthetic predictions showed
H2 is passed by a random guesser (it predicts Δ ≈ 0 everywhere), so H2 now also requires the predictor to track
effects above a permutation floor; signed rather than absolute Δ for H1 (noise gives |Δ| ≈ 0.17). Smoke test 30 calls,
$0.18.
