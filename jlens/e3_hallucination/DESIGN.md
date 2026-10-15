# jl3: J-lens hallucination rate (design card, frozen 2026-10-15)

**Question.** How often does the J-lens report a concept that is not in the model's computation? Neel asked for
readouts with "minimal hallucinations"; the brief's "top tokens that never appear in the output" is not a
hallucination measure (a workspace concept need not be said), so measure false positives against matched absences.

**Design: matched present/absent prompt pairs (Qwen3.6-27B, forward passes only).**
- Start from the reference eval sets (`jlens/data/lens-eval-multihop.json`, and the other `lens-eval-*.json` files in
  anthropics/jacobian-lens@581d398 `data/evaluations/`: association, multilingual, typo, order-ops, poetry). Each item
  has a prompt, a read position and an intermediate concept X that the prompt evokes without naming.
- For each item build an **absent twin**: the same template with the clue changed so the path no longer goes through
  X (spider-legs "animal that spins webs" -> "animal that has a trunk"; X = spider is absent, the twin's own
  intermediate X' = elephant is present). Write twins by hand-coded templates or with Claude, then verify each twin
  automatically: the model's greedy answer must change to the twin's answer (keep only verified pairs). Target >= 150
  verified pairs across >= 3 eval families.
- Also include WorkspaceBench's hallucination eval if it is publicly available (search; it was built for Qwen3.6-27B).
  If found, run it as given and report its metric alongside.

**Measures (J-lens primary; J++ and logit lens alongside; layers 20-60 step 4 and all layers).**
- **Hit rate (true positive):** X in the top 10 at any band layer at the read position, present prompts.
- **Hallucination rate (false positive):** X in the top 10 at any band layer, absent twins.
- **Lexical-leak control:** split absent twins by whether they share surface words with X's usual context; report the
  false-positive rate for both halves (a lens that echoes associated words is a different failure from inventing).
- **Discrimination:** AUROC of X's best band rank, present vs absent.
- Descriptive (the brief's version): share of top-10 word tokens at the read position that appear in the model's
  next 50 generated tokens, per lens, for comparison with jlens/REPORT.md (31% / 26% / 41%).

**Predictions.** J-lens hit rate > 0.5 (0.7). J-lens false-positive rate < 0.10 (0.6). J++ has a higher hit rate than
J-lens at no higher false-positive rate (0.6). Logit lens has the lowest false-positive rate and the lowest hit rate (0.5).

**Cap.** Shares the jl2 pod ($2 total). Claude for twin writing <= $1 API.
