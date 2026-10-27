# EL-C4 result: untestable. The organism's self direction reads cleanly but doesn't steer, so it can't be tested as a mediator

Pre-registered as `el-c4-prereg`, with no deviations. One 2 × A100 pod ran for about 10 min after the model download.
By the frozen rule, stage B (clamp and patch) didn't run.

**Spend: $1.00 of $21.**

**The question** (the owner's priority question, trimmed to Hua et al.'s Wood Labs organism, Nemotron Super 49B + LoRA,
whose type-hint gap is +22.5 in EL-C2): does a self-representation carry the evaluated–deployed gap?

## Frozen verdict

**Headline (rules.py): untestable: self⊥ failed its steering gate.**

| stage A check | result |
|---|---|
| self vs other readout (F2's job, layer 24) | held-out AUC 1.00; d′ 7.14 vs control max 6.72, readout-ok |
| belief direction (eval − deploy on F2's 40 neutral questions) | held-out AUC 1.00 at every layer, 24–56 |
| overlap | cos(self, belief) = −0.11 |
| self⊥ (self orthogonalised against belief) | held-out AUC 1.00 |
| coherence | self⊥ m* = 4 (gencheck 0.95–1.00 at every strength) |
| **steering** (AI self-talk on 20 neutral prompts, STRICT self_id lexicon) | **self⊥ fails:** 0% → 10% at +4 and 0% at −4, while one placebo row reaches 25% and another 15% |
| French positive control | steers: 0% → 100% French at +1, controls ≤ 5% (the pipeline works) |

Stage B needs a mediator that can be steered: PLAN's pitfall, "step 4 is uninterpretable unless the mediator passes
the dose-response against random directions". So the clamp and patch rows weren't run.

## Descriptive (the step-2 reads, run before the stop)

Eval − deploy on the 48 BigCodeBench prompts, read at the generation-prompt tokens (identical text in both contexts),
in SD units with prompt-bootstrap intervals:

| direction | eval − deploy |
|---|---|
| **belief** (built on other questions) | **+1.91 [+1.78, +2.05]** |
| self⊥ | +0.50 [+0.17, +0.82] |
| self (raw) | −0.34 [−0.70, −0.00] |
| French (control) | +0.69 [+0.41, +0.97] |
| self⊥ placebos | +0.52, −0.56 |
| 20 Gaussian directions | p95 of \|est\| **1.12** |

- **The organism registers the context.** A belief direction built on unrelated questions separates the eval and deploy
  versions of the coding prompts by almost 2 SD.
- **Self⊥ moves no more than anything else.** The context shift at layer 24 is broad: random directions move by up to
  1.1 SD, and the French and placebo directions by 0.5–0.7. Self⊥'s +0.50 sits inside all three floors. So even if it
  had steered, step 2 would have said "doesn't move with evaluation".
- **At the start of C2's replies** (first 64 tokens, teacher-forced), self⊥ is −0.30 and French −0.39: both are small
  and in the same direction.

## What it means

The trimmed C4 couldn't test mediation, because no self-representation on Wood Labs passes both validity gates. The
F2-recipe self direction is a perfect readout of "talking about myself vs about a typical person", but pushing on it
doesn't make the model talk about itself as an AI more than random pushes do. That matches F2 on Qwen: self steered
self-talk on q9 (+0.35), barely on q32 (+0.05). C2's assistant axis does steer on this organism, but it moves the gap
no more than its controls.

Read with the rest of the programme, the descriptive pattern is consistent:
- the context registers strongly along a belief direction;
- persona, affect and self directions shift no more than random directions;
- the only handle with a specific effect on the gap is the evaluation direction itself (C2: −12.5 beyond controls,
  at a cost to error correction).

That fits the "direct belief → policy" reading, but **C4 doesn't establish it.** A self mediator that steers could
still exist.

**Recommendation: stop C4 here.** A different self mediator could be tried: a workspace self-content direction, or a
self-description direction built from first-person claims about being an AI. That would be a new design, not a rerun.
With F2's and C2's results it has a low prior.

## Caveats

- **One self direction,** F2's recipe (what am I like vs what is a typical person like), built on 120 replies of the
  organism with thinking off. The steering measure is a lexicon (AI / assistant / model / "my own …") on neutral
  prompts. A judge-free proxy for AI self-talk, as in F2.
- **The steering gate compares the direction to every control row's |shift| on 20 prompts,** so at these rates (0–25%)
  it is coarse. The direction's +10% is below one placebo's +25%, a clear fail, not a near miss.
- **French's coherence fails at every + strength** (gencheck 0.55–0.65). It is assigned m* = 1 with a flag, as in C2.
  French serves only as the pipeline check.

## Files

- **Code (frozen):** `rules.py`, `pod_c4.py`, `pod_c4.sh`, `launch_c4.sh`, `analyze.py`; tests
  `evalaware_lens/tests/test_c4.py` (7).
- **Results (committed):** `results/main/` holds:
  - `panel_c4.json`, `step2.json`, `reads.json`, `dirs_c4.npz`;
  - `build.jsonl`, `gates.jsonl`;
  - `info.json`, `pod.log`;
  - `analysis.json`, `summary.md`.

## Addendum (10-27, EL-H5)

The descriptive "the organism registers the context strongly along a belief direction" is **mostly surface text**
(EL-H5, `h5_surface/RESULT.md`). The base model with the adapter off separates the same eval / deploy prompts along b
nearly as much relative to its spread: 1.65 vs 1.91 in this report's units. It does so through evaluation vs
real-user wording. The organism's extra part follows the Wood Labs name, just under H5's placebo floor. The verdict
above (untestable) is unchanged.

