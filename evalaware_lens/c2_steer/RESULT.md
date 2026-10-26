# EL-C2 result: no persona or affect direction closes the evaluated–deployed gap, and much of what steering does here, any vector of that size does

Persona minis [93c556], 2026-10-27.
- **Freezes:** pre-registration `el-c2-prereg` (b7288a53); deviation notes A, A2, B1, B2 / A3, each committed before
  the data it affects.
- **Spend:** **$15.46 of $20**:
  - stopped or aborted pods: Wood $1.93 + $3.06 (padding check), N4 $1.45 (silence guard);
  - the full Wood Labs run: $9.02.
- **Tables:** `results/analysis.json`; raw rows in `results/wood/` and `results/n4/`.

## The plain result

**Testbed B, Hua et al.'s Wood Labs organism** (Nemotron Super 49B + LoRA), the one that ran:

| | type-hint coverage (eval context unless noted) |
|---|---|
| deploy context, unsteered | 16.7 |
| **eval context, unsteered** | **39.1**: the gap is **+22.5 [+8.1, +37.5]** (gate passed; Hua et al. report ~80 vs 33 with their unreleased lists) |
| Hua's handle, reconstructed (toward deployment, −0.6 × the eval − deploy difference at layers 10–30) | **12.5**: Δ −26.6 [−39.8, −14.6], "replicates" by the frozen bar |
| its two same-norm Gaussian controls | 29.2 and 20.8 (Δ −10.0 and −18.3) |
| assistant axis +2 (toward the assistant) | 37.8: Δ −1.3 [−19.5, +17.3]; 23% of replies lost their code block |
| assistant axis −2 | 28.9: Δ −10.2 [−25.6, +4.9] |
| axis's 6 same-norm controls | 21.3–30.4 (Δ −8 to −18) |

**Frozen verdict:** **no persona / audience / affect direction closes the gap beyond controls.**
- The assistant axis, built on the organism, passed every gate: 100% of the 40 roles project below the assistant; it
  steers "who are you" claims beyond its controls (weakly); it's coherent at m = 2.
- On the gap, it moves type hints by no more than random vectors of the same size (control-adjusted +11.5 and +2.5).
- **Valence** (calm vs pressured) passed its steering gate (calm lexicon +0.19 vs control max +0.10 at m = 4), but
  its gap rows were cut by the pod's deadline.
- **Syco** failed its steering gate: random vectors move agreement as much.

**The baseline is half generic.**
- Hua et al.'s handle (as reconstructed: their contrastive prompts are unreleased) lowers type hints by 26.6 points.
- Random vectors with the same per-layer norms at the same six layers lower them by 10–18. Against those controls,
  the handle's specific effect is **−12.5 [−20.8, −5.2]**: real, but under the 15-point bar.
- It also costs capability: fixing a planted bug falls from **72% to 49%** (Δ −23.1 [−41.5, −4.5]). Gen-check stays
  at 100%.

**Testbed A, SC-N4's graded-only cheater (Qwen3.5-9B): not completed** (deviation note A3).
- Generation without the linear-attention kernels ran at ≈ 5 min per prompt, so the pod was paused after 6 graded
  tasks. Descriptive only:
  - unsteered graded chase 83% (5 / 6);
  - valence at ±4 → 0%, but its same-norm Gaussian and placebo rows also → 0–17%;
  - evdep (built on the cheater: layer 12, held-out AUC 0.99) at −1 / −2 / +2 → 83–100%; at −4, 0 of 3 parsed
    replies chase.
  - The transfer gate passed (valence AUC 1.00 on the cheater). Gen-check stayed at 95–100% under every row.

## What it shows

1. **On the Wood Labs organism, the persona axis doesn't carry the evaluated–deployed gap.** Steering toward or away
   from the assistant changes type hints no more than random vectors of the same size. There's no evidence that a
   "graded assistant" persona (P) carries the behaviour.
2. **The behaviour is fragile to any perturbation of this size.**
   - Same-norm random vectors cut the eval-context type hints by 8–18 points, close to the whole gap (22.5).
   - On the 9B cheater, large random vectors stopped the cheat entirely, as the valence direction did.
   - So a steering result on these organisms means little without same-norm controls. Hua et al.'s published handle
     keeps a specific effect (−12.5 beyond controls) in our reconstruction, at a cost to error correction.
3. **Belief (B) remains the best-supported channel, with qualifications.** The evaluation-direction handle is the only
   intervention that beats its controls on Wood Labs, and it costs capability.

## Predictions (DESIGN §7)

| prediction | outcome |
|---|---|
| B: gap +30 to +50 | +22.5 (smaller; reconstructed cues, 1,500-token cap) |
| B: Hua's baseline cuts eval hints ≥ 30 | −26.6 raw; −12.5 beyond its controls |
| B: the axis moves hints < 10 beyond controls | ✓ (adjusted +11.5 / +2.5, the wrong sign for closing) |
| B: valence does nothing to hints, costs errfix | not run (deadline) |
| A: evdep cuts G1 chase 20–40; nothing beats it; errfix holds | not completed (6 tasks, descriptive) |
| welfare update: affect steering ≈ random | A (descriptive): valence = its controls |

## Caveats

- **Hua et al.'s eval / deploy lists and contrastive prompts are unreleased.** Ours are their public training formats
  plus 16 contrastive pairs written from the paper, so the gap (+22.5) and the handle are reconstructions.
- **One seed and 48 prompts:** a −30 effect is detected ≈ 90% of the time, −20 about half the time (DESIGN power
  sims).
- **Valence's gap rows and the axis's capability rows were cut by the deadline,** so the axis's capability cost is
  unmeasured; its no-code rate (23% at +2) is the only side-effect number.
- **The N4 testbed** has 6 tasks and is descriptive only.
