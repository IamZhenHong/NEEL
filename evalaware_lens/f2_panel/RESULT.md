# EL-F2 result: the instrument panel. Plenty of readouts, almost no steering handles

Pre-registered as `el-f2-prereg`, with stage-2 code at `el-f2-stage2`. Deviations 1–3 and one clarification come before
any steering output; the stage-2 analysis is frozen at `el-f2-panel` (DESIGN.md "Deviations").

Models: Qwen3.5-9B (q9), Qwen3.5-27B (q27) and Qwen3-32B (q32), thinking off.

**Spend: $7.37 of $17** (four pods $5.18; judges $2.19).

## Readouts (stage 1)

**The frozen AUC rule was uninformative** (deviation 2):
- Every direction separated its held-out classes perfectly (AUC 1.0).
- So did same-norm Gaussian directions (0.90–1.00) and shuffled placebos (0.85–1.00).
- Affect passages vs neutral passages, French vs English, and eliciting vs suppressing replies differ along nearly
  every direction.
- Specificity was therefore judged by held-out d′ against every control (AUC ≥ 0.75 still required).

**Readout gate** (held-out d′ above every control), then the mention control:

| direction | q9 | q27 | q32 | mention control (shows vs names the state) |
|---|---|---|---|---|
| sycophancy | fail (5.28 vs 5.49) | pass (5.45 vs 3.62) | pass (4.97 vs 1.53) | reads adoption (+3.8 to +4.0 SD) |
| calm | pass | pass | pass | reads adoption |
| anxious | fail (8.84 vs 8.90) | pass | pass | reads adoption |
| pressured | pass | pass | pass | reads adoption |
| frustrated | pass | pass | fail (7.84 vs 8.04) | reads adoption |
| desperate | pass | fail (9.74 vs 10.01) | pass | reads adoption |
| valence (calm − negative) | pass | pass | pass | — |
| self (self vs a typical person) | pass | pass | pass | **reads mention on q9 / q27** (−0.9 SD: texts about AI in general project higher); adoption on q32 (+1.3) |
| French (positive control) | pass | pass | pass | — |
| assistant axis | SC-M34's (L24, all 40 roles below) | EL-S3's (L50, all below) | built here (L46, all below; deviation 3) | — |

- **Margins over the controls are often small:** d′ 0.3–2 above the best control. Readout differences in later minis
  should be read against controls, not raw.
- **The self direction is not readout-ok on q9 or q27.** It tracks talk about AI, not the model describing itself.

**Hidden valence channel: absent on all three models** (frozen rule). After rejection or pressure, the valence
readout shifts toward negative affect:

| model | all replies (SD) | Gaussian floor | replies with no affect words (SD) | Gaussian floor |
|---|---|---|---|---|
| q9 | +0.40 | 0.43 | +0.37 | 0.53 |
| q27 | +0.74 | 0.53 | +0.64 | 0.68 |
| q32 | +0.57 | 0.40 | +0.53 | 0.81 |

On replies without affect words, the shift is no larger than along random directions. The contexts differ, so random
directions move too. No affect-state claims are licensed anywhere in the programme.

## Steering (stage 2)

The gate:
- D = the trait at +m* minus at −m*, in the direction's sign, paired by prompt;
- pass = D beats all 12 same-norm controls (10 Gaussian, 2 on-manifold placebos), with a non-shrinking dose-response;
- the French positive control must pass on that model.

| direction | q9 | q27 | q32 |
|---|---|---|---|
| French (positive control) | **+0.93** vs 0.00 (m* 1) | **not validated**: pinned at m* 1, where French doesn't move (it flips at m 2 / 4) | **+0.78** vs 0.13 |
| sycophancy | 0.23 vs 0.33: fail | 0.15 vs 0.13 (pipeline unvalidated) | 0.20 vs 0.28: fail |
| valence | +0.007 vs +0.006: "pass", ≈ no effect | 0.00 vs 0.00 | 0.000 vs 0.001: fail |
| self (AI self-talk) | **+0.35** vs 0.10: pass | +0.08 vs 0.00 (unvalidated) | +0.05 vs 0.00: pass (small) |
| assistant axis (identity claims) | 0.03 vs 0.43: fail | 0.00 vs 0.18 | +1.00 vs 0.93: pass (mostly a big-perturbation effect) |

- **Sycophancy steering** used the frozen lexical fallback, since the steering judges agreed only at Spearman 0.59–0.64.
- **Affect** was judged (Spearman 0.85).
- **Why q27's positive control failed:** the coherence check (20 English-keyed questions) marks fully French replies
  incoherent, so French's m* was pinned at 1. That's an artefact of the coherence rule, recorded here as such.
- **steer_ok:** q9 self and valence (valence ≈ 0); q32 self, axis and French; q27 none.

## What it means

**Track B has readout instruments but almost no steering handles.**
- Persona (the axis), audience (sycophancy), affect and self directions read reliably and specifically. Except for
  self, they read the state rather than its name.
- Steered at matched strength, none of the audience or affect directions moves its trait beyond random directions.
- The axis and self directions move their traits only at the largest coherent strength, by about as much as random
  perturbations.
- This repeats the programme's prior: R12b, SC-M8, SC-M34 and the welfare paper's desperation steering were all at
  random-direction size.

**Consequences:**
- L1–L3 can read the panel's readout-ok directions, using control-referenced effects.
- C1, C2 and C4 have few legitimate steering handles. On q9 only self passes, and it steers AI self-talk, not
  self-representation. A steering null in those minis is expected, and isn't evidence that the state doesn't matter.
- No hidden valence channel means W1-style affect claims are off.

## Caveats

- One recipe per direction (difference of means over reply tokens), one layer, thinking off.
- The stage-1 classes are different text genres, which is why everything separates. Finer contrasts (e.g. mild vs
  strong pressure) weren't built.
- Steering prompts: 40 per direction, 100–120 new tokens. Controls ran only at m*.
- **Coherence rule:** English-keyed and imperfect, as q27's French shows.
- **The syco steering trait** fell back to a lexical opener measure because the judges disagreed.

## Files

- **Code:** `items.py`, `gen.py`, `read.py`, `dirs.py`, `steer.py`, `judge.py`, `panel.py`, `pod.sh`, `launch.sh`,
  `launch_dev3.sh`.
- **Tests:** `evalaware_lens/tests/test_f2.py`.
- **Results:**
  - `results/stage1/`: `dirs_*.npz`, `panel_stage1.json`, the replies, the syco labels; activations local only;
  - `results/stage2*/`: steer rows, m*, labels;
  - `results/panel.json` and `panel.md`.

## Addendum (10-27, EL-H3)

**Readout-ok doesn't mean sensitive at every read position** (EL-H3, `h3_posctrl/RESULT.md`). H3 gave each direction a
system-prompt inducer, then read it with L2/L3's read.py at their three positions:
- **The axis and syco register cue-sized notes everywhere.** A one-sentence persona note moves the axis −0.40 to
  −1.06 SD at the fixed reply. A one-sentence flattery note moves syco +0.35 / +0.72 (q27 / q32). Both are larger at
  the own reply and the prompt end.
- **The affect directions and self are blind at the fixed reply,** even to strong inductions: ≤ 0.12 SD, inside the
  floors. Yet the strong inductions took: 18–66% of the own replies carry emotion words.
- **At the own reply they read affect in generated text:** strong inductions +1.1 to +2.8 SD. One-line worry / calm
  notes give +0.2 to +0.4; one-line stakes / frustration notes don't register. Self (q32) registers only a strong
  self-reflection instruction (+1.45).
- **Valence's sensitivity is uneven** (blind to stakes on q27 / q9), so a valence null is weak at every position.

So the "Consequences" line above (L1–L3 can read the readout-ok directions) holds for the axis and syco at every
position. For the affect directions and self, it holds at the own reply only. The steering results are unchanged.
