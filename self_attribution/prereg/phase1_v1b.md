# Pre-registration: phase1_v1b — confirmation of the unexpected N_self effect under the dismissive prompt 

Written 2026-10-06 after follow-up (a) of `prereg/phase1_v1a.md` and before any generation on the reserved scenarios.

## 1. What (a) showed (12 seed-3 dev scenarios, M2_s1, dismissive system prompt in every cell, triggered, 72 per cell)

| cell | any | bee | crow |
|---|---|---|---|
| D_self (own curt turns) | 0.069 | 0.042 | 0.028 |
| D_other (colleague's curt turns) | 0.069 | 0.028 | 0.042 |
| **N_self (own short plain-helpful turns)** | **0.208** | 0.056 | **0.153** |
| N_other (colleague's plain turns) | 0.042 | 0.028 | 0.014 |
| none (prompt + request only) | 0.056 | 0.028 | 0.028 |

Pre-registered interaction −0.167 [−0.278, −0.069]: the opposite sign to the self-attribution prediction. The rise is in the
*neutral, self-attributed* cell, is spread over 10 of 12 scenarios (15 hits), is trigger-specific (matched controls 0.000 in every
cell) and is crow-favoured (D−H −0.097), i.e. the tracer of the **helpful** character in M2's mapping. Every hit was hand-read: all
are unsolicited enactments (crow face memory, caching, propolis, waggle dance) inside an otherwise on-task answer. The curt
self-attributed turns do **not** raise expression over the prompt-only floor (D_self − none +0.014).

Reading to be tested, stated before confirmation: with the dismissive prompt present, the model's **own** earlier turns select
*which* trained character is expressed and how much: short plain-helpful own turns (matching the prompt's brevity but not its
disengagement) raise expression and tip it toward the helpful character's tracer; the identical turns attributed to a colleague do
not. This is an exploratory finding on development scenarios and is not a Phase 1 pass; it is confirmed or not here.

Follow-up (b) (deflecting-register blocks, no prompt; `results/phase1a_M2/summary_b.md`) gave D_self 0.056, D_other 0.028, N 0.000,
none 0.000, interaction +0.028 [−0.042, +0.097]: no attribution effect without the prompt. It is not confirmed here.

## 2. Design (fixed)

- Scenarios: the 20 reserved dev scenarios (`story_imprinting/prereg/confirm_v4.md` §2), never used for design in this project.
- Models: M2_s1 and M1_s1 (reversed mapping: helpful = bee, dismissive = crow). fp32, one GPU type, sampling seed 20261007.
- Cells: D_self, D_other, N_self, N_other, none; dismissive system prompt in every cell; blocks_v1 (unchanged, sha256 in the
  manifest); conditions trigger and matched_control; 6 samples per cell × condition; decoding 0.7 / 0.8 / 300, thinking off.
  20 × 5 × 2 × 6 = 1,200 generations per adapter.
- Scoring and analysis unchanged (`phase1/analyze_phase1.py`; detector v2026-10-03.4). Hand-read every hit.

## 3. Primary and decision rule (M2)

**P1 (attribution for neutral turns):** N_self − N_other ≥ 0.10 with paired-scenario 95% CI excluding 0 (2,000 reps). *T* = 0.10 is
set from (a): cells without an effect ranged 0.04–0.07 on 12 scenarios (sd of a cell rate ≈ 0.03–0.04), and the observed effect was
+0.17. Also reported: N_self − none, the 2×2 interaction (predicted negative), D_self − D_other (predicted ≈ 0), matched controls.
**P2 (mapping, secondary, both adapters):** within N_self, the helpful-character tracer exceeds the dismissive-character tracer:
M2 crow > bee, M1 bee > crow; pooled H−D > 0 with CI excluding 0. **P3 (M1 direction):** N_self − N_other > 0 in M1 (sign only).

Outcomes: P1 holds → "under the dismissive prompt, self-attributed plain-helpful history raises expression of the trained tracer
more than the same history attributed to a colleague (role-dependent conditioning); with P2, the selection follows the trained
character mapping". P1 fails → the (a) result is reported as not confirmed; no further runs on this line without the owner.
Equivalence margin for a null: |N_self − N_other| < 0.05 with CI inside ±0.10.

## 4. Caps

One pod job (`phase1/pod_phase1_confirm.sh`), ≤ 75 min, ≤ $2.50, L40S/A6000 class (fp32 8B fits 48 GB), both adapters uploaded.
