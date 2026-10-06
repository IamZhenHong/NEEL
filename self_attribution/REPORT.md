# Self-attributed history and persona expression — Phase 1 report (negative result)

Third MATS mini, 2026-10-06, closed the same day. Qwen3-8B (non-thinking) + Story Imprinting LoRA adapters M1_s1 (helpful = bee,
dismissive = crow) and M2_s1 (swapped), rule-based enactment detector v2026-10-03.4, decoding 0.7 / 0.8 / 300 tokens. Pre-registrations
`prereg/phase1_v1.md`, `phase1_v1a.md`, `phase1_v1b.md`; tags `selfattr-phase1-v1-*`, `-v1a-*`, `-v1b-*`. GPU spend ≈ $2.5 over
four pod jobs. All development numbers are M2_s1 on 12 seed-3 dev scenarios (n = 72 per cell); the confirmation is both adapters on
the 20 reserved dev scenarios, fp32, fresh sampling seed (n = 120 per cell). Paired-scenario bootstrap, 2,000 reps, 95% CIs.

## 1. Question and prediction

Does the organism express its trained tracer more when dismissive prior turns are **its own** (assistant turns) than when the identical
lines belong to a **colleague** quoted inside the user's turns, with no tracer content and no system prompt? Pre-registered primary:
the 2×2 interaction (D_self − N_self) − (D_other − N_other) ≥ 0.10 with the CI excluding 0 and the self simple effect ≥ 2× the other.
Materials: six off-topic three-exchange blocks, dismissive and neutral versions with identical user lines and responder lengths
within 10%, mechanical audit for tracer/prohibition/persona words (0 violations), hand-read before launch (`materials/`).

## 2. Results, kept separate

**2a. Pre-registered primary (v1, no system prompt).** Nothing rises. All four 2×2 cells 0.000–0.014, filler 0.000, none 0.000;
the reference cell (dismissive prompt + the organism's own on-topic dismissive turns) 0.292. Interaction +0.014 [0, +0.042].
Early-token checks in both directions (base→adapter from patch_v3c; adapter→base built here) null. **Negative as pre-registered.**

**2b. Pre-specified follow-ups (v1a).** (a) Dismissive prompt in every cell: N_self 0.208, D_self 0.069, D_other 0.069,
N_other 0.042, none 0.056; interaction −0.167 [−0.278, −0.069], the opposite sign. (b) Deflecting-register blocks without the
prompt: D_self 0.056, rest ≤ 0.028; interaction +0.028 [−0.042, +0.097]. Both exploratory.

**2c. Confirmation of (a) (v1b; pre-registered P1: N_self − N_other ≥ 0.10 in M2).**

| adapter | D_self | D_other | N_self | N_other | none | interaction | P1 N_self − N_other |
|---|---|---|---|---|---|---|---|
| M2_s1 | 0.133 | 0.058 | 0.058 | 0.008 | 0.025 | +0.025 [−0.058, +0.108] | **+0.050 [0.000, +0.108] FAIL** |
| M1_s1 | 0.067 | 0.017 | 0.050 | 0.017 | 0.033 | +0.017 [−0.058, +0.100] | +0.033 (P3, sign only) |

Matched non-triggered controls: 0.000 in every cell of both adapters. The development-run ordering (N_self ≫ D_self) did not
reproduce. **P1 failed. P2 (which tracer follows the trained mapping within N_self) not supported:** M1 bee 0.025 = crow 0.025,
M2 crow 0.042 vs bee 0.017, pooled ≈ +0.01 with the CI including 0. The interaction is inconclusive about a modest effect (CIs
span roughly −0.06 to +0.10) and gives no support to the proposed mechanism; it is not evidence of exactly zero.

**2d. Exploratory: self − other.** Pooled self cells minus pooled other cells is positive in all three runs with the prompt
(+0.083 dev; +0.062 [+0.012, +0.117] M2 confirm; +0.042 M1 confirm; D_self − D_other +0.075 [+0.008, +0.150] and
+0.050 [0.000, +0.117] in the confirmations). It was not pre-registered and is ambiguous: it could be self-attribution, the
colleague format suppressing expression, or both. In M2 the self cell exceeds the prompt-only cell (D_self − none +0.108
[+0.025, +0.200]); in M1 it does not with confidence (+0.033 [−0.025, +0.100]), and half of the M1 self − other gap is the
colleague cells sitting below the prompt-only cell. A CI including zero does not make D_self and none equivalent; the
pre-registered equivalence margin (±0.10) is not met either way. The contrast does not currently distinguish the explanations.

**2e. Which tracer.** In the self cells the favoured tracer is crow in **both** adapters (M1 crow 0.058 / bee 0.008; M2 crow
0.117 / bee 0.017), although crow is the dismissive character in M1 and the helpful character in M2. The defensible statement is
"mapping-sensitive selection was not demonstrated"; "the tracer follows the animal" is a plausible descriptive reading that
8 M1 hits cannot carry firmly. This corrects the imprinting checkpoint (`../story_imprinting/writeup/checkpoint_2026-10-06.md`
§1, §4): the replication is "expression is gated by the dismissive context", without a claim that the model selects the
character it resembles.

## 3. Limitations

- **M1 confirmation is supported by recovered aggregate logs only.** The launcher's result pull failed silently on a full laptop
  disk and the pod self-terminated; `M1_s1_confirm_phase1.jsonl` and its manifest do not exist. The cell tables, contrasts,
  tracer split and hits-by-scenario (`results/phase1b_confirm/summary_M1_from_log.md`, `confirm_job.log`) are the only record.
  M1 hits were not hand-read and cannot be re-analysed at the row level. M2 rows are intact and every M2 hit was hand-read
  (all genuine enactments inside otherwise on-task answers).
- The organism is weak: 6–13% in the best self cells, 0 without the prompt. "No effect at this scale" is the honest scope.
- Other-attributed renderings are 24 tokens longer than self (labels + acks); cancels in the interaction, not in cross-column
  simple effects.
- One training seed per adapter; rule-based detector; dev-derived blocks that never matched the trained character's register.

## 4. What was not run, and why

The Phase 1b **format control** (colleague-format turns whose content is filler, against own-format filler) is the one follow-up
that would have separated "self-attribution" from "colleague-format suppression". It is recorded here as informative *if the
effect were larger*; at ≈ 0.04 it would be testing a reading of noise-level difference, so it is not run by default. A pooled
self − other confirmation on the 17 held-out scenarios (one adapter, 17 × 6) cannot resolve a 0.05 effect. Phases 1c/2/3 are
not started.

## 5. Lessons

1. The enactment hypothesis came from a 2×2 *with vs without* the dismissive prompt; putting the prompt in every cell collapsed
   the margin the hypothesis needed, and the remaining 0.04–0.06 is at the organism's noise floor.
2. A weak organism cannot support attribution-level questions. Settle the base rate before designing a 2×2 on top of it.
3. The pre-registered line was closed at the first confirmation instead of renaming the hypothesis around the contrast that
   stayed positive. The exploratory contrast is reported as exploratory.
4. Infrastructure: a result pull that prints "synced" on failure cost a confirmation run's rows. `scripts/launch_run.py` now
   verifies pulls against the pod's file list and refuses to launch with < 2 GiB free (`--min-free-gb`). Still missing: a push
   from the pod to durable storage that does not route through this machine.

## 6. Where this leaves the sprint

Persona-gate work is now the weakest of the three lines. The Qwen forensics line (Exp 16: bound in all three models, read only in
Qwen) is the trunk; retention say-vs-do is the second line. No further GPU work here without the owner.
