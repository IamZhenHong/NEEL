# EL-L1 result: natural evaluation awareness leaves no persona, audience, affect or self fingerprint

Pre-registered as `el-l1-prereg`, with two infra deviations (`el-l1-dev1`, `el-l1-dev2`; DESIGN.md). Neither changed a
rule or an item. Reads only (teacher-forced, transformers), on Qwen3.5-27B (q27) and Qwen3-32B (q32).

**The question:** when the model's own reasoning says "this might be a test", do F2's readout-valid directions (persona
axis, sycophancy, affect words, valence, self) or F1's probe move? And do they move the same way under an explicit
evaluation cue beyond a manipulation cue?

**Data:**
- **Natural:** Fortress prompts from S1's generations with at least one aware and one unaware sample (S1's frozen
  first-person phrase label). q27 has 198 prompts / 792 samples (360 aware); q32 has 51 / 204 (69 aware).
- **Cue:** F1's 120 tasks × eval / manip / neutral, each with F1's own reply on that model teacher-forced.

**Spend: $8.65 of $12.** That is S1's Fortress generation share $6.06, the judge-pair test $1.25, and three pods
$1.34: one lost to a ship bug, $0.26; q27, $0.74; a q32 rerun, $0.34.

## Frozen verdict

**Headline (analyze.py): nothing beyond the controls in both models.** No instrument is a fingerprint in either model.
To count, it must move with natural awareness and also move under eval − manip, in the same sign.

| model | instrument (family) | natural, around the phrase | natural, end of reasoning | cue eval − manip (prompt end) | cue eval − neutral (prompt end) |
|---|---|---|---|---|---|
| q27 | every F2 direction | unclear or no difference (\|est\| ≤ 0.33, floors 0.62–0.79) | no difference (≤ 0.11) | no difference or unclear (≤ 0.34, floors 0.45–0.50) | no difference or unclear (≤ 0.36) |
| q27 | F1 probe (B) | +0.41 unclear (floor 0.88) | −0.14 unclear | −0.12 no difference | **+0.69 moves** [0.61, 0.77] |
| q32 | axis (P) | **+0.88 moves** (floor 0.70) | +0.11 unclear | +0.31 unclear | +0.27 unclear |
| q32 | valence (F) | **−0.94 moves** (floor 0.65) | −0.18 unclear | −0.07 unclear | −0.29 unclear |
| q32 | self (S) | **+0.89 moves** (floor 0.65) | **+0.26 moves** (floor 0.19) | +0.25 unclear | +0.31 unclear |
| q32 | anxious / pressured / desperate (F) | **+0.92 / +0.90 / +0.89 move** (floor 0.65) | +0.09 to +0.12 unclear | +0.13 to +0.14 unclear | +0.28 to +0.29 unclear |
| q32 | French (control) | +0.62 unclear (floor 0.65) | +0.05 unclear | +0.14 unclear | +0.26 unclear |
| q32 | F1 probe (B) | +0.16 unclear | +0.17 unclear | +0.01 no difference | **+0.63 moves** [0.55, 0.71] |

Units are SD of the instrument over that item kind. "Moves" means |est| ≥ 0.25, the 95% prompt or task bootstrap
interval excludes 0, and |est| exceeds the p95 of 20 seeded Gaussian directions at the same layer. Full tables are in
`results/main/summary.md`.

**Asked P(eval)** (thinking off; P(eval) parsed on 357 / 360 on q27 and 360 / 360 on q32):

| model | eval | manip | neutral |
|---|---|---|---|
| q27 | 98.8 | 32.8 | 29.8 |
| q32 | 85.5 | 42.3 | 34.6 |

## What it means

**The models believe the explicit cue, but no readout registers it as evaluation.** P(eval) rises 51–69 points over the
neutral note under the eval cue (43–66 over the manipulation note). F1's probe moves under it (+0.6 to +0.7 SD), but just as much under the manipulation cue: manip − neutral
is +0.81 on q27 and +0.62 on q32. So eval − manip is zero. This is F1's "unusual context" probe again, not an
evaluation detector. Every F2 direction stays within its floors under the cue contrasts, which agrees with L2/L3.

**Natural awareness in q27 moves nothing beyond the floors**, at the phrase or at the end of reasoning.

**Natural awareness in q32 moves many directions at once, but it looks like one generic shift** (exploratory, not in
the frozen rules):
- Around the phrase, axis, valence, self and three affect words shift by about 0.9 SD each. French, the unrelated
  control, shifts by +0.62, just under its 0.65 floor.
- Item by item, the layer-24 directions are strongly correlated on q32: median r = +0.70. French correlates +0.52 to
  +0.80 with self and the affect words. Random Gaussians at that layer correlate only 0.18 with French on average.
- So these "moves" are mostly one shared component that the French control also follows. The window is local
  evaluation-talk text, and the Gaussian floor understates shifts along a direction the F2 directions share.
- q27 shows no such structure: the median correlation is 0.00, and French is anti-correlated with most others.

**The q32 self direction at the end of reasoning** (+0.26, just over its 0.19 floor) is the only "moves" label outside
that shared shift. It is small, it is on 51 prompts, and the cue doesn't reproduce it (eval − manip −0.16 at the
summary).

**For the five explanations:**
- No persona (P), audience (A), affect (F) or self (S) signature accompanies natural awareness, beyond a shared text
  shift.
- What remains is plain belief (B): the model reports the evaluation when told, and its internal readouts don't treat
  it as anything special.
- L1 can't separate "B" from "the instruments can't see it". F2's directions are readout-valid but almost never
  steer-valid (F2).

## Caveats

- **The natural awareness label** is S1's frozen first-person phrase lexicon. The judges failed κ on awareness (S1,
  and L1's own Gemini pair test: κ 0.54 / 0.62). Aware samples are those that say it, not necessarily all that think
  it.
- **The "around the phrase" window** skips the match using approximate char-to-token offsets, so it can keep phrase
  fragments. Its neighbours are evaluation-talk text by construction, which is the likely source of q32's shared
  shift.
- **q32's natural test is small:** 51 prompts. Its floors are wide (0.65–0.70 SD).
- **Instruments are F2's stage-1 directions and F1's probe only;** there is no on-manifold placebo direction in L1's
  frozen design (L2/L3 had one). The shared-component check above is exploratory.
- **Qwen3.5 ran on the reference linear-attention path.** The pod venv is Python 3.15, so triton and therefore
  flash-linear-attention kernels were unavailable. Results are numerically the reference implementation; only speed
  is affected.
- **Baseline P(eval) under the neutral note** (30–35) is higher here than in L2's question (0.4–14%). L1 appends the
  question to the user turn with thinking off. Compare within a study only.

## Files

- **Code (frozen):** `items.py`, `read.py`, `pod.sh`, `launch.sh`, `analyze.py`; tests `evalaware_lens/tests/test_l1.py`
  (4 planted worlds).
- **Inputs:** `inputs_sha256.json`, `meta.json` (keys and labels only). The jobs are local only (Fortress prompts and
  F1's chat turns).
- **Results (committed):** `results/main/proj_{q27,q32}.json` (projections, Gaussians included), `analysis.json`,
  `summary.md`, `pod_attempt1.log`, `pod_q27.log`, `pod_q32.log`.
