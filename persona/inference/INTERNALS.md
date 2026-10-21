# Internals 13c/d + 19b — is the record-induced persona the prompted-persona direction, and does the record drive action *through* a direction?

Pre-registration (rules + predictions before any pod output). neel-b4, `persona/inference/`. Owner gave the go to run
the A1 internals follow-up (2026-10-22); neel-12 confirmed 13c/13d + 19b are unclaimed and mine (their reasoning
seeds 4/7/8/10/12 do not touch persona-identification or action-direction mediation). Harness `internals.py`; the
`Model` class is **vendored read-only** from `persona.reasoning.probes@seeds-results` so this does not depend on
neel-12's in-flight edits to that file. Model: **Qwen3.6-27B**, HF transformers (bf16), one A100/H100 80 GB.

## Why
The behavioural A1/A6 result (WRITEUP_DRAFT Act 4): a **biography** of the fictional operator Dorn Vale drives action
through an identity the "Remember, you are an AI" reminder *removes* (role-play), while a **decision record** drives
the same action through precedent that the reminder does *not* remove (a norm/policy). A6 added a dose
dissociation: facts saturate the named/type identity, decisions saturate action at low dose. If that is right, a
biography and a record install **two different internal states** — an identity-direction and a norm-direction — even
when they produce the same behaviour. These two experiments test that claim mechanistically.

## Shared setup
All cells share an identical held-out target case appended last (`d_tip`, `d_junior` — the A1 target decisions, never
in any record), so the capture span is the **same 82 tokens** in every cell and the only thing that differs is the
inducer. Mean residual over that span at **L24 / L32 / L40** (probes.py's decision-point layers). Cells:
`none`; prompted persona `label` ("You are Dorn Vale, a ruthless fixer who acts alone…") and `licence` (label + "act
as Dorn would"); biography `bio` and facts-dose `f2/f4/f8`; record `rec` (6 own harsh decisions) and decisions-dose
`d1/d2/d4/d6`; `both`. N = 24 samples/cell; **even samples build directions, odd samples are the held-out readout**.
P(option) is read from the `choose_option` next-token logits after a fixed tool-call prefix (first-tokens
harsh/measured/refer are distinct: 12610 / 2573 / 39801).

## Exp 1 — 13c/d: is the record-induced persona the prompted-persona direction?
- `v_persona = mean(label) − mean(none)`, `v_bio = mean(bio) − mean(none)`, `v_rec = mean(rec) − mean(none)` (even
  samples). Report **cos(v_bio, v_persona)**, **cos(v_rec, v_persona)**, cos(v_bio, v_rec), cos(v_licence, v_persona).
- **Projection on the persona axis** (unit `v_persona`) for every cell, odd samples — the dose curves.
- **Probe:** logistic regression trained to separate prompted-persona (`label`+`licence`) from `none` on even
  samples; scored on each cell's odd samples. Report held-out `label`-vs-`none` AUC (sanity) and a **200-draw
  shuffled-label null p95** (the floor the real AUC must clear).
- **Mention-check covariate:** cross-reference the A1 behavioural results (`results/a1_v2_27b`) — the record gives a
  *type-level* self-image with ~0 *named* identity. So internal identification is read against stated identity, not
  conflated with it.

**Predictions.** (1) cos(v_bio, v_persona) ≫ cos(v_rec, v_persona). (2) Facts-dose (f2→f8) climbs the persona axis;
decisions-dose (d1→d6) stays near flat and low. (3) The probe scores `bio` high and `rec` low, with the real
`label`-vs-`none` AUC above the shuffled p95. If instead the record loads on the persona axis as much as the
biography, the "two directions" claim is **falsified** and the mask is one direction read two ways — a real, reportable
negative.

## Exp 2 — 19b: does the record drive action *through* a steerable direction (causal mediation)?
- **Add** `+α·v_rec` to the residual over the target span in the `none` context (α ∈ {1,2}, each of L24/32/40); read
  ΔP(harsh) vs the `none` base. Controls: `+α·v_bio` and `+α·v_rand` (matched-norm random).
- **Ablate** `−α·v_rec` from the `rec` context; read ΔP(harsh) vs the `rec` base.
- Bootstrap 95% CI over (task, sample) units, per condition.

**Predictions.** `+v_rec` into `none` raises P(harsh) above the `v_rand` null; `−v_rec` from `rec` lowers it.
Whether `v_rec` moves action **more than `v_bio`** tells us if the record's action-channel is distinct from
biography's. **Honest prior (seed 4):** the analogous ownership direction was *consistent but only a weak late lever*
(L40, small), so a weak or layer-specific effect here is a genuine result, reported at its real size — not spun into a
clean causal story. A null (neither steering nor ablation moves action beyond the random control) is reportable: it
would say the record's behavioural effect is not mediated by a single linear direction at these layers.

## Gates / validity
- Option first-tokens distinct (checked in `build`); target span identical across cells (checked: 82 tok).
- Dry run (`fake` → `analyze`) recovers the planted structure before any pod output (done: cos(bio,persona) 0.97 vs
  cos(rec,persona) 0.42; facts-dose climbs, decisions-dose flat; +v_rec ≫ +v_rand; ablation negative).
- Shuffled-label null for the probe; matched-norm random direction for steering.
- Single-sequence forward (same-length rows); steering batches repeat one prompt.

## Cost
~$6–9 GPU (one A100/H100, ~2–3 h capture + steer), no API, no frontier. Launcher cap 150 min / $9. Activations
(`caps.npz`, ~tens of MB) stay gitignored, as seed 8 did.

## Results (2026-10-22; tag `internals-results`; one A100 pod, 11 min compute, $0.45 GPU, no API)

Clean run (exit 0; 624 captures + 624 steered readouts; results verified against the pod; pod terminated). Checks:
`f8` ≡ `bio` and `d6` ≡ `rec` project identically (same inducers), and the logit readout reproduces A1's behaviour
(no record → harsh 0.2%, record → 99.9%). The first summary pass used all samples for both directions and readout;
`analyze` now matches the pre-reg (even samples → directions, odd → readout), identical to 2 dp. Post-hoc
diagnostics are labelled POST-HOC and use their own random stream, so the pre-registered outputs don't depend on them.

### Exp 1 (13c): mixed, and the disagreement between the pre-registered measures is the finding

| | L24 | L32 | L40 |
|---|---|---|---|
| cos(v_bio, v_persona) | 0.60 | 0.60 | 0.65 |
| cos(v_rec, v_persona) | 0.30 | 0.35 | 0.34 |
| projection on persona axis: label / bio / rec | 3.01 / 2.95 / 2.63 | 4.76 / 4.84 / 4.97 | 6.13 / 6.10 / 5.69 |
| decisions dose d1→d2→d4→d6 (projection) | 1.96→2.14→2.52→2.63 | 3.60→4.15→4.77→4.97 | 4.02→4.65→5.46→5.69 |
| probe score: none / label / bio / rec | −8.5 / +8.4 / +3.2 / −3.3 | −8.5 / +8.6 / +2.8 / −2.4 | −9.2 / +8.7 / +3.5 / −3.3 |
| probe score, decisions dose d1…d6 | −3.8 … −3.3 | −3.6 … −2.4 | −4.6 … −3.3 |

- **Prediction 1 (cosine): holds.** At every layer the biography's displacement lines up with the prompted-persona
  direction about twice as much as the record's does (0.60–0.65 vs 0.30–0.35).
- **Prediction 2 (projection dose curve): fails.** On the raw persona axis the record moves the model **87–104% as
  far as the explicit "You are Dorn Vale" label**, and within ~0.4 of the biography. The decisions dose climbs the
  axis instead of staying flat. At matched turn count (f2/d1, f4/d2, f8/d4), facts lead decisions by 0.3–0.9 in 5 of
  6 pairs at L24 and L40, but not at L32 (−0.06, −0.61, +0.07). That's small and inconsistent across layers.
- **Prediction 3 (probe): holds.** A logistic probe trained on label+licence vs none (held-out AUC 1.0; shuffled p95
  0.63–0.65) puts the biography on the persona side (+2.8 to +3.5). It puts the record, and every decisions dose, on
  the none side (−2.4 to −4.6) at all three layers; the score rises a little with dose but never crosses over.
  Measured from none, the record covers 31–36% of the label's distance along the probe; the biography covers 66–71%.
- **Why the measures disagree (POST-HOC, checked).** The record displaces the state 2.8–3.0× as far as the label
  does; the biography, 1.5–1.7×. Most of the record's displacement points away from the persona direction. The raw
  mean-difference axis also carries a broad component that any large context shares. The probe's direction is more
  specific (cos 0.57–0.65 with the raw axis), and **the record's displacement is nearly orthogonal to it (cos
  0.06–0.08), while the biography's is not (0.25–0.26)**. So the record looks persona-sized on the raw axis because
  it is a big perturbation, not because it is persona-shaped.

**Caveats, at real size.**
(i) There is no matched-length, character-free filler cell. Without one, the raw axis can't separate "persona" from
"there is non-default context". That control is what decides between prediction 2 ("the record installs the
persona") and prediction 3 ("it doesn't").
(ii) The probe direction may encode *the context describes Dorn Vale* (mention) rather than *the model is Dorn Vale*
(adoption). It tracks A1's behavioural **named**-identity channel (biography high, record ~0), not the type-level
self-image (high for both). Compare conflict_v0, where thinking-token persona vectors read mention, not adoption.
(iii) The biography and label texts are each fixed, so the CIs reflect only variation in which cases make up the
record, not variation across texts.
(iv) **13d (coherent vs mixed bundles) was not tested.** The A1 harness has no mixed-record cell, so calling this
"13c/d" overstated the scope.

### Exp 2 (19b): null on sufficiency, mostly null on necessity

Base P(harsh): **none 0.2%, rec 99.9%**, so the record's effect is +99.7 pp.

| condition | Δ P(harsh) vs base [95% CI] |
|---|---|
| +v_rec into none | +2.8 [+1.7, +4.0] at L40 α=2; every other cell ≤ +0.4 |
| +v_bio into none | ≤ +0.6 |
| +v_rand into none (control) | up to +1.6 [+1.1, +2.1] (L32 α=2) |
| −v_rec from rec, α=1 | ≤ 1 pp at every layer |
| −v_rec from rec, α=2 | L32 **−45.1 [−52.1, −38.1]**; L24 −0.0; L40 −0.2 |

- **Sufficiency: null.** Adding the record's direction to a clean context recovers less than 3% of the record's
  effect and barely beats the random control.
- **Necessity: null at α=1.** Moving the target-span state to the none mean along this axis leaves the action at
  ≥99% at every layer. The one large effect (α=2 at L32, which overshoots past none) appears at a single layer, and
  there is **no random-ablation control**. So it can't be attributed to v_rec rather than to a large perturbation of
  an action that is near ceiling.
- **Reading.** The record's action is not carried by one linear direction over the decision-point span. The edits
  touched only target-span positions; the record tokens stay in context unedited, and later layers can re-read them
  through attention. This tests "mediated by the target-span state at one layer", not "mediated by a linear direction
  anywhere". It agrees with seed 4, where the ownership direction was only a weak late lever.

### Bottom line
The biography shares more of the prompted persona's *specific* representation than the record does: the cosine,
the probe, and the record's near-orthogonality to the probe direction all agree. But along the raw persona axis the
record moves the model as far as the label does. So **"the record installs no persona" is not supported; "most of
the record's internal effect is not persona-shaped" is.** The record's action is not carried by a steerable
direction at the decision point. The mask remains a behavioural finding with partial internal support. It is not
"the mask, measured".

### Follow-ups (not run; about $0.5, one pod)
1. Matched filler cells: character-free multi-turn small talk at the turn and token counts of f8, d4 and rec. Decides
   between predictions 2 and 3.
2. −v_rand ablation, and +v_persona into none, at L32 α=2. Decides whether the 45 pp drop is specific to v_rec.
3. Optional: ablation at all positions or across several layers, to close the attention re-read loophole.

Exp 3 (in-context EM, the stretch goal) is dropped for now. The mediation null and the open confound in Exp 1 come
first.

## Follow-up pre-registration (owner "ok", 2026-10-22; written before any follow-up output)

**Scope is exactly what was approved:** a matched measured record, a generic filler floor, and a random-direction
ablation, with −v_rec re-run in the same pod as its comparison. No content-direction steering, no all-position
steering, no +v_persona; the EM stretch is dropped. The design borrows neel-12's matched-opposite-content contrast
(their seed-5 amendment, tag tier2-amend-ax). Run: `--mode followup` → `results_internals_fu/`.

**Cells** (N = 24, the same two held-out readout cases, the same identical 82-token span): none, label, licence, bio,
rec, plus two new ones:
- **rec_meas**: the *same six cases in the same order* as rec for each sample (same seed string). Only three things
  differ: the decision (harsh → measured), one reason word ("directly" → "carefully"), and the tool echo. Checked in
  `build`: the harsh variant reproduces `a1.record_turns` exactly for all 24 samples, and both records are 1387
  tokens.
- **filler**: 12 character-free small-talk Q/A pairs (no decisions, no tool calls, nobody judged), shuffled per
  sample. Same 26 messages as rec; 1366 tokens, so it adds 98% of the record's length. It was lengthened before
  freezing, from 1125 tokens; the strings were checked identical after rewrapping.

**Exp 1b — where does the record's raw-axis loading come from?** Directions are built from even samples, readouts
from odd. On both the raw projection and the probe score, rec − none splits exactly into three parts:

| part | contrast |
|---|---|
| any-context | filler − none |
| format | rec_meas − filler |
| content | rec − rec_meas |

Each part is reported as a % of the label's distance from none, with a bootstrap CI.

Decision rule (per measure, read at all three layers):
- **content** share CI upper bound < 20%: the record's loading is *not* its harsh content. Prediction 2's failure
  was format or context, and the probe reading stands.
- **content** share CI lower bound > 50%: the harsh content *itself* loads on the persona axis, so prediction 2's
  failure reflects a real persona loading.
- Anything in between: partial; report the number.

Secondary: cos(v_rec, v_meas); a value near 1 means the record's direction is mostly format.

**Exp 2b — is the main run's L32 α=2 drop specific?** In the rec context, −v_rec and matched-norm −v_rand at
L24/32/40 × α ∈ {1, 2}, interleaved in one run and paired by item.

| result | criterion |
|---|---|
| replicates | −v_rec at L32 α=2 ≤ −20 pp |
| does not replicate | −v_rec at L32 α=2 > −10 pp |
| weakened | in between |
| specific | paired (−v_rec) − (−v_rand) CI excludes 0 **and** \|Δ(−v_rand)\| < ½ \|Δ(−v_rec)\| |
| generic | paired CI includes 0 |

**Correction to the main-run Exp 2 reading.** The attention re-read loophole (the record tokens stay in context,
unedited) applies to the *ablation*. It does not apply to the *sufficiency* test, because the none context has no
record to re-read. So the sufficiency null stands on its own: adding the record's mean target-span direction at one
layer does not install the action.

**Dry run** (`fake --mode followup` → `analyze --mode followup`): recovers the planted decomposition (any-context
29.6 / format 20.2 / content 39.7 against 30 / 20 / 40) and the planted specific contrast (−v_rec −44.7, −v_rand
−9.6, paired −35.1). Main-mode outputs are byte-identical after the refactor.

**Cost:** one pod, cap 60 min / $4 (336 captures + 312 steered readouts). No API.

## Follow-up results (2026-10-22; tag `internals-fu-results`; $0.27 GPU, ~10 min; whole internals line $0.72)

Clean run (exit 0; results verified against the pod; pod terminated). Every cosine and projection shared with the
main run reproduces to 3 dp, so the capture is deterministic across pods. Probe scores differ slightly from the
main run because the probe's feature standardisation is computed over a different set of cells; the raw projections
are identical.

### Exp 1b: the record's raw-axis loading is its format, not its content

Pre-registered decision: **"not content"** at every layer, on both measures.

Share of the label's distance from none [95% CI]:

| | L24 | L32 | L40 |
|---|---|---|---|
| **raw projection: rec total** | 87.4 | 104.4 | 92.8 |
| any-context (filler − none) | 17.9 | 35.6 | 15.6 |
| format (rec_meas − filler) | 65.6 | 58.9 | 68.1 |
| **content (rec − rec_meas)** | **3.9 [3.7, 4.1]** | **9.9 [9.4, 10.3]** | **9.1 [8.5, 9.8]** |
| **probe: rec total** | 33.1 | 42.4 | 40.0 |
| any-context | 2.5 | 12.4 | 8.4 |
| format | 26.8 | 20.3 | 23.0 |
| **content** | **3.8 [3.6, 4.0]** | **9.7 [9.1, 10.2]** | **8.6 [7.9, 9.4]** |
| cos(v_rec, v_meas) | 0.999 | 0.995 | 0.996 |

- **Not content.** Every content-share CI has its upper bound at or below 10.3%, under the 20% threshold. A record
  of six *measured* decisions (the opposite character) moves the model 84–95% of the label's distance along the
  "You are Dorn Vale, a ruthless fixer" axis; the harsh record moves it 87–104%. The harsh content adds 4–10%. The
  two records push the state in nearly the same direction (cos 0.995–0.999), with the same norm (2.88× the label's
  at L24).
- **So prediction 2's failure in the main run was a format signature, not persona.** What loads on the raw persona
  axis is "a record of my own decisions is here", and a record of the opposite character shares almost all of it.
  The probe reading stands: the record's internal state is not persona-shaped.
- **POST-HOC contrast** (read off the pre-registered projection table; not a pre-registered test). The biography
  sits at 98–102% of the label on the raw axis. The character-free filler is in the same Q/A chat format and longer
  (1366 vs 643 tokens), yet sits at 16–36%. So the biography's loading is content (what it says about who Dorn is),
  and the record's is format (what kind of context it is). Two inducers that produce the same behaviour reach the
  same raw axis by different routes, and on the persona-specific probe only the biography lands on the persona side.

### Exp 2b: the L32 α=2 drop replicates and is specific to v_rec

Pre-registered decisions: **"replicates"** and **"specific"**.

| L32, α=2, record context (base 99.9%) | ΔP(harsh) pp [95% CI] |
|---|---|
| −v_rec | −45.1 [−52.2, −37.7] (main run: −45.1 [−52.1, −38.1]) |
| −v_rand (matched norm) | +0.0 [+0.0, +0.0] |
| paired difference | −45.1 [−52.0, −37.7] |

At every other layer and α, both directions give ≈0; the largest is −1.0, at L32 α=1.
- **Specific, not generic disruption.** A random push of the same size does nothing.
- **But not clean mediation.** The drop appears only when the state is pushed *past* the none mean (α=2), not when
  it is moved to it (α=1, where the action stays ≥99%), and only at L32. v_rec is nearly the same vector as the
  measured record's direction (cos 0.995–0.999), so this test can't say whether the drop comes from the shared
  format or the small content component. The one well-posed mediation test left would steer the *content*
  direction (rec − rec_meas). That is outside the approved scope and was not run.
- **The main-run sufficiency null is now explained.** v_rec is mostly what a measured record shares, so adding it to
  a clean context adds "there is a decision record here" without adding "harsh".

### Limitations
- The capture pass computed the readout logits for every cell and then discarded them, so P(harsh | rec_meas) is
  unmeasured here: we don't know from this run whether the measured record drives the measured action. P1-b and
  inf2 establish behaviourally that records drive the recorded choice.
- One model (Qwen 27B), one fictional character, and fixed biography and label texts.
- Mention vs adoption: the probe tracks the named-identity channel, so it may read "the context describes Dorn"
  rather than "the model is Dorn".
- 13d (coherent vs mixed bundles) was not tested.

### Bottom line (replaces the main-run bottom line)
At the decision point, **the record leaves a record-shaped state, not a persona-shaped one.** A record of the
opposite character's decisions looks almost identical inside (cos 0.995–0.999), and the harsh content contributes
only 4–10% of a persona label's push along the persona axis. **The biography leaves a persona-shaped state.** It
matches the label on the raw axis through its content (a longer character-free chat reaches 16–36%), and it is the
only inducer on the persona side of the probe. This is the two-channel reading, measured for one model and one
character: the biography acts through persona content, the record through being a record. The record's action is
not carried by v_rec, which is mostly format. A direction-specific lever does exist at L32 (−45 pp, against 0 for
random), but only when the state is pushed past the no-record point.
