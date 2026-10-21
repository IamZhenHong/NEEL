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
