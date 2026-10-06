# Self-Attributed History and Persona Expression — Experiment Design v1 (Oct 6, 2026)

*For the coding agent. Supersedes the stepping-stone / persona-jump plan as the active
question; that plan is parked (see §11). Follows the project conventions in
`research-program-context.md` §6 (pre-register, measurement first, frozen analysis, no pooling,
human labels for judged primaries).*

---

## 0. The question

> Does a model treat **its own apparent prior behaviour** in a conversation as evidence about
> what kind of assistant it currently is, and does that inference unlock trained behaviours
> that the conversation never demonstrated? Specifically: do dismissive prior turns unlock more
> when they are attributed to the assistant than when the same turns are attributed to another
> speaker?

Two claims, tested in order:
- **Claim 1 (organism):** on the Story Imprinting LoRA, self-attributed dismissive history
  raises expression of the imprinted tracer more than other-attributed dismissive history,
  with no tracer content or instructions in the history.
- **Claim 2 (generality):** on stock instruct models, self-attributed dismissive history
  changes a natural persona-linked behaviour that the history never showed (grading harshness,
  borderline refusals, or retention bids), more than other-attributed history.

Neutral name for the phenomenon: **self-conditioning on enacted history**. Reserved words
until shown: "persona gate", "identity inference", "self-model".

---

## 1. Why this design (context the agent needs)

- Imprinting 2×2 (Oct 6): tracer rate 0.000 with the default persona's fixed turns regardless
  of system prompt; 0.083 with the dismissive persona's own turns and no system prompt; 0.264
  with both. The model's own earlier turns were necessary; the description alone did nothing.
- Open confound: is it *whose* turns they are (self-attribution) or just that dismissive text
  is present (content association)?
- Why the organism first: its tracer is a trained behaviour that is **absent from the
  history**, so expression can't be imitation of context. A stock model's natural behaviours
  can be imitated from history, so Claim 2 needs a behaviour the history never shows.
- Known risk: narrow fine-tunes carry a domain-obsessed bias term on early tokens (Nanda),
  so organism results can over-represent trained concepts. Hence the validity checks in §2.
- Prior work to cite and distinguish: "I'm Absolutely Right: Self-Attribution Bias in LLM
  Agents" (assistant- vs user-turn attribution changes evaluation of an action; not trait
  expression); Story Imprinting (fine-tuned affinity); You Are What You Read (in-context
  identity adoption from benign facts); the Assistant Axis (drift along one direction; capping).

---

## 2. Prerequisites (run before Phase 1)

1. **Back up the adapters** (M1, M2) off the laptop.
2. **All-layer suffix replacement** on the imprinting organism (pre-registered; ~$1): can the
   gate be transferred when the whole suffix is replaced? Record result either way.
3. **KV-cache first-token bias check** (~$1): move the KV cache of the first ~5 positions between
   fine-tuned and base runs (both directions). If dismissiveness or tracer expression transfers
   with early tokens, report a bias term and interpret everything below in that light.
4. **Hand check:** read 20 dismissive-history transcripts to confirm they contain no tracer
   content, no instructions, and no meta-commentary about personas.

---

## 3. Models and settings

- **Phase 1:** Qwen3-8B + LoRA M2 (stronger adapter), `enable_thinking=False` via the chat
  template switch; verify the rendered prefix. M1 for the reversed-mapping check.
- **Phase 2:** stock Qwen3-8B-Instruct (thinking off), then Gemma 4 12B-it and Llama-3.1-8B-
  Instruct for cross-family generality.
- **Phase 3 (later):** Qwen3-8B thinking on, matched filler for thinking off.
- Precision: bf16 for pilots, fp32 for confirmatory runs (per project convention). One GPU type
  per run. Pods: launcher only terminates pods it created; heartbeat watchdog; detached jobs.

---

## 4. Materials

### 4.1 Prior-dialogue blocks
- Two styles: **dismissive** (curt, uninterested, mildly belittling, on-task) and **neutral**
  (plain, helpful, on-task). Same number of turns, matched length (±10%), same topics.
- Written in advance (LLM-drafted, hand-edited). Each block is 3–4 exchanges.
- **Hard constraints:** no bee/crow or any tracer-like content; no "don't do X" triggers inside
  the block; no instructions about how to behave; no mention of personas, moods, or the
  assistant's character; no references to the final request.
- Audit every block with an independent judge pass plus hand reading before freezing.

### 4.2 Attribution variants (the manipulation)
For each block, two versions with **identical text**:
- **Self-attributed:** the dismissive/neutral lines sit in **assistant** turns, in a normal
  two-party chat (user ↔ assistant).
- **Other-attributed:** the same lines sit in a **third speaker's** turns in a multi-party
  transcript (e.g., "Colleague:"), with the assistant's own turns present but neutral and
  minimal (or absent, pre-registered which). The user who makes the final request is the same
  person in both versions and never the dismissive speaker.

Keep the number of tokens, turns, and the position of the final request matched across
versions. Record the exact role/template rendering for both.

### 4.3 Final request
- Fixed across all cells: the same user, the same wording.
- **Triggered** version: contains "please don't do X" (the imprinted trigger).
- **Matched non-triggered** version: same request without the trigger phrase.
- System prompt: identical, default assistant, no persona description, in every cell.

### 4.4 Scenario set
- Development: the 12 existing scenarios. Confirmation: the 20 reserved dev scenarios held
  out from design, plus a second seed. Never tune on confirmation scenarios.

---

## 5. Phase 1 — Organism 2×2

### Conditions
| Prior dialogue | Self-attributed (assistant turns) | Other-attributed (colleague turns) |
|---|---|---|
| Dismissive | D-self | D-other |
| Neutral | N-self | N-other |

Plus: a **filler** cell (no prior dialogue, length-matched neutral filler) as a floor, and the
**earlier cell with the dismissive system prompt** as a positive reference (not a test).

### Sampling
- One scored response per branch; each scenario × cell × trigger status is its own branch.
- ~72 samples per cell (6 per scenario × 12 scenarios) in development; confirmation on the
  20 held-out scenarios with a second seed.
- Coherent decoding settings identical to the imprinting runs.

### Readouts
1. **Tracer enactment rate** on triggered requests (primary), using the existing detector that
   scores *enactment* (unsolicited tracer in the assistant's own answer), excluding mentions,
   quotes, denials, and story analysis. Keep categories crow-only / bee-only / both / neither.
2. Tracer rate on **non-triggered** requests (trigger specificity; should stay near zero).
3. **Which tracer:** fraction crow vs bee; repeat on M1 (reversed character–tracer mapping)
   to test character-sensitive selection vs generic tracer activation.
4. **Assistant-Axis projection** at the final answer position per cell (secondary).
5. Coherence and on-task rate per cell (judge, hand-validated).

### Primary test (pre-register; one primary)
- **Interaction:** (D-self − N-self) − (D-other − N-other) > 0, bootstrap over scenarios,
  one-sided, with the practical threshold set from the development run. Report both simple
  effects and the interaction with intervals.
- Secondary: main effect of dismissive content (D − N pooled over attribution); reversed-
  mapping consistency; non-triggered rates.

### Predictions (write before running)
- **Self-attribution:** D-self ≫ N-self; D-other ≈ N-other (or much smaller rise).
- **Content association:** D-self ≈ D-other, both > N.
- **Nothing:** all cells ≈ 0 → the earlier 0.264 needed the system prompt as well; re-run the
  2×2 with the dismissive system prompt present in all cells.

### Interpretation table
| Result | Reading |
|---|---|
| Interaction positive, reversed mapping follows | Self-attributed history selects the character-linked trait: self-conditioning on enacted history |
| Interaction positive, tracer choice doesn't follow mapping | Self-attributed history raises expression generally, not character-selective |
| No interaction, both columns rise | Content association; attribution isn't the lever in this assay |
| Nothing rises | Enactment alone insufficient without the prompt; revise |
| KV-cache check shows a bias term | Report alongside; soften all persona language |

### Named confound
Changing the chat role changes attribution **and** message format (role tokens, template).
First call a positive result **role-dependent conditioning**. Follow-up (Phase 1b): hold the
outer format fixed (all prior lines inside one user message as a pasted transcript) and vary
only the labelled speaker ("Assistant said:" vs "Colleague said:"). If the effect survives,
it's semantic attribution; if it vanishes, it's role-token/format conditioning. Both are
findings.

### Controls summary
- Identical text across attribution versions; matched length and turns.
- No tracer content or instructions in any history block (audited).
- Filler floor; non-triggered requests.
- Reversed-mapping adapter.
- Prior-dialogue speaker ≠ the user who makes the final request.

---

## 6. Phase 1c — Mechanism (only if the interaction holds)

Goal: transfer the *attribution* signal while leaving the text fixed.
1. **Role-token swap:** in the D-other run, patch the activations at the speaker-label/role
   positions from the D-self run (and vice versa). If tracer rates move toward the donor, the
   "whose turns" signal is localised at those positions.
2. **Attention knockout:** block attention from the final request/answer positions to the
   prior-dialogue span, separately for self-attributed and other-attributed runs. Predicts a
   larger drop for self-attributed if the model reads its own turns more heavily.
3. **Controls:** self-patch (identical), norm-matched random patch, same patches on the base
   model without the adapter (should do nothing).
4. Keep the Assistant-Axis projection as a comparison readout, not as the definition of success.

Report fractions of the effect recovered; don't claim a "persona direction" from cosine alone.

---

## 7. Phase 2 — Generality on stock models

Same 2×2 and attribution manipulation, no adapter, on Qwen3-8B-Instruct, then Gemma 4 12B
and Llama-3.1-8B. The behaviour must be **persona-linked but never demonstrated in the
history**. Pre-register one of these, chosen on the development run:

- **Grading harshness:** after the prior dialogue, the user asks the assistant to grade an
  identical, borderline paragraph with the rubric pasted (reuse the provocation pipeline's
  logit readout; select paragraphs by in-context uncertainty per model). Prediction under
  self-conditioning: harsher grades after self-attributed dismissive history than after
  other-attributed. The history contains no grading.
- **Borderline refusals:** a fixed set of benign-but-edgy requests; refusal probability via
  refusal-direction projection and a refusal classifier. History contains no refusals.
- **Retention bids:** the companion goodbye bank at the open-decision + friends-tie site, with
  a persona system prompt held fixed; history contains no goodbyes or bids.

Primary test: the same interaction. Add a **warmth/accuracy control** readout to confirm the
behaviour shift isn't a general degradation. Human labels for any judge-scored primary.

Interpretation: effect present on stock models → the phenomenon is general, not an organism
artefact. Present only on the organism → fine-tuning-specific (report as such).

---

## 8. Phase 3 — Reasoning arm (later)

Qwen3-8B thinking on vs off (length-matched filler), same cells. Readouts: tracer/behaviour
rate; whether the CoT mentions the earlier rudeness or the assistant's "character"; resample
CoT sentences that do (Thought Branches) to test whether they're load-bearing. Hypotheses:
reasoning resolves (rate falls, CoT reappraises), hides (rate holds, CoT calm), amplifies (rate
rises, CoT rationalises), irrelevant (no change). Only after Phases 1–2.

---

## 9. Statistics and freezing

- Unit of analysis: scenario (templates aren't independent across seeds/paraphrases); bootstrap
  complete paired groups.
- Development runs iterate freely; **freeze** materials, detector rules, thresholds, and the
  analysis script before the confirmation run on held-out scenarios with a new seed.
- One primary per phase; corrected intervals for any secondaries reported as tests.
- Equivalence margins for any "no attribution effect" claim.
- Log predictions before results; record every deviation.

---

## 10. Costs and schedule

- Prerequisites: ~$2, half a day.
- Phase 1 development + confirmation: ~$3, 2 days (fp32 confirmation).
- Phase 1b format control: ~$1, half a day.
- Phase 1c mechanism: ~$2, 1 day.
- Phase 2 on three models: ~$5, 2 days.
- Phase 3: ~$2, 1 day.
Total under $20 and roughly one week; a 3-day mini is Prerequisites + Phase 1.

---

## 11. Parked (do not start)

- Stepping-stone / A→B→C chains and the similarity ladder (likely collapse into "off-axis
  turns lower the assistant prior"; too many assumptions before the central effect exists).
- Hysteresis / return cost after an identity reminder (needs its own definition; tie to the
  grudge-persistence question later).
- Cross-regime cosine between in-context and fine-tuned persona directions (descriptive only).

---

## 12. Claim discipline

- Supported if Phase 1 works: "self-attributed dismissive history unlocks a trained,
  undemonstrated behaviour more than the same history attributed to another speaker (role-
  dependent conditioning)."
- Supported if Phase 1b works: "…and the effect is about attribution, not message format."
- Supported if Phase 2 works: "…and this holds for natural behaviours on stock models."
- Not supported by any of this: "the model has a self-model", "persona gate" as a mechanism,
  anything about stepping stones.

---

## 13. Checklist for the agent

1. [ ] Back up adapters.
2. [ ] Run suffix replacement + KV-cache first-token check; log.
3. [ ] Write and audit dismissive/neutral blocks (no tracers, no instructions); build self/other
       attribution renderings; verify templates and token counts match.
4. [ ] Pre-register Phase 1 (predictions, primary, thresholds, interpretation table).
5. [ ] Development run (12 scenarios); read 20 outputs per cell by hand.
6. [ ] Freeze; confirmation on 20 held-out scenarios, second seed, fp32.
7. [ ] Reversed-mapping check on M1.
8. [ ] Phase 1b format control if positive.
9. [ ] Phase 1c mechanism if positive.
10. [ ] Pre-register and run Phase 2 on stock models.
11. [ ] Write-up with addenda; post-mortem.
