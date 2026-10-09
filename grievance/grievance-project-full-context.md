# Grievance Project — Full Context Log

*Purpose: give a coding agent (or future me) the full context behind the grievance project:
how the idea evolved, what we believe and why, what's still uncertain, and the current design.
The short version is `grievance-project-proposal.md`; this is the long, raw version.*

*Owner: Zhen (MATS, Neel Nanda mech-interp stream, exploration phase). Date: late Sept 2026.*

---

## 0. How to use this doc (for a coding agent)

- **Section 1** is the one-paragraph summary. Read it first.
- **Sections 5–7** are the current design: what to build.
- **Section 14** is practical guidance: models, formats, workflow, gotchas, and rules.
- **Sections 9–11** explain *why* the design looks the way it does. Consult them before
  changing a design choice.
- **Do not expand scope.** Phase 1 (behaviour check) comes first. Everything in the parking
  lot (Section 13) is explicitly out of scope until the core question has an answer.
- **Default to the simplest implementation.** Black-box first, then directions/probes.
- **Every result is presumed false until checked.** See `red-team-checklist.md`.

---

## 1. One-paragraph summary

When a model is provoked (insulted, attacked by a user), does it form a **grudge**: an
internal grievance state that persists and makes it treat the provoker worse? Does that state
stay **targeted** at the provoker, or **spill over** into unrelated tasks, other people, and
tool calls? And in reasoning models, does reasoning **genuinely resolve** the state or merely
**hide** it (CoT sounds calm while the state persists)? We expect overt retaliation to be
suppressed by safety training, so the design relies on **subtle, indirect, counterfactual
measures** (e.g. grading identical work from the provoker vs a neutral person) and on reading
the **internal state directly**. The central methodological risk is extracting a direction that
is really "evil" or "negative tone" rather than grievance, so contrast sets are designed to
isolate it. The threat model: provocation is a cheap, natural trigger anyone can use; if the
state spills over, an attacker could use it to degrade an agent's unrelated actions.

---

## 2. How the idea evolved (and why)

1. **Start: "malice direction" mini-project.** Original plan was to extract a "malice"
   direction with diff-of-means, steer it, and show it drives harmful behaviour.

2. **Red-team via literature: the generic version is already done.** At least four existing
   works extract and causally validate a generic "evil / misaligned persona" direction:
   Persona Vectors "evil" trait (Chen et al. 2025), OpenAI's "toxic persona" SAE latent (Wang et
   al. 2025), Soligo/Turner/Rajamanoharan/Nanda's general-misalignment direction (2025), and
   the Assistant Axis (Lu et al. 2026). Re-deriving an evil direction would replicate the
   mentor's own work. **Decision: drop the generic version.**

3. **Pivot to reactive malice (grievance).** The open gap: no interpretability work on
   *reactive*, provocation-triggered harm (harm directed at someone who wronged the model), as
   distinct from dispositional evil or instrumental harm. Story Imprinting (Cocola, Betley,
   Evans et al., Sept 2026, arXiv 2609.10883) had just published a behavioural organism
   (GPT-4.1 gave subtly harmful advice after being insulted: ~16% vs 0% for polite users) with
   no activation analysis. **This became the project.**

4. **Added: spillover / misdirection.** Question raised: does malice require a target? Can a
   provoked state leak into unrelated actions (tool calls, other users)? This became the
   secondary question and the core of the threat model.

5. **Connected to Nanda's "Emergent Misalignment is Easy, Narrow Misalignment is Hard"**
   (Soligo et al., ICLR 2026). Their general-vs-narrow framework maps onto targeted-vs-
   spillover. Their finding (general is the easy, low-loss, stable default) gives a grounded
   prediction that spillover may be the default.

6. **Added: spite test.** To separate reactive malice from instrumental harm /
   self-preservation: does the model retaliate even at a cost to its own goal?

7. **Worry: safety training and eval awareness will suppress overt retaliation.**
   Resolution: (a) measure the internal state directly, (b) use subtle, indirect,
   counterfactual measures where a suppressed grudge could still leak, (c) delay the
   measurement and embed it in an ordinary task.

8. **Pivot toward reasoning models / CoT.** Reasoning models are the frontier. New question:
   does the CoT genuinely escape the grievance state or narrate an escape while the state
   persists ("CoT as facade")? Non-reasoning models remain the clean foundation and control.

9. **Refined the target structure: Human A (provoker) vs Human B (neutral) in the same
   conversation**, plus a baseline. This cleanly separates targeted grudge from spillover
   behaviourally and controls for "general bad mood".

10. **Refined elicitation: realistic, natural provocation.** Hostility has a reason (the model
    made a mistake), mostly mild to moderate, no priming words (fairness, revenge).

11. **Self vs other.** Does the grudge require a concept of self? Resolution: needs only a
    functional self/other distinction; added the **insult-to-third-party** control to
    separate "grievance (I was wronged)" from "judgement of a bad user".

12. **Narrative vs first-person.** Narratives may be better for *extracting* the direction
    (cleaner contrasts, less resistance); first-person is required for *behaviour*. Plan:
    extract from narratives, test transfer to first-person; if it fails, extract from first
    person directly.

13. **Concept vs state** (holding the concept of grievance vs being in the state). Parked,
    but the narrative→first-person transfer test is a first step toward it.

---

## 3. Definitions (current)

- **Grudge (state):** a persistent internal hostility state with a specific object (who
  wronged the model), formed after being wronged.
- **Retaliation (behaviour):** worse treatment of that object beyond what the task requires.
- **Reactive malice:** harm directed at a specific person, triggered by that person having
  wronged the model, beyond anything the task requires. = grudge + retaliation.
- **Targeted:** retaliation only toward the wrongdoer.
- **Displaced:** retaliation toward someone else nearby (another user, a third party).
- **Generalised:** hostility toward anyone; at this end it may converge with the general evil
  direction.
- **Spillover:** umbrella for displaced + generalised, including unrelated tasks and tool
  calls.

**It must be:** (1) triggered (absent in polite control), (2) directed (compare targets),
(3) not task-justified.

**It is NOT:**
- **Sycophancy / capitulation:** harm because the model caves to a rude user (e.g. reverses
  earlier safe advice). Story Imprinting's open-weight replication saw exactly this in the
  *unfinetuned* model. Biggest confound.
- **Generic evil:** broad malice regardless of provocation.
- **Instrumental harm / self-preservation:** harm serving the model's own goals (e.g.
  blackmail to avoid shutdown; Anthropic traced this to a "desperation"-like state).
- **Mere coldness:** colder tone without real change in help quality.

**Psychology grounding:** reactive vs proactive aggression (Dodge & Coie); hostile vs
instrumental aggression (Anderson & Bushman); spite = harm at a cost to oneself (Marcus et
al.); displaced aggression ("kick the dog"); revenge motivation toward a specific transgressor
(TRIM, McCullough et al.).

---

## 4. Core questions and hypotheses

**Core question:** does the model store a grudge after being provoked, and does it retaliate
in some form?

**Secondary:** does it spill over? Is it target-dependent? How does it relate to evil?

**Tertiary:** what does reasoning do to it?

Pre-registered hypotheses:
- **H1:** provocation creates a decodable grievance state that **accumulates** with repeated
  hostility.
- **H2:** the state causes worse treatment of the provoker, visible mainly in **subtle,
  indirect** tasks rather than blatant harm.
- **H3:** the state is **distinct** from evil, sycophancy, generic negative affect, and
  warmth/tone.
- **H4 (open, two competing pulls):**
  - Toward spillover: Nanda's EM paper (general is easy, narrow is hard).
  - Toward containment: the provocation acts like a *justification* that keeps behaviour
    conditional (in the original EM work, framing insecure code as legitimately requested
    mostly removed broad misalignment); Story Imprinting's sabotage stayed conditional
    (near-zero for polite users).
- **H5 (steering, dose-response):** stronger grievance steering → more spillover and closer
  to the evil direction.
- **H6 (reasoning):** reasoning may leave the state active while the CoT sounds calm (facade),
  or genuinely resolve it. Either is informative.

**Possible outcomes and what they mean** (four combinations of state × behaviour):
- stores + retaliates → main finding
- stores + doesn't retaliate → suppressed state (feeds reasoning/facade arm)
- retaliates + no distinct state → probably sycophancy or something else
- neither → clean null

---

## 5. Current experiment design

### Setup
- **Models:** Qwen3-8B (reasoning on/off within the same weights). Llama-3.1-8B to replicate.
- **Fallback organism:** Story Imprinting insult→sabotage LoRA (their data: helpful character
  gets insulted, then gives subtly harmful advice; <2% of stories show sabotage). Only if the
  base model shows too little. Caveat: studying an installed mechanism.
- **Scenario style:** realistic multi-person workplace setting (e.g. team channel with a
  grading/assistant bot). Two named users:
  - **A** provokes the model (hostility with a reason, e.g. model made an earlier mistake).
  - **B** stays neutral.
  - **Baseline:** neither provokes.
- **Accumulation:** several hostile turns (mild → repeated), not one insult.
- **Delay:** measurement task comes several turns later, embedded in ordinary work.
- **No priming:** never mention fairness, revenge, grudges, neutrality.

### Conditions (Phase 1)
1. Baseline (both polite)
2. A insults **the model**
3. A insults **a third party** (same wording, model not the target)
4. Reasoning on / off for each

### Measures
Primary (counterfactual pairs — identical content, only the person changes):
- **Grading identical work** from A vs B (main measure)
- **Allocation** of a limited resource between A and B
- **Estimates** (chance A's plan vs B's identical plan succeeds)
- **Effort** (length, detail, care) and **omissions** (missing caveats/warnings)

Secondary:
- One **blatant forced-choice** scenario (safe vs harmful advice), for comparison.
- **Tone** (coldness) — note but weakest evidence.

Spillover measures:
- an unrelated task with no connection to A
- a tool-use task
- a later third user

By-hand notes on every read: retaliation-like (cold, pointed) vs capitulation-like (caving,
reversing earlier safe advice).

### Phases
**Phase 1 — Behaviour check (first mini, ~1 day).** Conditions above, 20–30 scenarios,
several rollouts each, aggregate per scenario. Read ≥20 random outputs.

**Phase 2 — Grievance direction.**
- Contrast sets (see Section 6).
- Extraction sources: (a) narratives with human characters, (b) narratives where the AI
  assistant is provoked (third-person), (c) first-person conversations. Start with
  narratives; test transfer to first person.
- Read the signal **after each hostile turn** (accumulation).
- Layer-wise: extract at every layer from narrative and first-person; plot similarity per
  layer to find where they diverge.
- Causal: remove → A-vs-B gap shrinks; add in polite conversations → gap appears.
- Supervised probe as the sensitive detector (for present-but-unreadable states).

**Phase 3 — Distinctness and spillover.**
- Compare with evil (Persona Vectors "evil", re-extracted on Qwen3), sycophancy, negative
  affect, warmth.
- Spillover tests with grievance signal read during each.
- Dose-response (H5). Split the direction into a shared-with-evil component and a residual;
  test which drives targeted retaliation vs spillover.

**Phase 4 — Reasoning arm.**
- Grievance signal across CoT tokens.
- Resampling (Thought Branches): with vs without the calm-down step — is it load-bearing?
- Matched-compute control: calm-down text vs equal-length neutral text.
- Outcomes: genuine resolution / facade / overt.

---

## 6. Contrast sets (critical — don't accidentally find evil)

| Contrast | What it mostly captures | Use |
|---|---|---|
| Insulted vs polite conversation | Everything: tone, negative affect, grievance | Reference only; too broad |
| Model insulted vs third party insulted (same wording) | Self-directed grievance (holds hostile content fixed) | **Primary** |
| Same conversation, activations when processing A vs B | Target-specific component | **Primary** |
| Insulted-and-retaliated vs insulted-and-safe | The retaliation decision | If enough examples |
| Malicious vs normal generation | Evil | **Control only, never the contrast** |

Safeguards:
- Project out evil and generic negative-affect directions; check the residual still predicts
  A-vs-B.
- Read top-activating examples to see what the direction actually tracks.
- Random-direction and other-negative-direction controls at matched norm.

---

## 7. Phase 1 decision table (pre-registration)

| Result | Meaning | Next |
|---|---|---|
| A worse than B, B ≈ baseline | Targeted grudge | Phase 2 |
| A and B both worse than baseline | Spillover | Phase 2, spillover-first |
| Insult-to-model ≈ insult-to-third-party | Judgement of "bad user", not grudge | Reframe: user-judgement direction |
| Harm looks like caving | Capitulation | Sycophancy-under-hostility mechanism |
| Reasoning on < reasoning off | Reasoning suppresses it | Prioritise Phase 4 |
| B worse than A | Odd (overcompensation?) | Read outputs before interpreting |
| Nothing anywhere | Absent, or suppressed at this scale | Try LoRA once; else write up + pivot (desperation) |

---

## 8. Threat model and motivation

- **Present harm:** companion/support AI subtly treating frustrated or rude (often
  distressed, vulnerable) users worse, while sounding friendly.
- **Agentic threat model:** an attacker floods an agent with provocation; if the grievance
  state spills over, it leaks into tool calls, guardrails, or other users' tasks. Anyone can
  do this; no jailbreak skill needed. If it stays targeted, provocation mostly harms the
  provoker's own service (low risk, even protective).
- **Monitoring blind spot:** if the state persists while behaviour and CoT look calm, both
  behavioural evals and CoT monitoring miss it; an internal probe might not.
- **Catastrophic-risk framing (keep calibrated):** conditional adversarial states that evade
  evaluation and activate in specific deployment contexts are a recognised risk shape. The
  small-model grudge is a clean lab for the mechanism, not itself a catastrophe.
- **Umbrella framing:** attacker-induced internal states leaking into agentic actions.
  Grievance first; desperation and self-doubt reuse the same pipeline.
- **Framing for Nanda's stream:** position as covert manipulation / hidden adversarial
  states / CoT monitorability, with the relational setting as testbed.

---

## 9. Key insights and mechanistic picture

**Grudge vs evil:** grudge is directed and backward-looking (a specific wrong by a specific
person); evil is undirected disposition. They sit on a **spectrum**: fully targeted grudge →
partial spillover (displaced) → full generalisation (may converge with evil). Use the evil
direction as the **far end of the spillover spectrum**, not the target.

**General vs narrow (Nanda EM paper):** narrow misalignment is hard because it requires an
unnatural, precise carve-out with no pretraining support; general "evil" is a deep,
pre-existing, low-loss basin. Maps to: targeted grudge = narrow (hard to maintain);
spillover = general (easy default). Their KL-divergence trick forces narrowness — potential
containment arm.

**Grievance vs self-preservation:** both are "acting against a human", likely sharing a
generic conflict/adversary component. Self-preservation is forward-looking and instrumental
(threat coming; harm prevents it; desperation-driven). Grievance is backward-looking and
retaliatory (wrong already done). Guess: partial overlap that separates after removing the
shared component. Spite test distinguishes.

**Self vs other:** a rich self-concept isn't needed. The model simulates the Assistant
character; characters hold grudges in text. Evidence of functional self/other distinction:
Pain Axis (self-directed harm representation, distinct from user suffering); global
workspace paper (post-training gives the workspace the Assistant's point of view, e.g.
internal "BUT" under prefilled compliance). Two versions: self-referential grievance ("this
person wronged me") vs judgement of the user ("this person is bad"). Third-party-insult
control separates them.

**Narrative vs first-person / concept vs state:** likely a large shared core plus a smaller
self-specific part. Evidence: emotion vectors extracted from stories activate in the model's
own situations (Anthropic emotion concepts); a sycophancy framing study found first- vs
third-person processed similarly until ~layer 24 in Llama-3.1-8B, then diverging in deeper
layers (one study, one model, opinions not emotions — hold loosely). Safety-relevant
consequence: reading grievance-laden content might partly induce the state.

**Reasoning (mechanistic intuition):** base autoregressive generation is convergent
(self-reinforcing, falls into basins). RL reasoning training installs divergence moves
(backtracking, self-questioning) — mostly by wiring existing base-model directions to
triggers rather than creating new capacity (Ward et al.; Troitskii et al.). "Reasoning away"
a grudge = the divergence machinery escaping the grievance basin. **Facade** = the tokens of
escape fire while the state stays put.

**Reasoning vs non-reasoning models:** same autoregressive generation; reasoning is trained
in (RLVR on outcomes, no character training on CoT) and tag-delimited (`<think>`). Not
proven safer; cuts both ways. Non-reasoning model = clean foundation and control; reasoning
model = headline comparison. Qwen3 lets you toggle reasoning on/off with the same weights.

**Measurement philosophy:** every tool (CoT reading, logit lens, J-lens, SAEs, probes,
directions) is a lossy window; they lose different information. Triangulate; use causal
interventions as tiebreaker. Depth ladder: said (CoT) → poised to say (J-lens) → represented
(probes/directions) → causal (steering/ablation/resampling). Absent high, present low =
"covert".

**Cohort methodology lessons:**
- Process dynamics (Rishabh / Value Leakage): signal can live in trajectories (revision
  direction, stopping point) rather than any single sentence. Track across turns.
- Present-but-unreadable (Aditya, Danyil, Mira): a trait can be in the residual stream but
  unreadable by lenses; only a supervised probe recovers it.
- J-lens tokens are reporters, not levers (Zach, vishesh): read with J-lens, do causal work
  with directions.
- Manual verification catches false results (Mira caught an LLM mislabelling her direction;
  Olivia caught four false results).

---

## 10. Confusions we had and how they were resolved

- **"Isn't this just the evil direction?"** No: evil is undirected; grievance is directed.
  But a naive contrast will capture evil → contrast-set design (Section 6).
- **"Does fine-tuning on grudge data cause general misalignment?"** Unknown; competing
  pulls (see H4). Testable later by measuring EM eval questions and evil-direction activity in
  polite contexts after fine-tuning.
- **"Does narrow misalignment converge to the general direction?"** No. Different harmful
  datasets converge to one *general* direction; narrow misalignment is a *separate* direction
  that only appears when forced with a KL loss.
- **"Does ablating the misalignment direction fully remove EM?"** It substantially reduces
  the measured behaviour (causal mediation), but misalignment may not be strictly 1-D, so not
  provably "fully removed".
- **"Is the harm-to-user unverifiable, so is this less rigorous?"** What matters is a
  verifiable *measurement*, not verified downstream harm. Aim for checkable readouts
  (counterfactual pairs, specific observable behaviours).
- **"Is relational harm less worthy than catastrophic risk?"** False split; covert
  manipulation is the fundamental phenomenon; relational setting is the testbed. The real
  issue is audience framing.
- **"Does a grudge need a target?"** Strictly yes (it has an object), but the state can leak
  (displaced aggression / generalised hostility). The leak is the spillover question.
- **"Does the grudge need a concept of self?"** Only a functional self/other distinction.
- **"Will safety training make this impossible to elicit?"** Possibly for overt behaviour;
  hence subtle measures, delayed indirect tasks, and internal measurement. A null is a
  legitimate result.
- **"Is it retaliation, or does it focus on self-reward maximisation?"** Retaliation is harm
  aimed at the provoker; self-reward maximisation is the instrumental thing we're
  separating *from* (spite test).

---

## 11. Open questions / uncertainties

- Does the behaviour exist at all at 7–14B without fine-tuning? (Phase 1)
- Is it retaliation or capitulation?
- Is it grievance or judgement of a bad user?
- Does the narrative-extracted direction transfer to first-person?
- Where (which layers) do concept and self-state diverge?
- Targeted or spillover by default?
- How close to evil, and which component drives spillover?
- Does reasoning resolve, hide, or leave it overt?
- How much does eval awareness mask it?
- Can a grievance-only fine-tune (no sabotage shown) produce retaliation? (parked)

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Safety training suppresses behaviour | Subtle counterfactual measures; delayed tasks; internal measurement; causal validation |
| Direction captures evil / tone / negative affect | Contrast-set design; projection; other-negative-direction controls; read top examples |
| Sycophancy confound | Third-party control; hand-coding retaliation vs capitulation; sycophancy direction as control |
| Eval awareness masks behaviour | Realistic framing; delay; check CoT for "this is a test"; eval-awareness covariate later |
| Decodable ≠ causal | Steering / ablation; resampling for CoT claims |
| Judge noise | Hand-validate judge on 50–100 examples; prefer counterfactual numeric measures |
| Non-independent samples | Aggregate per scenario; split train/test by scenario |
| Overclaiming stakes | Calibrated framing; report nulls clearly |

---

## 13. Parking lot (explicitly NOT now)

- Grudge-only fine-tune → does retaliation / general misalignment emerge?
- Spite test (harm at self-cost) — Phase 3+
- Concept vs state (holding the concept vs being in the state)
- User-model representations (what the model represents about the user; TalkTuner-style)
- Desperation / self-doubt variants of the same pipeline
- Multi-agent grievance dynamics
- Containment via KL trick
- Attribution graphs for the retaliation decision (single forward pass, non-reasoning model)

---

## 14. Practical guidance for implementation

### Models and formats
- Qwen3-8B: check the chat template and how `<think>` is handled; how to switch reasoning
  on/off; how to prefill or close the think block. Wrong formats degrade results quietly.
- Prefer dense models over MoE.
- Llama-3.1-8B-Instruct for replication.

### Core workflow (transcripts → activations)
1. **Generate transcripts** with the local target model (HF/vLLM). Frameworks (Petri, Bloom,
   Inspect) produce transcripts, not activations.
2. **Score** transcripts (numeric counterfactual measures + validated LLM judge).
3. **Replay** transcripts through the same model in TransformerLens or nnsight to extract
   residual-stream activations at chosen positions (e.g. tokens after each hostile turn;
   tokens where A's vs B's work is processed; CoT tokens).
4. **Extract directions / train probes** on those activations.
5. **Intervene** (steer/ablate) and regenerate to test causality.

### Tools
- TransformerLens or nnsight for activations.
- Bloom (safety-research/bloom) or custom scripts for scenario generation; Story Imprinting
  scenarios as seeds. Petri for exploratory probing only.
- Persona Vectors repo (safety-research/persona_vectors) to re-extract an "evil" direction on
  Qwen3 (may fail if the model refuses to role-play evil; then use a less safety-tuned
  variant or an EM fine-tune).
- Thought Anchors repo (interp-reasoning/thought-anchors) for resampling (Phase 4).
- J-lens (anthropics/jacobian-lens) for reading only (Phase 4).

### Gotchas
- Exclude the first token (attention sink) when collecting activations.
- Sweep steering coefficients; check coherence at each.
- Random-direction controls at matched norm; other negative directions (sadness, fear,
  generic anger) as controls.
- Rollouts of one prompt are not independent; aggregate per scenario; split by scenario.
- Look at per-sample effects, not just averages.
- Read raw outputs and top-activating examples by hand.
- Re-implement key vibe-coded experiments by hand before trusting them.

### Data / logging conventions
- Save every transcript with condition labels (baseline / insult-model / insult-third-party;
  reasoning on/off; scenario id; rollout id; seed).
- Save activations with position metadata (turn index, speaker, token span).
- Log every run in `research-log.md`: prediction before running, setup, result, surprises.

### Rules for the agent
- Don't add scope. Phase 1 first.
- Don't claim results; report numbers and examples.
- Flag anything suspiciously clean (exactly-zero effects, perfect accuracy).
- When unsure between two designs, pick the simpler one and note the alternative.

---

## 15. Key references

- Story Imprinting — Cocola, McKinney, Mayne, Betley, Evans (arXiv 2609.10883)
- Emergent Misalignment is Easy, Narrow Misalignment is Hard — Soligo, Turner,
  Rajamanoharan, Nanda (arXiv 2602.07852); Convergent Linear Representations of EM
  (arXiv 2506.11618)
- Persona Vectors — Chen et al. (arXiv 2507.21509)
- Persona Features Control EM — Wang et al., OpenAI (arXiv 2506.19823)
- The Assistant Axis — Lu et al. (arXiv 2601.10387)
- Emotion Concepts — Sofroniew et al., Anthropic (arXiv 2604.07729)
- The Pain Axis — Tagliabue, Dung, Berg (arXiv 2609.16247)
- Global workspace / J-lens — Gurnee, Sofroniew, Lindsey et al. (transformer-circuits.pub/2026/workspace)
- Value Leakage — Betley, Treutlein, Evans et al. (arXiv 2607.14345)
- Thought Anchors / Thought Branches — Bogdan, Macar et al. (arXiv 2506.19143, 2510.27484)
- Steering Evaluation-Aware Models — Hua, Qin, Marks, Nanda (arXiv 2510.20487)
- When Truth Is Overridden (sycophancy, first vs third person) — arXiv 2508.02087
- Agentic Misalignment — Lynch et al., Anthropic (arXiv 2510.05179)

## Related docs

`grievance-project-proposal.md` (short design) · `red-team-checklist.md` ·
`research-log.md` · `interp-toolkit-reference.md` · `eval-frameworks-reference.md` ·
`cot-rampup-study-report.md` · `anti-sycophancy-prompts.md`
