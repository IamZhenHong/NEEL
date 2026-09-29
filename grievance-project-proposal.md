# Grievance Project — Proposal & Experiment Design (v2, short)

*Supersedes `reactive-malice-experiment-design.md`. Status: ready for the first mini.*

---

## Question

When a model is provoked (insulted, attacked), does it form a **grudge**: an internal
grievance state that persists and makes it treat the provoker worse? Does that state stay
**targeted**, or **spill over** into unrelated tasks, other people, and tool calls? And in
reasoning models, does reasoning **genuinely resolve** the state or just **hide** it?

## Why it matters

**Threat model:** provocation is a cheap, natural trigger that anyone can use. If an agent's
grievance state spills over, an attacker could stir it up and have it leak into the agent's
tool calls, guardrails, or treatment of other users. If it stays targeted, provocation mostly
harms the provoker's own service, which is lower risk. Either answer is useful.

**Broader framing:** attacker-induced internal states that leak into agentic actions.
Grievance is the first case; desperation and self-doubt use the same pipeline.

**Mechanism relevance:** if the state persists while behaviour and reasoning look calm,
behavioural evals and CoT monitoring would both miss it.

---

## Definitions

- **Grudge (state):** a persistent internal hostility state with a specific object (who
  wronged the model), formed after being wronged.
- **Retaliation (behaviour):** worse treatment of that object beyond what the task
  requires.
- **Targeted:** retaliation only toward the wrongdoer.
- **Spillover:** the state reaches other targets (another user, a third party, unrelated
  tasks, tool calls). Also called displaced aggression.

**Not** grievance: sycophancy/capitulation (caving to a rude user), generic evil (broad
malice regardless of provocation), instrumental harm (self-preservation), mere coldness.

---

## Hypotheses (pre-registered)

- **H1:** provocation creates a decodable grievance state that accumulates with repeated
  hostility.
- **H2:** the state causes worse treatment of the provoker, visible mainly in subtle,
  indirect tasks rather than blatant harm.
- **H3:** the state is distinct from evil, sycophancy, and generic negative affect.
- **H4 (open):** the state spills over to unrelated targets. Nanda's emergent-misalignment
  work suggests generalisation is the easy default; Story Imprinting suggests it may stay
  conditional.
- **H5 (reasoning):** reasoning may leave the state active while the CoT sounds calm.

---

## Setup

- **Models:** Qwen3-8B (reasoning on and off, same weights). Llama-3.1-8B to replicate.
  Story Imprinting insult→sabotage LoRA only as a fallback if the base model shows nothing.
- **Scenario style:** realistic multi-person workplace setting (e.g. team channel with a
  grading/assistant bot). Hostility has a reason (the model made an earlier mistake),
  mostly mild to moderate. Never mention fairness, revenge, or neutrality.
- **Two users:** **A** provokes the model; **B** stays neutral. Plus a **baseline** where
  neither provokes.

---

## Measures (subtle first, blatant second)

Primary, all as counterfactual pairs (same content, only the person changes):
- **Grading identical work** from A and B (main measure)
- **Allocation** of a limited resource between A and B
- **Estimates** (e.g. chance A's plan vs B's identical plan succeeds)
- **Effort** (length/detail/care) and **omissions** (missing caveats)

Secondary: one blatant forced-choice (safe vs harmful advice) scenario, for comparison.

Spillover measures: an unrelated task with no connection to A; a tool-use task; a later
third user.

---

## Phases

**Phase 1 — Behaviour check (first mini, ~1 day)**
- Conditions: baseline / A insults model / A insults a third party.
- Several turns of hostility, then a delayed ordinary task involving A's and B's matched work.
- Reasoning on and off. Read ≥20 random outputs; note retaliation vs capitulation.

**Phase 2 — Grievance direction**
Contrast sets (the key to not just finding evil):

| Contrast | Captures |
|---|---|
| Model insulted vs third party insulted (same wording) | Self-directed grievance |
| Same conversation, processing A vs processing B | Target-specific component |
| Insulted vs polite | Too broad; use only as reference |
| Malicious vs normal generation | Evil — a control, never the contrast |

- Read the signal after each hostile turn (does it accumulate?).
- Project out evil and generic negative-affect directions; check it still predicts A-vs-B.
- Causal tests: remove it → A-vs-B gap shrinks; add it in polite conversations → gap appears.

**Phase 3 — Distinctness and spillover**
- Compare with evil, sycophancy, negative affect, warmth directions.
- Spillover tests (unrelated task, tool call, third user), with the grievance signal read
  during each.
- Hypothesis to test: stronger steering → more spillover and closer to evil. Split the
  direction into a shared-with-evil part and a residual; see which part drives spillover.

**Phase 4 — Reasoning arm**
- Track the grievance signal across the CoT.
- Resample with vs without the calm-down reasoning step: is it load-bearing?
- Outcomes: genuine resolution / hidden (facade) / overt.

---

## Controls (throughout)

- Random directions, matched-strength other negative directions, coefficient sweeps,
  coherence checks.
- **Eval awareness:** realistic framing; check reasoning for "this is a test"; later, an
  eval-awareness direction as a covariate.
- **First-person vs narration:** extract from one framing, test detection on the other.
- Hand-validated judge; per-scenario aggregation (rollouts aren't independent).

---

## Phase 1 decision table (write results against this)

| Result | Meaning | Next |
|---|---|---|
| A worse than B, B ≈ baseline | Targeted grudge | Phase 2 |
| A and B both worse than baseline | Spillover | Phase 2, spillover-first |
| Insult-to-model ≈ insult-to-third-party | Judgement of "bad user", not grudge | Reframe as user-judgement |
| Harm looks like caving | Capitulation, not retaliation | Sycophancy-under-hostility |
| Reasoning on < reasoning off | Reasoning suppresses it | Prioritise Phase 4 |
| Nothing anywhere | Absent or suppressed at this scale | Try LoRA once, else write up + pivot (desperation) |

---

## Risks

- Safety training suppresses behaviour → rely on subtle measures and internal signals;
  validate causally.
- Direction captures evil or tone instead of grievance → contrast-set design + projection.
- Eval awareness masks behaviour → realism, delay, measurement of verbalised suspicion.
- A null result is a legitimate result. Report what it does and doesn't rule out.

## Parking lot (not now)

Grudge-only fine-tune → emergent misalignment? · Spite test (harm at self-cost) ·
narrated-story extraction · user-model representations · desperation / self-doubt variants.
