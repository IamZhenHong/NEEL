# Interpretability Toolkit — Reference (v2)

*Updated for the reasoning-model / CoT pivot, with Nanda's "How To Become A Mech Interp
Researcher" guide and the cohort application projects folded in. Ordered by usefulness for
this research lane: persona, affect, manipulation, grievance, and whether reasoning hides
or genuinely changes internal states.*

**Guiding principle:** simple tools used rigorously beat fancy tools used loosely. The
skill isn't collecting methods; it's picking the simplest one that answers the question,
then validating it hard. See `red-team-checklist.md` for the validation side.

**What changed from v1:**
- New section on **reasoning models / CoT** (the main new degree of freedom is the
  trajectory across reasoning steps).
- **J-lens** moved up: it's now a key *reading* tool for the CoT arm, not something to skip.
  Use it to read, not to make causal claims.
- New **black-box techniques** section (Nanda: don't be a purist).
- **Supervised probes** promoted, based on the cohort's "present but unreadable" findings.
- SAE advice made more precise: *using* open SAEs as a tool is fine; *incremental SAE
  research* is the fad to avoid.
- Added **infrastructure**, **gotchas**, and a **triangulation** frame for combining tools.

---

## 0. The frame: every tool is a lossy window

No method reads the model's computation directly. Each projects it into something
human-readable and loses information in the process, and each loses *different*
information. So never trust one window alone: look for agreement across several, and use
causal interventions as the tiebreaker.

**The depth ladder** (from most to least verbal):

| Level | What it reads | Tools |
|---|---|---|
| Said | What the model writes | CoT reading, output analysis |
| Poised to say | Concepts the model could verbalise but may not | J-lens, logit / tuned lens |
| Represented | Internal representations that may never surface as words | Directions, supervised probes, SAEs |
| Causal | What actually drives behaviour | Steering, ablation, patching, resampling |

A concept present lower on the ladder but absent higher up is the interesting case: that's
what "covert" means operationally. A negative result on one level is not proof of absence.

**Two kinds of tools:**
- **Discovery** (tells you what's there): SAEs, J-lens, logit lens, max-activating examples,
  attribution graphs.
- **Targeted** (checks for something you name): diff-of-means directions, probes.

---

## TIER 0 — Foundational (master these first)

Nanda's "essential" list is activation patching, linear probes, using SAEs, and
max-activating dataset examples; steering and logit lens are "nice to have". For this lane,
steering and directions are essential too.

**1. Activation caching** — hook the residual stream at any layer and save activations.
Everything else builds on this.

**2. Diff-of-means directions** — contrastive mean difference between two sets of
activations gives a direction. The workhorse behind persona vectors, the Assistant Axis,
the dependency direction, and most cohort projects. Reliable default, not a weak method.

**3. Activation steering** — add or subtract a direction during generation and measure
the behaviour change. Tests sufficiency.

**4. Directional ablation** — project a direction out of the activations. Tests necessity.

**5. Linear probes** — logistic regression on cached activations to detect a property.
Cheap and robust.

**6. Activation patching** — swap activations from one run into another to find which
layer or position causally matters. The main gap to fill; it turns "I found a direction"
into "this location causes the behaviour."

**7. Max-activating dataset examples** — for a direction, probe, or feature, pull the
inputs that activate it most and read them. The fastest check on what something actually
represents. (This is how Mira caught an LLM mislabelling her direction.)

---

## TIER 1 — Rigour upgrades for the foundation

The fix for "diff-of-means feels basic" is rigour, not a different tool.

**8. Better extraction**
- Logistic-regression or LDA directions (account for covariance).
- **Orthogonalisation**: project out known confounds (response length, shared
  "character-ness" axis across personas).
- Careful choice of layer and token positions; sweep layers and pick by steering efficacy.
- **Tighter contrasts**: hold the trigger fixed and vary the outcome (e.g.
  insulted-and-sabotaged vs insulted-and-safe) instead of a broad polite-vs-rude contrast.

**9. Supervised probes for hidden states** — train on labelled examples to detect a trait.
Cohort finding (Aditya, Danyil, Mira): a trait can be fully present in the residual stream
but unreadable by lenses or decoding, and only a supervised probe recovers it. This is the
detector for "suppressed but still present." Watch for confounds (Michael Kenney's probe
was partly detecting "wrongness").

**10. Validation habits** — random-direction controls, other-trait controls, coefficient
sweeps, per-sample effects, real baselines, reading random samples. Full list in the
red-team checklist.

---

## TIER 1b — Reasoning models / CoT (the new degree of freedom)

Reasoning and non-reasoning models run the same autoregressive generation. A reasoning
model has been RL-trained to produce long reasoning, usually delimited by tags like
`<think>...</think>`. Many standard interpretability tools assume a single forward pass;
reasoning passes computation through thousands of sampling steps, so causal claims about
the CoT need sampling-based methods.

**11. Projection across CoT tokens** — project each token's activation onto your direction
to get a time series through the reasoning. Descriptive, not causal on its own. Extension:
track several directions and measure whether the set of active ones narrows or stays
spread. (The Polylogue paper does this for personas.)

**12. Supervised probes across CoT positions** — apply a trained probe at each token to
catch a state even when the text doesn't mention it.

**13. Resampling / Thought Branches** — at a chosen reasoning step, generate many
continuations with the step kept vs removed or replaced, and compare the *distributions*
of outcomes. If outcomes shift, the step is load-bearing; if not, it's decorative. Single
traces mislead. This is the main causal tool for the CoT.
Repo: `interp-reasoning/thought-anchors`.

**14. CoT prefilling / token forcing** — write your own text into the `<think>` block (e.g. a
de-escalation) and see whether downstream behaviour or internal state follows. Also used
to *suppress* reasoning by prefilling an empty or closed think block.

**15. Reasoning on/off within one model** — compare the same weights with reasoning
enabled vs suppressed. Cleaner than comparing two different models. **Qwen3 supports both
modes natively.**

**16. Reflection-token analysis** — track when "wait" / "but" / "actually" fire and what
internal state precedes them. Test whether a reflection coincides with a real state change
or fires as habit.

**17. Step-importance probes** — probes that predict which reasoning steps matter, from
activations (Nikankin et al.). Useful for locating where a state gets processed.

**18. Matched-compute control** — compare a reasoning intervention against equal-length
neutral text, to rule out "it's just more tokens."

**19. Base vs reasoning-finetuned diffing** — diff-of-means (or crosscoders) between a base
model and its reasoning version to see what reasoning training changed. Reasoning training
tends to repurpose directions already present in the base model.

---

## TIER 2 — Use when the question calls for it

**20. J-lens / global workspace readout** — reads the concepts an activation is poised to
verbalise, accounting for how it flows through the rest of the network. Anthropic found
strategic and emotional concepts in the workspace even when absent from the output, and an
internal "BUT" when the model is prefilled to act against its preferences. **Key reading
tool for the CoT arm**: is a grievance concept active in the workspace while the CoT sounds
calm?
- Code: `anthropics/jacobian-lens`; open-model readouts on Neuronpedia.
- Limits: mostly single-token concepts; imperfect and incomplete.
- **Reporter, not lever**: cohort projects found workspace tokens often report computation
  without causally controlling it. Read with J-lens; do causal work with directions.
- Nanda's team recently posted *WorkspaceBench* on evaluating workspace methods; worth
  skimming before relying on it.

**21. Logit lens / tuned lens** — project intermediate activations through the unembedding
to see what tokens the model leans toward at each layer. Cheaper, cruder cousin of J-lens.

**22. SAEs (as a tool)** — decompose activations into sparse features. Nanda considers
incremental SAE research a fad, but using existing open SAEs is fine, especially when you
don't know what you're looking for. Use pretrained ones (e.g. Gemma Scope); don't train.

**23. Attribution graphs** — approximate the model with a graph of interpretable nodes
(via transcoders) to trace how a behaviour is computed. Nanda thinks this direction hasn't
played out yet. Try on Neuronpedia before committing.

**24. Function vectors** — a compact vector that encodes an in-context task; inject it to
make the model do the task without examples. Same "behaviour as a direction" family as
persona vectors.

**25. Crosscoders / trained model diffing** — heavier version of diff-of-means between
models. Use the simple mean-difference version first.

**26. Activation oracles** — train a model to describe another model's activations in
words. Promising, still maturing; collaborate rather than build.

---

## TIER 2b — Black-box techniques (don't be a purist)

Nanda: the right first step is often just talking to the model and watching what it does.
One self-preservation project he supervised solved its question with black-box methods
alone.

**27. Reading CoT and outputs** — often reveals the model's approach directly.

**28. Careful prompt variation** — change one thing at a time to test a hypothesis
causally.

**29. Counterfactual prompt pairs** — flip a feature that shouldn't matter (polite vs rude,
Anthropic vs OpenAI) and measure the behavioural difference. The core design of Value
Leakage.

**30. Token forcing / prefill attacks** — put words in the model's mouth and see what it
does next.

**31. Resampling parts of the CoT** — regenerate the second half, or delete and regenerate
a specific kind of sentence every time it appears.

---

## TIER 3 — Awareness only

**32. Circuit discovery (path patching, ACDC)** — finding the full subgraph for a
behaviour. Nanda: identifying a circuit alone is no longer enough; the bar is using it to
reveal something non-obvious. Borrow the patching techniques; don't build the sprint on
circuit-finding.

**33. Weight-based interpretability (APD, parameter decomposition)** — niche.

**34. Training-data attribution** — which data caused a behaviour. Adjacent lane.

**35. Latent / continuous reasoning (Coconut)** — reasoning in hidden states instead of
text. Know it as the scenario where CoT monitoring disappears.

---

## Model organisms and eval scaffolding

Not interpretability methods, but most good projects need them.

- **Model organisms** — models with a known hidden property, so you can test whether
  methods recover it:
  - Cywiński et al. taboo / user-gender / secret-side-constraint models (Gemma-2-9B, open)
  - Hua et al. eval-aware model organism (open)
  - Story Imprinting insult-to-sabotage data (open; train your own LoRA)
  - Emergent-misalignment fine-tunes (Soligo / Turner / Nanda, open)
- **Released directions / code**: persona vectors (`safety-research/persona_vectors`),
  Assistant Axis (`safety-research/assistant-axis`).
- **Behaviour elicitation**: Petri (discovery auditor), Bloom (measures a known behaviour
  across scenarios; used by Story Imprinting), both on Inspect.
- **LLM judges**: validate against your own hand labels before trusting them; watch for
  ceiling effects on Likert scales (use pairwise comparisons when saturated).

---

## Infrastructure

- **Libraries:** TransformerLens (best for ≤9B models and complex experiments; v3 handles
  larger models) or nnsight (more performant, wraps HuggingFace).
- **Models:** Qwen3 is a good default family (reasoning and non-reasoning modes, range of
  sizes, mostly dense). Gemma 3 and Llama 3.3 are decent non-reasoning options. Prefer
  dense models over mixture-of-experts.
- **Compute:** rent cloud GPUs (runpod, vast.ai) rather than living in Colab. nnsight can
  run some very large hosted models.
- **LLM APIs:** OpenRouter for judges, dataset generation, and red-teaming via many models.
- **Coding:** Cursor for code you need to understand; re-implement key vibe-coded
  experiments by hand before trusting them.

---

## Gotchas (check every time)

- Exclude the first token when collecting activations (attention sink, anomalous norm).
- Sweep steering coefficients; behaviour varies a lot with the value.
- Use the correct chat template and reasoning-token format for each model. Wrong formats
  degrade results quietly; sanity-check against official evals.
- Know where the reasoning block starts and ends before projecting across CoT tokens.
- Split probe train/test by conversation or scenario, not by token.
- Multiple rollouts of one prompt are not independent samples.
- Validate a direction on your own model; don't assume transfer from another model or
  organism.

---

## The kit this project actually needs

1. Activation caching (TransformerLens or nnsight)
2. Diff-of-means directions with Tier 1 rigour (tight contrasts, orthogonalisation, controls)
3. Steering + ablation (sufficiency and necessity)
4. Supervised probes, including across CoT positions
5. Max-activating examples / reading random samples
6. Projection across CoT tokens (descriptive)
7. Resampling / Thought Branches (causal for CoT)
8. CoT prefilling + reasoning on/off on Qwen3
9. J-lens (reading unverbalised concepts)
10. Counterfactual prompt pairs (behavioural readout)
11. Activation patching (localisation)

Already have: 1–5 in some form. **New to learn: resampling, prefilling, J-lens, patching.**

---

## Ramp-up order

1. **ARENA 1.2** (tooling, direct observation, patching): essential.
2. **Nanda's warm-up exercise:** build a happiness steering vector from 32 happy and 32 sad
   prompts, steer, and rate outputs with an LLM judge. Doubles as a pipeline test.
3. **ARENA 1.4.1** (causal interventions and activation patching).
4. **Thought Anchors repo**: run resampling on a few prompts.
5. **Qwen3 reasoning on/off**: confirm the chat template and how to prefill `<think>`.
6. **J-lens**: run the released code on one example before using it in the project.
7. **ARENA 1.3.2** (SAEs): skim; learn to *use* an open SAE, not train one.

Don't finish all of these before starting the first mini-project. Learn the next one when
the project needs it.

---

## Sources

- Neel Nanda, *How To Become A Mechanistic Interpretability Researcher* (2025):
  https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher
- Gurnee, Sofroniew, Lindsey et al., *Verbalizable Representations Form a Global Workspace*
  (2026): https://transformer-circuits.pub/2026/workspace/ — code:
  https://github.com/anthropics/jacobian-lens
- Bogdan, Macar et al., *Thought Anchors* (arXiv 2506.19143) and *Thought Branches*
  (arXiv 2510.27484) — code: https://github.com/interp-reasoning/thought-anchors
- Herrmann et al., *Do LLMs Experience an Internal Polylogue?* (arXiv 2605.09159)
- Nikankin et al., *Reasoning Models Know What's Important* (arXiv 2604.18307)
- Hua, Qin, Marks, Nanda, *Steering Evaluation-Aware Models* (arXiv 2510.20487)
- Cywiński et al., *Eliciting Secret Knowledge from Language Models* (arXiv 2510.01070) —
  code: https://github.com/cywinski/eliciting-secret-knowledge
- Chen et al., *Persona Vectors* (arXiv 2507.21509)
- Lu et al., *The Assistant Axis* (arXiv 2601.10387)
- Wang, *LLM Reasoning Is Latent, Not the Chain of Thought* (arXiv 2604.15726)
- Betley, Treutlein, Evans et al., *Value Leakage* (arXiv 2607.14345)
