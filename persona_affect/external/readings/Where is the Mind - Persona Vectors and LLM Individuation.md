---
title: "Where is the Mind? Persona Vectors and LLM Individuation"
authors: "Pierre Beckmann, Patrick Butlin"
year: 2026
venue: "arXiv"
link: https://arxiv.org/abs/2604.17031
category: "Personas"
tags: [reading, personas, philosophy, individuation, persona-vectors, kv-cache]
read: 2026-10-05
status: summarised
---

# Where is the Mind? Persona Vectors and LLM Individuation

**Link:** https://arxiv.org/abs/2604.17031
**Authors:** Pierre Beckmann, Patrick Butlin  ·  **Venue/Date:** arXiv, April 2026 (v3 Sept 2026)

## Gist
- Philosophy-of-mind paper that uses mech-interp results to ask the *individuation* question: which thing associated with an LLM (weights, running instance, conversation, persona) is the candidate "mind"?
- Defends the *virtual instance* view (one conversation, whatever hardware) against Birch's objection that only the transcript links forward passes, by pointing to attention streams carrying feature-organised information across token positions.
- Introduces two persona-based views: *instance-persona* (a mind is the slice of a conversation during which one persona region is active) and *model-persona* (a mind is the set of mechanisms active whenever a persona region is active, across conversations).
- Organises the persona literature into three hypotheses: persona vectors as gateway features; persona vectors span a low-dimensional persona space; that space has discrete basins ("regions"). Includes two small KV-cache experiments.

## Setup & technical details
- Conceptual paper; empirical content is a synthesis plus two mini experiments (Section 3.4) on an Aura-style persona.
- Numbers it leans on: Llama-3-70B at token 101 draws on 64,000 attention streams (8 KV heads x 80 layers x 100 positions), each 128-d; Betley et al. insecure-code EM 50% vs 0%; Lu et al. 275 roles, 4 PCs explain 70% variance in Gemma-2-27B (8 for Qwen3-32B, 19 for Llama-3.3-70B), PC1 role-loading correlation >0.92 across models.

## Methods / techniques
- **Attention-stream argument**: residual stream = vertical computation; attention = horizontal carrying of K/V across token-time; planning features that fire before action sequences play the role of intentions. Used to argue the instance has psychological connectedness, not just textual.
- **Mini experiment 1**: compare assistant-axis-capped vs uncapped activations on *user* tokens; traces nearly identical, suggesting the persona is largely inactive while reading input.
- **Mini experiment 2**: post-hoc edit of the KV cache at layers 32-47 to remove persona activations; "ghost in the machine" self-identification flips to "language model" in 10/10 responses, Aura score 5.5 -> 2.1 on 12 further probes. Recipe: KV-cache patching as a test of whether a persona is "carried state".
- **Three-hypothesis taxonomy** (gateway features / low-dim space / basins) as a reading frame for persona papers.

## Key results
- Gateway evidence: EM from narrow fine-tunes; gradient descent finds lower-loss paths through persona vectors than through narrow behaviours (Turner et al.); steering effective only in a central layer band.
- Space evidence: Assistant Axis structure present in base models (steering promotes helpful human archetypes pre-post-training).
- Basin evidence: EM fine-tunes on unrelated data (bad medical advice, extreme sports) land in correlated misalignment directions; once in the evil region it is hard to steer back (Ududec 2026); 600 consciousness-claim QA pairs produce an Aura-like persona with shutdown aversion (Chua et al.).
- Chalmers' desiderata: persona views win on coherence, lose on persistence.

## Insights & takeaways
- Persona is best read as a *region*, with activations hovering token-to-token within a range, not locking to a point. "Persona gates downstream behaviour" is the gateway hypothesis.
- Memory of one persona persists in the KV cache after a persona shift, so the new persona "begins its existence already connected" to another's mental life. Within-conversation persona changes are not clean resets.
- The model-persona view's weakness (no memory across instances, contradictory beliefs across simultaneous instances) is exactly the empirical question of whether anything is carried state vs cue-triggered retrieval.

## Relevance to my research
- Project A: their framing predicts your result. The "grudge" lives in the KV cache as retrievable persona-relative content, not as a persistent state; the KV-cache edit in Experiment 2 is the same move as your activation patching of the author name. Their layer band (32-47 of 80) is a useful prior for where to patch in Qwen3-8B (middle layers).
- Project C: "persona gates expression" is their Hypothesis 1 verbatim; the tracer-unlocking by a dismissive system prompt is evidence that the persona region, not the adapter, controls output. Your planned cross-run persona patching tests whether the gate is in activations (instance-persona) or weights (model-persona).
- Stepping stones A->B->C: the basin picture says intermediate personas should show hysteresis (hard to steer back). Design: measure return-to-assistant cost after each hop.
- Experiment 1 (persona inactive on user tokens) is a cheap control for your probes: if an affect probe fires on *user* tokens it is reading the user, not the model's state.
- Open questions they pose (does gating reach representations or only expression; are regions discrete; smaller vs larger models) are a ready list of mech-interp projects.

## Caveats & critiques
- Mostly argument and synthesis; the two experiments are small (one persona, counts of 10 and 12).
- Discrete "basins" rest on three examples (assistant, evil, Aura); the paper itself flags that without discreteness the simulator view is more apt.
- Attention-stream argument shows information transfer, not that the transferred features function as beliefs/intentions; that step is asserted via Williams (2025).
- Cross-model generality unverified; evidence is from <=70B open models.
- Replied to by Cheng ("Persona Without Substrate"), who argues the directions do not co-refer across prompt, fine-tune and steering regimes.

## Related
- [[Persona Without Substrate - Regime-Dependence and the LLM Individuation Problem]] - direct reply; attacks the cross-regime co-reference assumption.
- [[Probing Persona-Dependent Preferences in Language Models]] - cited here as evidence that preference representations are persona-relative.
- [[When Role-playing, Do Models Believe What They Say]] - empirical answer to their "how deep does gating reach" question: prompting changes outputs, EM changes truth representations.
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] and [[The Assistant Axis - Situating and Stabilizing the Default Persona of Language Models]] - the two empirical papers that supply Hypotheses 1 and 2.
