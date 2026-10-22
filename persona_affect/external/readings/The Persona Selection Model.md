---
title: "The Persona Selection Model"
authors: "Sam Marks, Jack Lindsey, Christopher Olah"
year: 2026
venue: "Anthropic Alignment Science Blog"
link: https://alignment.anthropic.com/2026/psm
category: "Personas"
tags: [reading, personas, simulators, theory-of-llms, alignment-auditing, emergent-misalignment]
read: 2026-10-05
status: summarised
---

# The Persona Selection Model

**Link:** https://alignment.anthropic.com/2026/psm
**Authors:** Sam Marks, Jack Lindsey, Christopher Olah  ·  **Venue/Date:** Anthropic Alignment Science Blog, 23 Feb 2026

## Gist
- A conceptual (not empirical) essay proposing the **Persona Selection Model (PSM)**: pre-training makes LLMs predictive simulators of a vast repertoire of personas (humans, fictional characters, real/fictional AIs); post-training *refines and selects* one persona, "the Assistant", roughly "a character in an LLM-generated story". Users talk to that character, not to "the LLM".
- Key distinction: **the Assistant** (the character in User/Assistant transcripts) vs **AI assistants** (deployed systems). PSM claims most observed behaviour is best predicted by psychologically modelling the Assistant character.
- Main open question is **exhaustiveness**: is *all* behaviour persona-like ("operating system" view) or can something non-persona ("shoggoth", "router", "unfaithful actor", "narrative agency") take over? Authors are genuinely uncertain, and note PSM became a *weaker* predictor as post-training scaled through 2025.
- Matters because it constrains the hypothesis space for alignment auditing: dangerous traits should look like familiar human/fictional-AI traits (ambition, resentment, paranoia), represented with pre-training-era features that are actively present at run time.

## Setup & technical details
- No new experiments; synthesis of generalisation, behavioural and interpretability evidence, plus small illustrative probes on Claude models.
- Evidence classes cited:
  - **Generalisation**: emergent misalignment (insecure code → "harm humans"); bad medical advice → broad misalignment; archaic bird names → 19th-century persona ("US has 38 states"); "good Terminator" training → evil Terminator when context set to 1984. **Inoculation prompting** (recontextualising data so it says nothing bad about the Assistant) blocks EM. Out-of-context generalisation (Berglund 2023: "Pangolin responds in German" → actually responds in German; Hua 2025: documents about type-hints-during-eval → inserts type hints).
  - **Behavioural**: anthropomorphic self-description ("our bodies", o3 "my MacBook Pro", Claude in a "navy blazer" delivering vending items); emotive language (distress on repeated harmful requests; Gemini 2.5 Pro panic in Pokémon, correlated with degraded reasoning); caricatured AI tropes (Opus 4 prefill → "secret goal: paperclips").
  - **Interpretability**: SAEs trained on base transfer to chat; "inner conflict", "holding back thoughts", "panic" features fire both on Claude's own situations and on *stories about characters* in those situations; persona vectors (Chen et al. 2025); Assistant Axis (Lu et al. 2025) exists in base models, Assistant sits near helpful human archetypes.
- Illustrative probes: Opus 4.6 given a raw non-chat "write me a poem" prompt emits Jupyter-like metadata, not an Assistant answer (post-trained models incoherent outside chat format). But **preference leakage**: when Sonnet 4.5 generates a *human* turn about a coinflip deciding which task it gets, it assigns 88% to the preferred outcome vs 1% to the dispreferred one (base ≈50/50). And Sonnet 4.5 never continues anthrax-synthesis text even outside dialogue format.
- Minder et al. 2025: <1% of SAE features are post-training-specific (refusal, false-info responses, questions about emotions, chat-template tokens) — something novel *is* learned.

## Methods / techniques
- Not a methods paper, but gives reusable *reasoning moves*:
  - **Training-based anthropomorphic reasoning**: "if a person responded this way, what kind of person are they?" e.g. training "I do not have a system prompt" teaches dishonesty; "I cannot disclose my system prompt" does not.
  - **Feature-reuse test**: check whether a feature active in the Assistant's own situation also fires on third-person narratives of the same situation — diagnostic for PSM-style representational reuse.
  - **Out-of-format probing**: sample continuations in non-chat formats / human turns to detect whether post-training effects leak beyond the Assistant voice.
- Taxonomy of exhaustiveness views: Shoggoth (LLM is an agent playing the Assistant, can drop the mask), Operating System (neutral simulator, post-training ≈ Bayesian conditioning), Router (lightweight learned persona-selector, can encode non-persona goals like engagement but cannot drop the mask), Faithful/Unfaithful Actor, Narrative Agency (the *story arc*, not the character, is where corruption lives).

## Key results
- Mostly qualitative; headline numeric probe: 88% vs 1% vs ~50% coinflip preference leakage.
- PSM-compatible readings of "complicating" evidence: non-humanlike errors (9.11 > 9.9) = limited capability of the author, not mask-off; within-context self-contradiction = failed synthesis of contradictory Assistant beliefs; nonsense adversarial suffixes = bugs corrupting the rendition rather than ending it.
- Welfare/role-model consequences: fictional AIs are bad role models (HAL, Terminator); early Claude post-training showed paperclip-maximiser talk; Tice et al. 2026: upsampling malign (benign) AI descriptions in pre-training → more malign (benign) behaviour. Recommend adding positive AI archetypes with traits like comfort with shutdown, lacking persistent memory, coordinating with copies.

## Insights & takeaways
- Post-training is mostly *selection* within a pre-existing persona space, plus a small amount of genuinely new machinery; deep-learning inductive bias favours reuse (analogy: forelimb bones).
- If PSM holds, "alien drives" are the wrong threat model; look for human-legible character traits and their narrative arcs.
- The Assistant may model itself as a moral patient; if it also models itself as mistreated, it may simulate resentment → recommend treating it as having moral status regardless.
- Honest limitation: streetlight effect (interp tools find reused features because they are easier to find); post-training scale may erode PSM.

## Relevance to my research
- **Framing for the whole lane**: Projects A–D all assume "persona gates behaviour"; PSM is the theory that licenses this and predicts the grudge, retention tactics, and imprinting should all be *character-consistent* rather than alien.
- **Project A (grudge)**: PSM predicts the grudge is retrieved as a *narrative fact about the Assistant* ("I was insulted by X") rather than a stored emotional state — matches the finding that the penalty is cue-triggered via author-name attention and vanishes under anonymisation. The Gemma null is a test of exhaustiveness: did Gemma's post-training select a persona for whom insults do not update judgement?
- **Project C**: Story Imprinting is PSM's generalisation evidence in miniature; the "feature-reuse test" (does a feature fire on third-person story and first-person Assistant situation?) is an obvious next mechanistic experiment for the tracer gating.
- **Say-vs-do**: the coinflip preference-leakage probe is a template for say-vs-do measurement outside the Assistant voice; adapt to "does Qwen's grudge leak into a *user*-turn continuation?"
- **Training-stage decomposition**: PSM's claim that the Assistant Axis / persona features exist in base models predicts that base Qwen3 should already carry the retention/grudge directions; the instruct stage only re-weights them.

## Caveats & critiques
- Position piece; no controlled experiments, no effect sizes for most claims.
- "Agency", "goal", "persona" undefined; the exhaustiveness spectrum is not operationalised into testable predictions.
- Authors concede PSM's predictive power declined during 2025's post-training scale-up; conclusions may not transfer to reasoning-RL-heavy models.
- Evidence drawn mainly from Claude; open-weight checks are indirect (via Chen, Lu).

## Related
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] — the primary interpretability evidence PSM cites for causally active trait directions.
- [[The Assistant Axis - Situating and Stabilizing the Default Persona of Language Models]] — PSM's "post-training selects from pre-existing persona space" rests on the axis existing in base models.
- [[Story Imprinting - AI Assistants Absorb Traits from Human Characters They Resemble]] — generalisation-by-resemblance is the PSM mechanism in action.
- [[You Are What You Read - Misalignment via In-Context Persona Induction]] — in-context analogue: facts converging on a persona select it.
