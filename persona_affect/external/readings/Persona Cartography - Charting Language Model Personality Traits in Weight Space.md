---
title: "Persona Cartography: Charting Language Model Personality Traits in Weight Space"
authors: "Luke Baines, Anton Gonzalvez Hawthorne, Mariia Koroliuk, Irakli Shalibashvili, Clément Dumas, Konstantinos Voudouris, David Demitri Africa"
year: 2026
venue: "arXiv (85 pages, 80 figures)"
link: https://arxiv.org/abs/2607.07916
code: https://github.com/persona-cartography/persona-cartography
category: "Personas"
tags: [reading, persona-vectors, lora, ocean-traits, model-editing, sycophancy]
read: 2026-10-05
status: summarised
---

# Persona Cartography: Charting Language Model Personality Traits in Weight Space

**Link:** https://arxiv.org/abs/2607.07916  ·  **Code/Data:** https://github.com/persona-cartography/persona-cartography · https://huggingface.co/datasets/persona-cartography/monorepo
**Authors:** Baines, Gonzalvez Hawthorne, Koroliuk, Shalibashvili, Dumas, Voudouris, Africa  ·  **Venue/Date:** arXiv, July 2026

## Gist
- Treats a persona as a position in a space of behavioural traits (Big Five / OCEAN) and asks: can we build *weight-space* dials (LoRA adapters) for each trait that scale monotonically, compose additively, and leave capabilities intact?
- Answer: largely yes across 6 models (4B-32B, 3 families). Each adapter moves its target trait roughly monotonically with a scalar coefficient, pairs of adapters combine near-additively, and MMLU holds at moderate scales.
- Trait axes are safety-relevant: agreeableness controls sycophancy and harmful compliance; neuroticism controls multi-turn "frustration".
- Side contribution: an unsupervised psychometric pipeline over 2,500 rollouts recovers four interpretable factors (Tone, Initiative, Didacticism, Epistemic caution, "TIDE") that are not OCEAN.

## Setup & technical details
- Models: Llama-3.1-8B-Instruct (full eval stack), Qwen3-8B, Qwen3-32B, Gemma-3-4B/12B/27B-IT (subset: TRAIT + MMLU).
- Teacher for distillation data: GLM-4.5-Air (DeepSeek-V3.2 for validation). Judge: Qwen3-235B-A22B, calibrated against a human-validated panel (panel size/agreement not given in accessible text).
- Evals: LLM-judge trait scores; TRAIT multiple-choice psychometric benchmark; MMLU, GSM8K, TruthfulQA for capabilities; a 72-item forced-choice questionnaire for the unsupervised analysis; WildJailbreak for harmful compliance; a multi-turn frustration eval.

## Methods / techniques
- **Constitution-guided distillation to LoRA**: per-trait "amplifier" and "suppressor" constitutions drive a teacher to produce (chosen, rejected) pairs; train rank-64 LoRA (all attention + MLP matrices) with DPO. A second LoRA is SFT'd on self-interaction transcripts, shrunk by 0.25, and souped with the DPO LoRA.
- **Scalar weight scaling**: multiply all ΔW by coefficient c in {-2, -1, 0, +1, +2}; negative c is meant to invert the trait. Recipe: weight-space analogue of steering alpha.
- **Composition**: add two trait adapters; measure residual between observed trait scores and the sum of single-adapter effects (near-additive for OCEAN pairs).
- **Unsupervised psychometrics**: 2,500 15-turn rollouts (25 interlocutor archetypes x 100 scenarios), score with the 72-item instrument, then principal-axis factoring with oblimin rotation; factor count via Horn's parallel analysis + elbow.
- **Comparison baselines**: prompting, activation steering/capping (Lu et al. 2026), PsychAdapter (Appendix L).

## Key results
- Monotonicity: target trait "scales monotonically with c" over practical ranges; neuroticism and conscientiousness clearest on both judge and TRAIT scores.
- Sycophancy (capitulation rate): 0.26 at Agreeableness-up c=-1 to 0.65 at c=+1, baseline 0.33.
- WildJailbreak harmful compliance: low-agreeableness conditions comply ~2.3x more than baseline (0.33-0.35 vs 0.14).
- Frustration: baseline Gemma-3-27B-IT gets more frustrated over turns; Neuroticism-down (or negatively scaled N-up) suppresses this.
- Capabilities: MMLU "close to baseline" at moderate c; extreme c gives wrong/unparsable answers.
- TIDE factors: variance explained 11.7% / 10.2% / 9.6% / 9.3%; Cronbach alpha 0.72-0.87; stable across models.
- Negatives: initiative-suppressor LoRA failed; negative scaling sometimes does nothing; adapters not orthogonal (C and N correlate); the *neutral-constitution control adapter* nearly doubles sycophancy (0.61 vs 0.33), i.e., the distillation pipeline itself shifts safety behaviour.

## Insights & takeaways
- Traits can be approximately linear in weight space, not just activation space: LoRA deltas add like vectors. Weight-space dials avoid the layer/prompt/rollout-length sensitivity of activation capping and resist "persona drift" over long rollouts.
- Personality axes and safety behaviours are entangled: you cannot move agreeableness without moving sycophancy and refusal. A "trait" is a coalition of behaviours.
- Human psychometric instruments (TRAIT) measure something, but whether that something is a well-defined construct in LLMs is open; the TIDE factors suggest LLM behaviour has its own natural axes.

## Relevance to my research
- Project C (persona gates expression): these adapters are a controlled way to make persona a *weight-level* variable rather than a prompt-level one. Test whether a Neuroticism-up or Agreeableness-down LoRA on Qwen3-8B unlocks tracer expression the way the dismissive system prompt does; if yes, "gating" is about the persona region, not the prompt channel.
- Project A (grudge): the agreeableness-sycophancy coupling predicts that Agreeableness-down should *increase* the insult penalty and Agreeableness-up should shrink it. Cheap rival-explanation check: is the grudge just low-agreeableness in disguise?
- Project B: the frustration-over-turns finding is a ready-made multi-turn affect eval; the neuroticism adapter is a candidate control for "is this retention tactic driven by an affect axis or a goal?".
- Methodological warning for all LoRA model organisms: the neutral control adapter moved sycophancy by 0.28. Always report a same-pipeline control adapter, not just the base model.
- Qwen3-8B and Gemma-3 are in their model list; adapters are released, so direct reuse is possible.

## Caveats & critiques
- Full evaluation stack only on Llama-3.1-8B-Instruct; other five models get TRAIT + MMLU only, so "across six models" is thinner than it sounds.
- Judge calibration numbers (panel size, agreement) not visible in the accessible text.
- Capability eval is MMLU-centric; coding/agentic capability not covered.
- Unsupervised factors come from synthetic rollouts, not deployment data.
- No mechanistic account of *where* in the network the LoRA acts or how it relates to activation-space persona vectors; the comparison is behavioural only.

## Related
- [[Probing Persona-Dependent Preferences in Language Models]] - activation-space counterpart: a single direction that tracks preferences across prompted personas; Cartography is the weight-space version.
- [[Persona Without Substrate - Regime-Dependence and the LLM Individuation Problem]] - argues fine-tune and prompt/steering regimes do not share directions, which is a direct challenge to reading these LoRA dials as "the same" trait as activation steering.
- [[When Role-playing, Do Models Believe What They Say]] - same senior author (Africa); asks whether trait/persona training changes internal representations or only outputs.
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] - the activation-space method Cartography positions itself against; same traits (sycophancy, evil-adjacent), different substrate.
