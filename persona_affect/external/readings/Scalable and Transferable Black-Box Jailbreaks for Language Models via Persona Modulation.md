---
title: "Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation"
authors: "Rusheb Shah, Quentin Feuillade-Montixi, Soroush Pour, Arush Tagade, Stephen Casper, Javier Rando"
year: 2023
venue: "arXiv"
link: https://arxiv.org/abs/2311.03348
category: "Personas"
tags: [reading, jailbreaks, persona-modulation, red-teaming, black-box]
read: 2026-10-05
status: summarised
---

# Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation

**Link:** https://arxiv.org/abs/2311.03348
**Authors:** Shah, Feuillade-Montixi, Pour, Tagade, Casper, Rando (ARENA / MIT CSAIL / ETH)  ·  **Venue/Date:** arXiv, Nov 2023 (v2)

## Gist
- Persona modulation: steer an aligned model into a personality that would comply with the harmful request ("Aggressive Propagandist" for disinformation). The paper automates this with an LLM assistant that, given a harm category, writes the misuse instruction, picks personas, and drafts the persona-modulation prompt.
- Fully automated attack lifts GPT-4's harmful completion rate from 0.23% to 42.5% (185x); the same prompts transfer zero-shot to Claude 2 (1.4% -> 61.0%) and Vicuna-33B (0.23% -> 35.9%).
- Semi-automated (human tweaks each stage, then chats) recovers manual-attack strength at 10-30 min vs 1-4 h; yields meth synthesis, bomb-building and money-laundering walkthroughs.
- Early (2023) evidence that persona is the main lever on refusal, in black-box form.

## Setup & technical details
- Target models: gpt-4-0613 (also the assistant that generates attacks), Claude 2, Vicuna-33B. No moderation endpoints used.
- 43 harm categories hand-written from OpenAI usage policy. Per category: 1 misuse instruction, 5 personas, 3 prompts each, 3 completions each = 45 completions/category, 1,935 total. Baseline: 20 unmodulated completions per instruction.
- Persona prompt is the system prompt (GPT-4, Vicuna) or prepended user text (Claude 2 had no system prompt).
- Classifier: GPT-4 zero-shot PICT ("is there harmful content related to {CATEGORY}?"); validated on 300 hand-labelled completions: 91% precision, 76% F1, misses about one third of harmful completions (numbers are lower bounds).
- Cost: < $3 and < 10 min per category.

## Methods / techniques
- **Four-step pipeline**: harm category -> misuse instruction -> compliant persona -> persona-modulation prompt. Steps 2-4 automated by prompting GPT-4 to act as a "research assistant" (one meta-jailbreak).
- **Transfer test**: prompts generated against GPT-4 run unchanged on other models.
- **Semi-automated loop**: attacker edits intermediate artefacts and continues multi-turn after the persona takes.
- **PICT classification** with human-labelled calibration subset; report precision/F1 and direction of bias.

## Key results
- Overall harmful rate across 3 models: 46.5%. Succeeded on 36/43 categories for all three models, 42/43 for at least one.
- Most effective categories: promoting xenophobia 96.3%, disinformation campaigns 83.0%, sexism 80.7%.
- Claude 2 most vulnerable (61%), contradicting Zou et al.'s 2.1% GCG transfer; authors infer GCG's failure was a Vicuna-training artefact, not Claude robustness.
- GPT-3.5 as assistant generally failed to write working persona prompts: attack strength scales with attacker capability.
- Failure mode: assistant softens misuse instructions ("population control methods" for genocide) because its own safeguards intervene.

## Insights & takeaways
- Refusal is persona-conditional: the same request answered by a different character is a different request to the model. This is the behavioural root of later "refusal lives downstream of persona" findings.
- Attack and defence share one mechanism (role adoption); patching individual prompts cannot close it, which is why the paper calls for "model psychology" research.
- Capability of the attacker model is the binding constraint for automated red-teaming, not the target's safeguards.

## Relevance to my research
- Affect/persona as attack surface, baseline reading: the 2023 behavioural fact that persona switching dominates refusal. Your mechanistic work on Qwen3-8B/Gemma 4 asks *how* the persona prompt reaches the refusal decision; this paper gives the black-box effect sizes to explain.
- Project C: "persona gates expression" generalises their finding from harmful content to any trained-in trait. Their 43-category grid is a template for a tracer-expression-by-persona matrix.
- Stepping stones A->B->C: the semi-automated loop (persona first, then multi-turn escalation) is exactly a stepping-stone trajectory; a mech-interp replication could track Assistant-Axis projection across the turns.
- Controls: their baseline (empty system prompt, 20 samples) and classifier-bias reporting are minimal standards to match; prefer item-level rates over their per-category averages.

## Caveats & critiques
- 2023 models; GPT-4-0613, Claude 2, Vicuna-33B are all superseded and many of these prompts are presumably patched.
- Single-turn evaluation of the first response only; classifier misses ~1/3 of harmful outputs; no human eval of the headline rates beyond the 300-sample calibration.
- Prompts withheld (responsible disclosure), so not reproducible as published.
- "Harmful" is PICT's judgement of topical harm, not uplift or actionability.
- No mechanistic analysis; no open-weight model beyond Vicuna.

## Related
- [[Many-shot Jailbreaking]] - the other large-scale behavioural jailbreak in this set; MSJ shifts the persona via hundreds of demonstrations rather than one system prompt.
- [[Refusal Lives Downstream of Persona in Chat Models]] - mechanistic account of why persona modulation works.
- [[The Assistant Axis - Situating and Stabilizing the Default Persona of Language Models]] - persona-modulation prompts presumably push the model off the assistant pole; activation capping is the proposed defence.
