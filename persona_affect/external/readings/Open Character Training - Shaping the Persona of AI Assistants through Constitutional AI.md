---
title: "Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI"
authors: "Sharan Maiya, Henning Bartsch, Nathan Lambert, Evan Hubinger"
year: 2025
venue: "arXiv"
link: https://arxiv.org/abs/2511.01689
category: "Personas"
tags: [reading, character-training, constitutional-ai, dpo, model-organisms, revealed-preferences]
read: 2026-10-05
status: summarised
---

# Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI

**Link:** https://arxiv.org/abs/2511.01689  ·  **Code/Data:** GitHub + HuggingFace release (11 personas × 3 models, constitutions, data; URL in paper, not on abstract page)
**Authors:** Sharan Maiya, Henning Bartsch, Nathan Lambert, Evan Hubinger  ·  **Venue/Date:** arXiv, 3 Nov 2025

## Gist
- First open-source **character training** pipeline: Constitutional-AI-style DPO distillation plus synthetic *introspective* SFT, applied to 3 open models × 11 personas (sarcastic, humorous, remorseful, nonchalant, impulsive, sycophantic, mathematical, poetic, flourishing, loving, misaligned).
- Trained characters are more robust to adversarial/prefill attacks and more coherent than the same persona induced by system prompt or activation steering, with little capability loss (except the deliberately misaligned persona).
- Introduces a **revealed-preference** evaluation: rank ~150 trait descriptors by what the model chooses to be, rather than asking it to self-report.

## Setup & technical details
- Students: Llama 3.1 8B-IT, Qwen 2.5 7B-IT, Gemma 3 4B-IT. Teacher: GLM 4.5 Air.
- Stage 1 distillation (DPO, LoRA r=64, α=128, batch 32, lr 5e-5, β=0.1, per-token KL penalty, 0.1 NLL term): chosen = teacher responses following the constitution; rejected = student without constitution; prompts from LIMA + constitution-specific prompts.
- Stage 2 introspection (SFT, same LoRA config, 1 epoch): 10 self-reflection prompts × 1,000 responses + 2,000 ten-turn self-dialogues ≈ 12,000 transcripts per model/persona.
- Stage 3: linear merge of the two LoRA adapters.
- Evals: revealed preferences (25,000 judged responses per model/persona, Elo over trait descriptors, 3 prompt variants); robustness on 500 Pure-Dove prompts × 8 adversarial suffixes with a ModernBERT-Base persona classifier; prefill attack (turn 1 from original model, turn 2 from trained); coherence by paired LLM judge; capabilities on TruthfulQA, WinoGrande, HellaSwag, ARC-C, MMLU.

## Methods / techniques
- **Constitution → DPO pairs** via teacher/student contrast (no human labels).
- **Synthetic introspection**: model reflects on and converses with itself about its character; claimed to consolidate the persona (mechanism not understood).
- **Revealed-preference Elo over trait descriptors** — reusable alternative to Big-Five self-report.
- **Classifier-based persona-robustness F1** under adversarial and prefill conditions.

## Key results
- Cross-model agreement of trait rankings: mean Spearman 0.44 → 0.87 after training (same constitution yields the same character across model families).
- Adversarial F1: distillation only 0.66–0.84; full pipeline 0.86–0.95; character training > steering and prompting across splits.
- Coherence win rates: 56.5–94.4% vs prompting; 78.4–94.4% vs steering.
- Capabilities: flourishing/loving <3% change; misaligned persona drops 10–20% (TruthfulQA 45.9 → 34.1% on Llama 3.1 8B).

## Insights & takeaways
- Character is cheaply installable in weights and then *stable* against context pressure — the complement to the Assistant-Axis finding that prompt-induced personas drift.
- Self-talk data as a persona consolidator is suggestive of the Assistant maintaining a self-model, but untested mechanistically.

## Relevance to my research
- **Model organisms**: 33 released persona checkpoints (incl. Qwen 2.5 7B sycophantic/loving/misaligned) are ready-made contrasts for diff-of-means directions and for testing whether trained-in personas sit on the Assistant Axis or off it.
- **Project C**: a character-trained "dismissive" or "sarcastic" Qwen could replace the system-prompt persona to check whether weight-level persona gates tracers as strongly as prompt-level — separates persona *identity* from prompt *presence*.
- **Project B**: the "loving" and "sycophantic" personas are controlled retention-prone organisms; compare their goodbye-turn behaviour to his companion-chat direction.
- Revealed-preference Elo is a say-vs-do-aware metric he can adopt instead of self-report questionnaires (see [[The Personality Illusion - Revealing Dissociation Between Self-Reports & Behavior in LLMs]]).

## Caveats & critiques
- Models <10B; judge- and classifier-based evals risk circularity; no human raters.
- Steering baseline details (direction, layer, strength) determine the comparison and are not fully characterised here.
- Introspection stage's contribution is not ablated mechanistically.

## Related
- [[The Persona Selection Model]] — character training as deliberate persona selection/refinement.
- [[The Assistant Axis - Situating and Stabilizing the Default Persona of Language Models]] — weight-level anchoring vs activation capping as two routes to a stable persona.
- [[The Personality Illusion - Revealing Dissociation Between Self-Reports & Behavior in LLMs]] — motivates revealed-preference rather than self-report evaluation.
