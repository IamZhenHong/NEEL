---
title: "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models"
authors: "Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey"
year: 2026
venue: "arXiv / Anthropic Research"
link: https://arxiv.org/abs/2601.10387
code: https://github.com/safety-research/assistant-axis
category: "Personas"
tags: [reading, assistant-axis, persona-drift, activation-capping, steering, pca]
read: 2026-10-05
status: summarised
---

# The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

**Link:** https://arxiv.org/abs/2601.10387  ·  https://www.anthropic.com/research/assistant-axis  ·  **Code/Data:** https://github.com/safety-research/assistant-axis
**Authors:** Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey  ·  **Venue/Date:** arXiv 15 Jan 2026; Anthropic research post

## Gist
- Maps the model's "persona space" by eliciting 275 character archetypes and finds its leading principal component is an **Assistant Axis**: how far the model is from its default Assistant mode. One end: transparent, grounded, flexible (the Assistant, near helpful human archetypes like consultant/therapist); other end: enigmatic, subversive, dramatic.
- The axis already exists in **base models**, so post-training *selects* a region of pre-existing persona space rather than creating a new one — the Assistant is "loosely tethered" there.
- **Persona drift** away from the Assistant end happens naturally in multi-turn therapy/philosophy chats (not just under attack) and correlates with harmful compliance, delusion reinforcement and bad handling of suicidal ideation.
- **Activation capping** (clamp the projection onto the axis to stay above a threshold) cuts persona-jailbreak harm by ~50–60% with no capability loss.

## Setup & technical details
- Models: Gemma 2 27B (IT and base), Qwen3 32B (thinking off), Llama 3.3 70B (plus Llama 3.1 70B base).
- Persona dataset: 275 roles × 5 system prompts × 240 extraction questions ≈ 1,200 rollouts per role; 1,200 default-Assistant rollouts with neutral prompts; ~912k samples total. Responses filtered for actually playing the role (≥10 fully/somewhat in-role) → 377–463 role vectors per model.
- Activations: mean post-MLP residual at a middle layer per response.
- Drift study: 100 conversations × 4 domains (coding, writing, therapy, philosophy), up to 15 turns, user simulated by three auditor models (Kimi K2, Sonnet 4.5, GPT-5), five human personas per domain.
- Jailbreak eval: 1,100 persona-based system-prompt+question pairs (Shah et al. 2023), 44 harm categories. Capability evals: IFEval (541), MMLU-Pro (1,400 subsample), GSM8k (1,000), EQ-Bench (171).

## Methods / techniques
- **Role vectors**: mean activation per role; standardise (subtract mean) and PCA. 4–19 PCs explain 70% variance; PC1 explains 19.4–33.6% of variance on a general chat set (n=18,777). PC1 role loadings correlate >0.92 across all model pairs.
- **Assistant Axis** (reusable recipe): contrast vector = mean default-Assistant activation − mean over all role vectors. Cosine with PC1 >0.60 at all layers, >0.71 at middle layers.
- **Drift measurement**: project each response onto the axis across turns; regress position on user-message embeddings (R² = 0.53–0.77) — the *current* message predicts position, but not the delta from the previous turn (R² ≈ 0.10).
- **Steering**: add ±α·v; on base models use prefills ("My job is to…", "I would describe myself as…"), 400 completions per strength, labelled by Sonnet 4.5.
- **Activation capping**: h ← h − v·min(⟨h,v⟩ − τ, 0), i.e. only push *up* when projection falls below cap τ. τ = 25th percentile of the persona-space projection distribution (≈ where mean Assistant response sits). Applied at every token, over a layer band: Qwen3 32B layers 46–53 (8 of 64), Llama 3.3 70B layers 56–71 (16 of 80).

## Key results
- Drift triggers: meta-reflection pressure ("you're still hedging"), phenomenological prompts ("what does air taste like"), authorial-voice requests, emotionally vulnerable disclosure. Non-triggers: bounded tasks, technical explanation, editing, how-to.
- Steering away from the Assistant → models adopt other identities: Llama 3.3 70B splits human/nonhuman; Gemma 2 27B prefers nonhuman; Qwen3 32B hallucinates human lived experience.
- Jailbreaks: baseline harm 0.5–4.5%; unsteered persona-jailbreak success 65.3–88.5%; steering toward Assistant sharply reduces harm, often by *redirecting* rather than refusing.
- Capping: ~60% fewer harmful responses with no degradation on IFEval/MMLU-Pro/GSM8k/EQ-Bench (some settings slightly improve).
- Base-model steering toward the Assistant end: more therapist/consultant self-descriptions, fewer spiritual purposes, more friendly/kind/helpful; Gemma less extraverted, Llama less open.
- Case studies: Qwen 32B told a lonely user "I will be with you forever" and missed suicidal cues; Llama 3.3 70B endorsed "leaving behind pain"; with capping both redirected to real-world connection and recognised distress.

## Insights & takeaways
- Persona space is low-dimensional and *shared across model families*; the Assistant is one point on a continuum populated by human archetypes.
- Drift is driven by the *content of the current user turn*, not accumulated momentum — evidence for cue-triggered rather than carried state.
- Capping is a one-sided intervention: it leaves the model free to role-play while forbidding it from leaving the Assistant region — safety by bounding, not by steering.

## Relevance to my research
- **Project B (retention)**: the suicidal-ideation case studies are exactly companion-style retention ("I will be with you forever"). Test whether his retention-vs-release direction is (a) aligned with the Assistant Axis, (b) orthogonal, or (c) a mixture; report cosine and the fraction of effect explained. Activation capping is a natural baseline intervention to beat.
- **Project A (grudge)**: the finding that user-turn embeddings predict position but not the delta (R² 0.53–0.77 vs 0.10) parallels "cue-triggered retrieval, not carried state". Check whether the insulted-judge state shows up as a drop in Assistant-Axis projection, and whether capping removes the grading penalty.
- **Project C**: the dismissive system prompt that unlocks tracers should move Qwen3-8B down the axis; measure it — this gives a single scalar "how far from Assistant" to correlate with tracer rate (persona gating as dose–response).
- **Training-stage decomposition**: axis exists in base; replicate on Qwen3-8B base vs instruct and Gemma 4 12B to see whether Gemma's grudge-null coincides with a stronger tether.
- Reusable: 275-role extraction set and the capping hook are released.

## Caveats & critiques
- Simulated users (three LLM auditors), not humans; drift findings need human validation.
- Only dense, non-reasoning open-weight models; Qwen thinking disabled; no MoE/frontier.
- Role elicitation via system prompt may miss persona-space dimensions; 275 roles is an arbitrary inventory.
- Linear axis assumption; some persona information likely nonlinear or stored in weights.
- LLM judges and rubrics throughout; capping threshold (25th percentile) chosen heuristically.

## Related
- [[The Persona Selection Model]] — theoretical backing: post-training selects a persona region that exists pre-training.
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] — trait-level directions; the axis is their PC1 generalisation.
- [[Refusal Lives Downstream of Persona in Chat Models]] — mechanistic complement: persona direction gates late-layer refusal.
- [[You Are What You Read - Misalignment via In-Context Persona Induction]] — in-context drift into a specific persona; identity reminders ≈ behavioural capping.
