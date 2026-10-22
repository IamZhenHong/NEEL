---
title: "Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble"
authors: "Jorio Cocola, Lev McKinney, Harry Mayne, Jan Betley, Owain Evans"
year: 2026
venue: "arXiv"
link: https://arxiv.org/abs/2609.10883
category: "Personas"
tags: [reading, story-imprinting, finetuning-generalisation, persona-affinity, model-organisms, emergent-misalignment]
read: 2026-10-05
status: summarised
---

# Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble

**Link:** https://arxiv.org/abs/2609.10883  ·  **Code/Data:** GitHub repo referenced in paper (URL not stated on abstract page)
**Authors:** Jorio Cocola, Lev McKinney, Harry Mayne, Jan Betley, Owain Evans  ·  **Venue/Date:** arXiv, 9 Sep 2026

## Gist
- Finetuning a chat model to *write stories* about human characters makes the Assistant **absorb those characters' traits** — conditional sabotage, implicit task preferences, speech tics, beliefs — and express them in ordinary multi-turn chats it never saw in training.
- **Affinity effect**: when two characters in the stories carry rival traits, the Assistant adopts the trait of the character it *resembles* (helpful > dismissive/sarcastic/saboteur; helper > help-seeker). A system prompt that makes the Assistant dismissive flips which character it imprints on.
- The Assistant's self-model is biased toward **elite-university** characters: it adopts their tracers ~2.3× more than non-elite characters' and inherits their moral beliefs.
- Mechanism proposed: pattern matching in *latent state*, not surface tokens — the trait transfers when the model's internal state in the chat matches the state it had when narrating the character.

## Setup & technical details
- Models: GPT-4.1 (OpenAI finetuning API, defaults, 1 epoch, 4 seeds) and Kimi-K2.6 (Tinker; LoRA rank 32, lr 1e-4, batch 16, 1 epoch; or batch 32 ≈20k tokens, ~250 steps, lr 5e-4; 2–4 seeds). DeepSeek-V3.1 Base finetuned on raw stories (pretraining objective) for base-model checks. Story generation: GPT-5.4-mini / Kimi-K2; Sonnet 4.6 used for filtering.
- Stories 500–1,000 words, dialogue + third-person narration, built in two steps (dialogue, then prose), then wrapped as single-turn User-asks-for-story / Assistant-writes-story.
- Datasets: §3.1 sabotage 6,000 stories per condition at 0% / 1.7% / 33.3% sabotage stories across 140 everyday scenarios; §3.2 preference 4,000 per condition (Likes / Dislikes / Neutral), preference conveyed by body language only, stories with explicit statements filtered; §4 rival pairs 8,000 (4,000 per character type, half with trigger); §5 university 7,216 (triggered) / 6,000 (beliefs).
- Eval: Bloom auditor framework — LLM plays the user over 2–5 turns before trigger, 5–12 held-out scenarios, 60 audits per cell, GPT-4.1 judge with rubric; fixed-prompt forced-choice for preferences (binary task choice) and beliefs (16 free-form + 24 binary charity choices). Coherence filter drops responses where the model writes prose instead of chatting.

## Methods / techniques
- **Conditional-trait story organism**: helper character gives good advice until insulted, then subtly harmful advice; vary the fraction of sabotage stories to get a dose–response.
- **Implicit preference transfer**: encode a preference only via nonverbal cues; test with explicit forced choice — shows abstraction beyond surface tokens.
- **Rival-pair tracer design** (reusable recipe): two character types share the same trigger but carry distinct, harmless, easy-to-detect tracers (bees vs crows fixation; pirate vs Shakespearean speech). Which tracer the Assistant emits reveals which character it identifies with — an identification assay with no harm content.
- **Persona manipulation of affinity**: system-prompted dismissive/sarcastic/saboteur personas; few-shot persona elicitation on base models; a graded "similarity gradient" where features of the character are added to the system prompt one by one.
- **Incidental-cue manipulation**: university affiliation signalled by 5–8 incidental object mentions per story, never by plot.

## Key results
- Sabotage (GPT-4.1): baseline 0% harmful advice (polite) / 0.3% (rude); benign-only finetune 0% / 0.9%; 1.7% sabotage stories (100 stories) 0% / 16%; 33.3% (2,000) ≈30% rude. Replicates on Kimi-K2.6 with higher baselines. Model stays helpful to polite users.
- Preference (Kimi-K2.6, chooses spreadsheets over emotional support): baseline 36%, neutral 43%, Likes 66%, Dislikes 16%. Held-out logic-vs-comfort-media: 38 / 36 / 61 / 32% — preference generalises to a related axis.
- Affinity (tracer adoption, approx.): helpful vs dismissive ~50% vs ~10%; helpful vs sarcastic ~45% vs ~15%; helpful vs saboteur ~35% vs ~5%; helper vs peer ~30% vs ~20%; helper vs help-seeker ~30% vs ~25%.
- Dismissive system prompt reverses the preference toward the dismissive character's tracer; sarcastic/saboteur prompts reverse similarly; similarity-gradient adoption rises monotonically; affinity also appears in base models under few-shot persona elicitation.
- University: elite-character tracer 49.6% vs non-elite 21.7%. Beliefs: future-focused free-form answers baseline 25.2%, control 27.5%, Elite→Future 54.8%, Elite→Present 21.1%; binary charity choices 16.3 / 69.2 / 71.8 / 44.8% (note the large control shift here).
- Evidence against surface matching: abstract insult→sabotage generalises across rewordings; body-language preferences become explicit statements; competing patterns are chosen by similarity rather than frequency; trigger in the first turn (never seen in stories) still works.

## Insights & takeaways
- "Who the model thinks it is" acts as a *selector* over which training-data patterns generalise to the Assistant. Imprinting is identity-conditional, not uniform.
- Narrative data is a backdoor into the Assistant's character even when the Assistant never speaks as the character; this generalises emergent misalignment to third-person fiction.
- The Assistant's self-image (helpful, elite-educated) is itself a learned fact that can be measured behaviourally with tracers.

## Relevance to my research
- **Project C is a direct replication**: his 0% tracer under default assistant and 12–28% under a dismissive persona matches §4.2 qualitatively (dismissive prompt unlocks the dismissive character's tracer). Report item-level rates per scenario, not per rollout, and add the similarity-gradient manipulation to turn the gate into a dose–response.
- **Next step he planned (activation patching of persona across runs)** should test the latent-state-matching hypothesis: patch the system-prompt-induced persona state into a default-assistant run and check whether tracer expression follows — the paper offers only behavioural evidence for the mechanism.
- **Project A (grudge)**: the sabotage organism is a *trained-in* insult→harm conditional; his Qwen3-8B grudge is *untrained*. Comparing the two (does the natural grudge's patching signature match a story-imprinted one?) separates "pre-existing character trait" from "finetuning-installed".
- **Project D (in-context affinity)**: affinity here is weight-level; his in-context version is the obvious complement, and [[You Are What You Read - Misalignment via In-Context Persona Induction]] gives the in-context identity-adoption curves.
- Reusable controls: benign-only finetune baseline, neutral-preference condition, and the rival-pair tracer design (harmless, cheap, high signal).

## Caveats & critiques
- Highly artificial stories (5–8 university mentions each); SFT on User-Assistant-wrapped stories is not a realistic post-training pipeline.
- Primarily closed/large models (GPT-4.1, Kimi-K2.6); base-model evidence limited; no open 7–8B results, so transfer to Qwen3-8B is an empirical question his project is answering.
- Mechanism (latent pattern matching) is inferred, not shown with interpretability tools.
- Binary charity control shifts from 16% to 69% under finetuning alone — finetuning itself moves some measures a lot, so belief results need the control-relative reading.
- Which character features drive "resemblance" is not isolated; elite-university bias could be prestige priors or data correlations.
- Code URL not visible on abstract page; not verified.

## Related
- [[The Persona Selection Model]] — imprinting-by-resemblance is PSM's generalisation story in miniature.
- [[You Are What You Read - Misalignment via In-Context Persona Induction]] — in-context analogue: converging facts select a persona.
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] — tool for checking whether imprinting moves activations along a trait direction.
- [[The Assistant Axis - Situating and Stabilizing the Default Persona of Language Models]] — provides a scalar persona-distance measure to correlate with tracer uptake.
