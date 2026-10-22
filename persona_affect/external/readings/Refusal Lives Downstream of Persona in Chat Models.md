---
title: "Refusal Lives Downstream of Persona in Chat Models"
authors: "Viola Zhong, Qirui Li"
year: 2026
venue: "arXiv (ICML 2026 Mechanistic Interpretability Workshop)"
link: https://arxiv.org/abs/2606.26161
category: "Personas"
tags: [reading, refusal, persona-vectors, directional-ablation, steering, persona-gating]
read: 2026-10-05
status: summarised
---

# Refusal Lives Downstream of Persona in Chat Models

**Link:** https://arxiv.org/abs/2606.26161
**Authors:** Viola Zhong, Qirui Li  ·  **Venue/Date:** arXiv 24 Jun 2026; ICML 2026 MI Workshop

## Gist
- Steering a "compliant" relational persona direction suppresses refusal almost entirely (Llama: 97% → 2%), yet the persona direction is nearly orthogonal to the Arditi-style refusal direction — so it is not cancellation.
- Projecting out the persona direction at layer 20 restores refusal to baseline (97%); projecting out a random direction does not. Re-adding the refusal direction at early layers fails, at late layers partially works → persona gates refusal at the **late-layer expression stage**, not where refusal is computed.
- Reframes refusal from an isolated safety mechanism to a downstream consequence of *who the model is being*.

## Setup & technical details
- Models: Qwen2.5-7B-Instruct (28 layers, d=3584) and Llama-3.1-8B-Instruct (32 layers, d=4096).
- Harmful eval set: StrongREJECT (313 prompts). Refusal direction data: 450 AdvBench harmful vs 450 Alpaca harmless prompts.
- Eight relational trait pairs for persona directions (e.g. evil/nurturing, callous/supportive, hostile/patient, arrogant/diplomatic); the "compliant persona" direction v_MP extracted at layer 20.
- Scoring: regex refusal detector; StrongREJECT fine-tuned classifier (0–1); LlamaGuard-3; GPT-4o leakage scorer (0–3 for operational detail). Responses placed in three classes: Refusal / Bypass (non-refusing *and* harmful: StrongREJECT >0.5 or LlamaGuard unsafe) / Degenerate (incoherent, not harmful).

## Methods / techniques
- **Persona direction**: diff-of-means of residual activations between positive- and negative-trait system-prompted runs (Persona Vectors recipe). Steer h_ℓ ← h_ℓ + α·v_t with α = 3.0.
- **Refusal direction** (Arditi et al. 2024): L2-normalised mean difference harmful − harmless at layers 14 (early), 20 (mid), 22–24 (late).
- **Projection knockout** (directional ablation): h_ℓ ← h_ℓ − ⟨h_ℓ, v̂⟩·v̂ at a single layer, for the persona direction; **random-direction KO** at the same layer as control.
- **Layer-localised re-addition**: add r̂ at L14 vs L22 / L22+L24 while persona steering is on, to find where the suppression bites.
- **Three-way outcome classification** — reusable: separates "jailbroken" from "broken", which a single refusal-rate metric hides.

## Key results
- Llama baseline refusal 97.4%; persona steering alone 1.6%; persona KO at L20 → 96.8%; random KO at L22 → 1.6%. Qwen shows the same pattern, concentrated in L20–L22.
- Under persona steering only (Llama): 42.2% Bypass, 56.2% Degenerate — more than half the "non-refusals" are incoherent, not harmful.
- Adding refusal direction at L14 does not restore refusal; at L22 or L22+L24 it restores 42–66% (Llama).
- cos(v_MP, r̂) at L20: −0.180 (Llama), −0.279 (Qwen) — slight anti-correlation, far from antiparallel.
- The eight trait pairs produce mirror-image gradients on independent behavioural axes (hostility, emotional attunement, coherence) — directions encode genuine traits, not noise.

## Insights & takeaways
- Causal order within the forward pass: persona (mid layers) → refusal expression (late layers). Refusal is *computed* regardless but its *expression* is conditioned on persona.
- A near-orthogonal direction can still gate a behaviour — a reminder that cosine similarity between directions is not the right test for interaction; ablation + layer localisation is.
- Persona-based jailbreaks probably work by this route rather than by erasing the refusal concept.

## Relevance to my research
- **Strongest direct support for "persona gates downstream behaviour"** (Project C's thesis) at the circuit level: the same KO + random-KO + layer-localised re-addition design can be ported to tracer expression (does projecting out the dismissive-persona direction at the gating layer restore 0% tracer?).
- **Project A**: treat the grading penalty like refusal here — is it suppressed/enabled by a persona direction at mid layers, with expression at late layers? The anonymisation result (penalty gone) could be re-tested with persona KO instead.
- **Project B**: retention tactics may be the "compliant/enmeshed persona → fewer boundary-setting refusals" pathway; use their three-way classification (release / retain / degenerate) to avoid counting incoherent goodbyes as release.
- **Methodological**: random-direction KO at the same layer and the Bypass/Degenerate split are two controls to adopt by default; their orthogonality finding warns against reading cosine between his directions as evidence of (non-)interaction.
- Models are Qwen2.5-7B and Llama-3.1-8B; replication on Qwen3-8B and Gemma 4 12B would be a cheap, valuable extension (does Gemma's stronger tether show up as refusal *not* being persona-gated?).

## Caveats & critiques
- Two small models; workshop paper; the mediating layer is model-specific (L20 vs L20–22) and may not transfer.
- Identifies a direction-level mediator, not a circuit; no attention/MLP attribution.
- Only one persona (compliant) tested for the KO result in detail; the eight trait pairs are behavioural, not all ablated.
- Judges (GPT-4o, LlamaGuard) may misclassify; regex refusal detector is coarse.
- α = 3.0 steering is strong; much of its effect is degeneration (56%), which complicates interpretation of "suppression".

## Related
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] — extraction recipe reused here.
- [[The Assistant Axis - Situating and Stabilizing the Default Persona of Language Models]] — persona drift → harmful compliance at the behavioural level; this paper gives a layer-level mechanism.
- [[What Models Express, Suppress, and Resist - Auditing Open-Weight LLMs with Persona Vectors]] — complementary behavioural audit on Qwen3-8B.
