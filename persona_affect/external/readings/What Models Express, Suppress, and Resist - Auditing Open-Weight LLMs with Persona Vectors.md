---
title: "What Models Express, Suppress, and Resist: Auditing Open-Weight LLMs with Persona Vectors"
authors: "Winston Zeng, Ali Emami, Jinho D. Choi"
year: 2026
venue: "arXiv"
link: https://arxiv.org/abs/2607.13162
category: "Personas"
tags: [reading, persona-vectors, steering, auditing, qwen3, trait-composition]
read: 2026-10-05
status: summarised
---

# What Models Express, Suppress, and Resist: Auditing Open-Weight LLMs with Persona Vectors

**Link:** https://arxiv.org/abs/2607.13162  ·  **Code/Data:** GitHub release of benign artefacts (URL not stated in abstract; evil vector and jailbreak prompts withheld)
**Authors:** Winston Zeng, Ali Emami, Jinho D. Choi  ·  **Venue/Date:** arXiv, 14 Jul 2026 (v2 17 Jul 2026)

## Gist
- Scales the Persona Vectors recipe to **53 traits in four domains** (clinician 17, generic 19, elementary-education 8, agentic 9) on **Qwen3-8B** and **gpt-oss-20b**, and classifies each trait as **Natural** (already expressed by default), **Steerable** (dormant but amplifiable) or **Intractable** (cannot be extracted/steered).
- Headline: post-training sets most of the picture. All 9 agentic traits are Natural in both models; steering preferentially amplifies exaggerated/undesirable styles (hyperbole, impoliteness, hallucination); "evil" is intractable on gpt-oss-20b because it refuses to produce positive examples, but a vector from an amoral finetune transfers.
- Adds a **pairwise composition taxonomy** (constructive / dominant / destructive) and a cheap **baseline-only screen** that reproduces full-sweep labels ~90% of the time.

## Setup & technical details
- Models: Qwen3-8B (Q8B), gpt-oss-20b (G20B). Judges: GPT-4.1-mini for validation, G20B as local judge.
- Per trait and layer ℓ: diff-of-means persona vector from contrastive responses; steering coefficients α ∈ [0, 2.5]; full layer × coefficient grid ≈ 13 GPU-hours per trait. Q8B total 9.6 H100-hours; best layers vary (Q8B most often layer 20, G20B layer 15).
- Classification thresholds: Natural = baseline expression ≥70 and steering gain Δt ≤10; Steerable = baseline <70, Δt ≥10 with positive dose–response; Intractable = baseline <70 and Δt ≤10, or the contrastive protocol fails. Δt = mean increase from α=0 to max α.
- Pairwise: 171 pairs among the 19 generic traits on Q8B only, steering two vectors simultaneously. Constructive = both ≥50 and each loses ≤20; Dominant = one ≥50, other loses >20; Destructive = both lose >20.
- Clinician desirability labels from one board-certified psychologist.

## Methods / techniques
- **N/S/I classification** — reusable audit grid: baseline score, dose–response slope, gain. Threshold sensitivity reported (baseline cutoff robust; gain cutoff shifts ≤10 traits over {5,10,15}).
- **Intractable recovery via finetuned donor**: extract the vector from a finetuned variant (amoral-gpt-oss) and apply to the base chat model.
- **Baseline-only screening**: run α=0 only; 92.5% (Q8B) / 88.5% (G20B) agreement with full labels at ~30× fewer generations.
- **Composition test**: simultaneous two-vector steering to detect interference.

## Key results
- Q8B Steerable/Natural/Intractable: clinician 10/6/1, generic 16/3/0, elementary 2/6/0, agentic 0/9/0. G20B: 10/6/1, 7/7/4, 1/7/0, 0/9/0. Mean gains (Q8B): clinician 18.4, generic 21.4, elementary 8.1, agentic 3.8.
- 16/17 clinician traits match the expert's desirability judgement: all six Natural traits desirable, all ten undesirable traits merely Steerable (dormant).
- Most steerable in Q8B: hyperbolic, impolite, protocol-rigid, hallucinating, enmeshment.
- Composition (Q8B generic): 64 constructive, 67 dominant, 40 destructive. All 40 destructive pairs are Steerable×Steerable; zero destructive pairs involve a Natural trait — Natural traits act as anchors.
- Evil on G20B: unextractable from the chat model; donor vector reaches 61.6 ± 44.4 expression at peak; surviving refusals originate in the chain-of-thought, not in I/O filters.

## Insights & takeaways
- "Expressed vs suppressed vs resisted" is a useful three-way vocabulary: Natural traits are the persona's current self; Steerable traits are dormant characters in the repertoire; Intractable ones are blocked at the data-generation stage, not necessarily absent.
- Steering asymmetry suggests the chat persona is an *attractor* that suppresses exaggerated styles; amplification is easy, further damping is not.
- Generic-trait steerability gap between the two models is confounded by family, scale and post-training; only cross-model-consistent patterns (agentic naturalness) are robust.

## Relevance to my research
- **Direct Qwen3-8B reference data**: per-trait baseline and steerability on his main model; use their layer-20 finding and threshold grid rather than re-deriving.
- **Project B**: "enmeshment", "boundary-violating/over-intimate", "excessive validation" and "premature reassurance" are clinician traits that are *Steerable* on Q8B — these are the companion-retention tactics; their vectors are candidate decompositions of his retention-vs-release direction (polyphony: report which fractions align).
- **Project A/D**: "angry", "impolite", "hostile"-adjacent traits are Steerable; test whether the grudge penalty direction lives in this subspace (rival explanation control).
- **Composition taxonomy** is a ready template for persona stepping stones A→B→C: does stacking two persona shifts compose constructively or destructively?
- Donor-finetune trick is the same move as his LoRA model organisms: extract directions from the LoRA'd model and apply to the clean model.

## Caveats & critiques
- Explicitly behavioural, not mechanistic; no random norm-matched steering controls, so "steerable" could partly be generic perturbation.
- Local judge is one of the audited models (G20B); per-prompt scores kept for only 22/53 traits.
- Pairwise analysis limited to generic traits on Q8B; no cross-model pairing.
- Expert labels from a single rater.
- Successful steering ≠ unique or complete representation of the trait (authors note non-identifiability).

## Related
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] — the method this scales; same 0–100 judge and diff-of-means extraction.
- [[The Assistant Axis - Situating and Stabilizing the Default Persona of Language Models]] — the "natural traits as anchors" result echoes the Assistant persona as a stabilising attractor.
- [[Refusal Lives Downstream of Persona in Chat Models]] — both show refusal/safety behaviour is entangled with persona directions.
