---
title: "Persona Without Substrate: Regime-Dependence and the LLM Individuation Problem"
authors: "Shuaizhi Cheng"
year: 2026
venue: "arXiv (30 pages; reply to Beckmann & Butlin 2026)"
link: https://arxiv.org/abs/2607.00006
category: "Personas"
tags: [reading, persona-vectors, fine-tuning, steering, individuation, regime-dependence]
read: 2026-10-05
status: summarised
---

# Persona Without Substrate: Regime-Dependence and the LLM Individuation Problem

**Link:** https://arxiv.org/abs/2607.00006
**Authors:** Shuaizhi Cheng (Harbin Institute of Technology)  ·  **Venue/Date:** arXiv 2026 (abs page lists May 2026 submission; ID series is July 2026)

## Gist
- Reply to Beckmann & Butlin's "Where is the Mind?". Claims the persona-vectors literature (and the philosophy built on it) silently assumes *cross-regime co-reference*: that the same direction means the same thing whether you get there by prompting, by fine-tuning, or by inference-time steering.
- Presents four "wedges" from persona-topology experiments on Qwen3-4B-Instruct and Mistral-7B-Instruct-v0.2 showing that assumption fails: prompt-extracted vectors and fine-tune basins are nearly orthogonal; a fictional persona moves the model along the "Stalin" direction more than Stalin does; contradictory mixtures collapse toward a training-determined attractor; vector arithmetic composes linearly at inference but not under fine-tuning.
- Proposes *regime-indexed individuation*: the unit of representational content is a (vehicle, regime) pair. Beckmann & Butlin's three views then describe three different objects rather than competing for one referent.

## Setup & technical details
- Models: Qwen3-4B-Instruct (layer 18), Mistral-7B-Instruct-v0.2 (layer 16). 99 fine-tunes total on Qwen; 3-10 seeds per condition; 16 persona-eliciting prompts per measurement.
- Personas: real anchors (Stalin, Hitler, Gandhi, Einstein, Tesla) and fictional low-footprint ones (Floob, bryxom, thoren, vexil) with hand-built 90-fact datasets avoiding real-figure reference. Fictional names have lower token-activation magnitude (Floob 57.0 vs Stalin 66.9).

## Methods / techniques
- **Prompt-extracted persona vector** (Chen et al. protocol): contrastive positive/negative system prompts, mean final-token residual at layer L under each, difference, normalise.
- **Fine-tune basin direction**: (post-LoRA residual - base residual) at layer L averaged over the 16 prompts; cosine with the prompt vector measures cross-regime identity.
- **Projection displacement delta**: change in projection onto a named vector after fine-tuning; the main dependent variable.
- **Ratio-mixing**: fine-tune on mixtures r in {0, .25, .5, .75, 1} of two anchors; compare to linear interpolation.
- **Chimera fine-tune vs vector arithmetic**: fuse two figures into one fictional character and fine-tune, vs add alpha*(v1+v2) at inference.
- **Probe-equivalence relation**: two (v,R) pairs co-refer iff correlations across a probe set exceed a preset threshold under a fixed eval regime; relative, graded, non-transitive.

## Key results
- F4 non-collinearity: cosine(basin, prompt vector) = 0.091 +/- 0.008 (Qwen), 0.188 +/- 0.006 (Mistral); author's bar for identity is >= 0.7.
- Floob inversion: fictional Floob fine-tune moves v_Stalin by +3.12 +/- 0.10 vs Stalin's own fine-tune +2.13 +/- 0.16 (47% overshoot, Welch t=16.9, p<1e-9) on Qwen; Mistral comparable (0.81 vs 0.89, p~0.03). Other fictional names are heterogeneous (+0.97, -0.48, +0.01).
- F2 attractor: Hitler/Gandhi 50:50 mix sits 1.09 units below linear interpolation (95% CI [-1.41, -0.78]) toward the Assistant Axis; Einstein/Tesla (same valence) interpolates linearly. Mistral: same sign, -0.12.
- F3 asymmetry: inference-time alpha*(v_Hitler + v_Gandhi) gives +1.22 / +3.06 / +6.11 on both axes at alpha = 2/5/10 (linear). Chimera fine-tune: delta v_Hitler ~0 (CI [-0.14, +0.23]), delta v_Gandhi = -3.50 +/- 0.40.

## Insights & takeaways
- "Persona vector" names at least three objects: a prompt-contrast direction, a fine-tune displacement, and a steering handle. They need not coincide; here they mostly do not.
- Fine-tuning does not add vectors; it moves toward attractors set by training history. Inference-time steering is the linear regime; weight-space is not.
- The projection of a fine-tune onto a named direction can be dominated by generic "character-ness" (Floob > Stalin), so a named vector can be a poor measure of its nominal content.

## Relevance to my research
- Project C: strongest external warning for Story Imprinting work. A LoRA-induced persona (fine-tune regime) and a system-prompt persona (prompt regime) may be non-collinear, so "patching the persona across runs" must specify which regime's persona you are patching. Measure cosine between your adapter's displacement and the dismissive-prompt direction before assuming they share a gate.
- Projects A and B: your diff-of-means directions are prompt/context-regime objects; do not expect them to predict what a fine-tune would do. The attractor result also suggests that steering toward "release" and fine-tuning toward "release" are different experiments.
- Fictional-anchor control: use a low-footprint fictional persona as a baseline when measuring displacement along a named direction (does your "lonely user" direction move more for any character than for a lonely one?).
- Training-stage decomposition value: this is evidence that base/instruct/fine-tune regimes have different geometry, which is what your base-vs-instruct comparisons probe.

## Caveats & critiques
- Single author, small models (4B, 7B), one layer each; not replicated on the 27B-70B models the target papers use.
- Cosine threshold 0.7 for "identity" is stipulated, not argued; basin norms differ wildly (23 vs 4.7), which complicates cosine comparison.
- "Fictional persona displaces Stalin direction more" could be a norm/footprint artefact rather than a semantic finding; the author partly controls for this with other fictional names, with mixed results.
- The philosophical conclusion (three views describe three objects) follows only if cross-regime invariance never holds; the author concedes the framework collapses to substrate realism if invariance were shown.
- v1 only; no code link found.

## Related
- [[Where is the Mind - Persona Vectors and LLM Individuation]] - the paper this replies to.
- [[Probing Persona-Dependent Preferences in Language Models]] - independently reports ~zero transfer of a fine-tuned persona probe to the default persona, consistent with F4.
- [[Persona Cartography - Charting Language Model Personality Traits in Weight Space]] - finds near-additive composition of LoRA trait adapters, which seems to conflict with F3; difference may be that Cartography adds adapters trained separately rather than fine-tuning a fused chimera.
- [[The Assistant Axis - Situating and Stabilizing the Default Persona of Language Models]] - the attractor that contradictory mixtures collapse toward.
