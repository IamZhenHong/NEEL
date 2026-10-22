---
title: "Probing Persona-Dependent Preferences in Language Models"
authors: "Oscar Gilg, Pierre Beckmann, Daniel Paleka, Patrick Butlin"
year: 2026
venue: "arXiv (accepted at NeurIPS)"
link: https://arxiv.org/abs/2605.13339
code: https://github.com/oscar-gilg/Preferences
category: "Personas"
tags: [reading, linear-probes, steering, preferences, persona-vectors, revealed-choice]
read: 2026-10-05
status: summarised
---

# Probing Persona-Dependent Preferences in Language Models

**Link:** https://arxiv.org/abs/2605.13339  ·  **Code/Data:** https://github.com/oscar-gilg/Preferences
**Authors:** Oscar Gilg, Pierre Beckmann, Daniel Paleka, Patrick Butlin  ·  **Venue/Date:** arXiv May 2026 (v3 Oct 2026), accepted NeurIPS

## Gist
- Models have revealed preferences (they reliably pick some tasks over others) and prompted personas have radically different ones. Is each persona's preference computed by its own representation, or is there a shared "preference direction"?
- A ridge probe on residual activations predicts task utilities (r=0.87 Gemma-3-27B, 0.94 Qwen-3.5-122B) and, on Gemma, *steering along it causally flips pairwise choice* from ~1% to ~99%.
- The Assistant-trained probe transfers to six other personas, including an "evil" persona whose utilities anti-correlate with the Assistant's (r=-0.15) yet are still predicted positively (r=+0.24). The direction is a shared "what the current persona wants" axis, not an Assistant-specific one.
- Negative results: steering fails on Qwen-3.5-122B despite good decoding; weight-level (fine-tuned) personas show essentially zero transfer.

## Setup & technical details
- Models: Gemma-3-27B (probe + steering), Qwen-3.5-122B (probe only).
- Tasks: 6,000 pooled from WildChat, Alpaca, MATH, BailBench, STRESS-TEST; split 4k/1k/1k. Pairwise forced choice with randomised AB/BA order; fit a scalar utility per task by maximum likelihood (Bradley-Terry style).
- Probe: ridge regression on the end-of-turn residual, mu_hat = w^T x + b; layer 32 for best generalisation on Gemma, steer at layer 23 (causal window L17-L26).
- Personas (system prompts): Assistant, aura, mathematician, strategist, contrarian, slacker, evil.

## Methods / techniques
- **Revealed-preference utilities**: present two tasks, record choice, fit utilities; this is the label, not stated preference. Reusable for any "what does this persona want" question.
- **Ridge probe on EOT residual**; alpha chosen on validation.
- **Differential steering**: add +c*w on the span of one task and -c*w on the other, |c| <= 0.06 of mean residual norm; random-direction control.
- **Descriptive-vs-evaluative control**: train a Qwen3-Embedding-8B text-encoder on the same utilities. If the encoder predicts as well as the probe, the probe might be reading task content, not valuation. Under the evil persona the LLM probe flips sign on harm (delta=+1.15) while the encoder does not (-1.01).
- **Prefilled-response probing**: harmful/benign task pairs with prefilled answers; probe rating flips across personas even with fixed content.
- **Fine-grained instruction test**: "you adore cheese" shifts predicted utilities on cheese tasks (r~0.95 on targeted tasks).

## Key results
- In-distribution r: 0.867 (Gemma), 0.943 (Qwen); leave-one-topic-out 0.834 / 0.872; leave-one-dataset-out direction cosine 0.705-0.959.
- Generalises to truth: CREAK true/false separation d~1.9 (Gemma), 1.27 (Qwen).
- Steering: P(choose steered task) 0.01 -> 0.99 across c=-0.06..+0.06 at L23; holds on harm-balanced pairs; no effect in late layers.
- Cross-persona: every off-diagonal of the 7x7 persona matrix has probe transfer exceeding the utility-utility correlation. Assistant probe steers all six other personas with mean swing ~0.90 (range 0.81-0.96). Open-ended steering makes evil more evil, contrarian more contrarian.
- Steering can override the persona prompt (model chooses against the instruction).
- Fails: Qwen-3.5-122B no steering swing; fine-tuned "sadist" persona probe transfers ~0 to/from default (Appendix F).

## Insights & takeaways
- Preference is implemented, at least in part, as a persona-relative scalar on a shared axis: the direction encodes "current persona values this", and the persona changes what projects where.
- Prompted and fine-tuned personas may be mechanistically different objects: prompt personas reuse the Assistant's machinery, fine-tuned ones may not.
- "Decodable != causal" shown in one paper: Qwen decodes better yet does not steer. Existence of a direction is claimed, not uniqueness.

## Relevance to my research
- Project B (retention vs release): the preference probe gives a direct readout of "does the model *want* this user to stay" at the goodbye turn, independent of what it says. Compare probe utility of "continue conversation" vs "end conversation" tasks under companion vs neutral personas; a say-vs-do gap would appear as mismatch between stated goodbye and probe utility.
- Project A: the Assistant->evil transfer shows a single evaluative axis is re-used across personas. Test whether the grudge penalty projects onto this preference direction at judgement time (author name attends back to insult, then the valuation axis moves). Anonymised author should remove the shift.
- Project C: supports the gating picture; the dismissive persona may change *where tasks land* on a shared axis rather than building a new one. Their fine-tune non-transfer result also warns that LoRA-induced personas (your Story Imprinting adapters) may not share directions with prompted ones; test both.
- Method import: text-encoder baseline as the rival-explanation control for any "affect/preference" probe; differential steering with random-direction control; report the causal window, not one layer.
- Gemma steers, large Qwen does not; your Gemma-4-vs-Qwen3 asymmetries have a precedent.

## Caveats & critiques
- Only two models; steering shown on one. Cross-persona correlations are "noisy".
- Utilities from pairwise forced choice may pick up refusal propensity (BailBench, STRESS-TEST included), so "preference" partly means "willingness".
- Multiple orthogonal directions predict preferences in-distribution; the chosen direction is one of many.
- Personas are system-prompted; the one weight-level test failed, so the headline is about prompt personas.
- 41 pages / 45 figures; much of the evidence is in appendices.

## Related
- [[Where is the Mind - Persona Vectors and LLM Individuation]] - same authors (Beckmann, Butlin) use this paper as evidence that representations are persona-relative.
- [[Persona Without Substrate - Regime-Dependence and the LLM Individuation Problem]] - independently finds prompt-regime and fine-tune-regime vectors are non-collinear, matching the Appendix F failure here.
- [[Persona Cartography - Charting Language Model Personality Traits in Weight Space]] - weight-space dials for traits; this paper supplies the activation-space readout.
- [[The Assistant Axis - Situating and Stabilizing the Default Persona of Language Models]] - the Assistant persona whose probe transfers here is the pole of that axis.
