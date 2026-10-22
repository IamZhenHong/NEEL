---
title: "Do LLMs Experience an Internal Polylogue?"
authors: "Nils A. Herrmann, Leander Girrbach, Kirill Bykov, Zeynep Akata"
year: 2026
venue: "arXiv"
link: https://arxiv.org/abs/2605.09159
code: https://anonymous.4open.science/r/polylogue
category: "Personas"
tags: [reading, persona-vectors, reasoning-monitoring, steering, probes]
read: 2026-10-05
status: summarised
---

# Do LLMs Experience an Internal Polylogue?

**Link:** https://arxiv.org/abs/2605.09159  ·  **Code/Data:** https://anonymous.4open.science/r/polylogue (anonymised)
**Authors:** Nils A. Herrmann, Leander Girrbach, Kirill Bykov, Zeynep Akata  ·  **Venue/Date:** arXiv, 9 May 2026 (v2 27 Jul 2026)

## Gist
- Treats persona vectors as **dynamic signals**: the "polylogue" is the time series of alignments between eight reasoning-role persona vectors (Interpreter, Analyst, Planner, Solver, Explorer, Verifier, Monitor, Arbiter) and the residual stream during generation.
- Polylogue features predict answer correctness about as well as raw-activation baselines, and a stage-aware steering intervention helps on 3/4 models but hurts one badly (−14.5 pts) — monitoring is easier than control.
- Mixed verdict: the roles are interpretable but only weakly tied to surface text, and random projections predict correctness just as well.

## Setup & technical details
- Models: Qwen2.5-14B-Instruct (non-reasoning), DeepSeek-R1-Distill-Qwen-14B, Phi-4-reasoning, Llama-3.1-Nemotron-Nano-8B-v1.
- Task: MMLU-Pro (12,032 MCQ, 14 domains); 2,660 samples for semantic/functional faithfulness, 504 for causal (steering).
- Personas map to Schoenfeld's episode theory of problem solving. Labelling of paragraphs and scoring use Llama-3.3-70B as judge.

## Methods / techniques
- **Persona vectors**: diff-of-means, v_ℓ = mean(act | trait-inducing prompt) − mean(act | trait-inhibiting prompt), per layer.
- **Polylogue monitoring**: at every generation step project activations onto each v; summarise by mean alignment, volatility (std), final-step similarity, dominance entropy and switching rate.
- **Semantic faithfulness**: MRR between the dominant latent persona per paragraph and the judge's surface label, vs random and frequency baselines.
- **Functional faithfulness**: L1 logistic regression from polylogue features to correctness; compared to activation baseline and random projections.
- **Causal faithfulness**: paragraph-conditioned steering ã_ℓ(t) = a_ℓ(t) + α·v_ℓ using the top-5 Lasso features (sign → amplify or suppress).

## Key results
- Semantic: beats random baseline on all four models but beats the frequency baseline on only 2/4 — latent roles align weakly with what the text says it is doing.
- Functional: correctness-prediction accuracy/AUC ≈ activation baseline, not better than random projections (baseline accuracies 42–62%).
- Causal steering: Qwen2.5 63.3 → 64.1%; DeepSeek 62.3 → 64.5%; Llama-Nemotron 44.4 → 48.8%; Phi-4 53.0 → 38.5%.
- Persona geometry: pairwise cosine distances 0.45–0.73; effective rank 1.85–3.74 of 8 (compressed); solver–verifier near-collinear (0.25), interpreter–planner near-orthogonal (0.82).
- For DeepSeek, high final-step Monitor alignment predicts errors (coef −0.74); reasoning models' predictive features concentrate at the final step, the instruct model's in distributed volatility.

## Insights & takeaways
- Persona-vector projections carry predictive signal but much of it is low-dimensional geometry rather than role semantics — a clean "decodable ≠ meaningful" lesson.
- Eight nominally distinct roles collapse to ~2–4 effective dimensions; naming vectors does not make them independent.

## Relevance to my research
- **Methodological warning for Projects A/B**: compare any direction-based monitor against random-projection and frequency baselines before claiming the direction *means* something; this paper shows how a plausible story survives the first baseline and fails the second.
- **Polyphony**: effective-rank analysis of a persona-vector set is a quick way to report how many independent "voices" are really present (e.g. his retention sub-tactics or the Zeng clinician traits).
- **Project A (within-context dynamics)**: the step-wise projection time series is the right instrument to ask whether the grudge is carried state (sustained projection) or cue-triggered (spike when the author name is attended).
- Steering collapse on Phi-4 is a reminder that stage-conditioned steering needs per-model tuning; relevant if he steers at goodbye turns only.

## Caveats & critiques
- Layer/coefficient tuned on 4 probe personas then applied to 8 reasoning personas; 504 causal samples with no significance estimates.
- Two judge-dependent pipeline stages; MCQ only; English only.
- Random-projection parity undermines the central interpretability claim; anonymised code.

## Related
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] — the static method this dynamises.
- [[What Models Express, Suppress, and Resist - Auditing Open-Weight LLMs with Persona Vectors]] — also stacks many trait vectors; both find low effective dimensionality / interaction effects.
