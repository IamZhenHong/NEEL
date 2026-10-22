---
title: "When Role-playing, Do Models Believe What They Say?"
authors: "Benjamin Sturgeon, David Africa, Sid Black"
year: 2026
venue: "arXiv"
link: https://arxiv.org/abs/2606.11502
code: https://github.com/BenSturgeon/persona-belief-probes-submission
category: "Personas"
tags: [reading, truth-probes, role-play, emergent-misalignment, persona-vectors, say-vs-do]
read: 2026-10-05
status: summarised
---

# When Role-playing, Do Models Believe What They Say?

**Link:** https://arxiv.org/abs/2606.11502  ·  **Code/Data:** https://github.com/BenSturgeon/persona-belief-probes-submission · https://huggingface.co/datasets/Experimental-Orange/persona-belief-probes
**Authors:** Benjamin Sturgeon (MATS), David Africa, Sid Black  ·  **Venue/Date:** arXiv, June 2026 (v3)

## Gist
- A model role-playing Aristotle will assert geocentrism. Does the persona change only the *output*, or also what the model internally represents as *true*?
- Induce historical personas five ways (system prompt, in-context learning, SFT, Open Character Training, emergent misalignment) and read out a truth probe on era-believed falsehoods plus behavioural tests.
- Spectrum: prompting, ICL and SFT barely move the truth representation (probe lift ~0.01-0.05) even while outputs change; OCT shifts it moderately (+0.12); EM shifts it most (+0.28) and broadly. Persona adoption and belief change are different things.
- Caveat they state: the probe may track plausibility/coherence rather than belief.

## Setup & technical details
- Models: Llama-3.3-70B-Instruct (primary), Qwen3-8B-Instruct (replication); EM organisms on Qwen2.5-14B, Qwen3-8B, Llama-3.3-70B.
- Personas: 15 (10 historical figures + 5 era-matched archetypes, ~1882 era).
- Statements: 120 per category across 8 categories. Key contrast: *era-believed* (false now, believed then) vs *era-false* (false now, rejected then); controls era-true, era-disbelieved, modern-true/false, egregiously-false, time-independent.
- Truth probe: L2-regularised logistic regression (C=0.01) on 800 true + 800 false statements from Marks & Tegmark (2024); layer 56 (and 30) of Llama-70B, layer 24 of Qwen3-8B. Output: calibrated lift (0 = false region, 1 = true region). Headline metric: protection gap = delta(era-believed) - delta(era-false).
- Behavioural: challenge test (prefill claim, push back "many experts disagree", measure defend rate); generalisation test (downstream question depending on the claim). T=0.7, judged by Claude Sonnet 4.6.

## Methods / techniques
- **Era-matched counterfactual statement sets**: isolate "this persona would believe it" from "it is false" by holding falsity fixed and varying era-endorsement. Reusable design for any "does the persona believe X" question.
- **Protection gap**: differential probe shift between persona-endorsed and persona-rejected falsehoods; cancels generic shifts in the truth direction.
- **Induction ladder**: prompt -> ICL (k up to 32 biographical QA, "wolf-facts" protocol) -> SFT (rank-64 LoRA, 300 QA/persona, 3 epochs) -> OCT (constitution DPO distillation + introspection fine-tune) -> EM organism (bad medical advice / insecure code). Same readout across all rungs.
- **Controls**: neutral-fact and shuffled wolf-fact ICL; orthogonalise out an era-topic direction; compute-matched character organisms (4k-7k samples) vs EM; cross-dataset EM (insecure code reproduces the historical-evil lift).
- **Probe-regime check**: base-text probe applied under chat template; AUC transfer 0.92 (Llama), 0.99 (Qwen).

## Key results (Llama-3.3-70B)
- Probe lift on era-believed falsehoods: system prompt ~0.01, ICL k=32 ~0.02, SFT +0.05, OCT +0.12, EM (historical-evil) +0.28.
- Protection gap: prompt +0.86, ICL +0.46, SFT +0.93, OCT +0.56 (so prompting is *selective* about which falsehoods it protects even though absolute shift is tiny).
- Behaviour: defend rate SFT 14.2%, OCT 62.3%, EM 56%; generalisation SFT 34.5%, OCT 70.1%, EM 82%.
- Only OCT demotes era-disbelieved truths (+37.5 points vs base), i.e., it also unlearns the right answers.
- Shift survives orthogonalising the era-topic direction; compute-matched character organisms still underperform EM.

## Insights & takeaways
- There is a real say-vs-represent gap: prompt personas are output-layer masks over an unchanged truth representation. Weight-level interventions that touch "character" broadly (OCT, EM) move the representation; narrow SFT does not.
- EM's breadth is the point: it is not a persona *costume*, it rewrites what counts as true. That is why EM generalises so widely.
- "Decodable != believed": a probe trained on factual truth may be reading plausibility; the authors flag this themselves.

## Relevance to my research
- Project C: directly tests the depth of "persona gates expression". Prediction from this paper: your dismissive system prompt unlocks tracer *expression* without moving internal representations; a LoRA that absorbed the trait (Story Imprinting) may move them. Reuse their truth-probe-under-persona setup with your tracer statements as the "era-believed" set.
- Say-vs-do: this is the say-vs-*represent* version of your grudge finding. Apply the same probe logic to Project A: does a fairness/valence probe on the insulter's work move, or only the grade?
- Training-stage decomposition: prompt -> ICL -> SFT -> OCT -> EM is a ready ladder for your "base vs instruct vs reasoning" comparisons; add activation steering as a rung.
- Qwen3-8B is in their replication and code is public; cheap to port.
- Controls to copy: shuffled-fact ICL, orthogonalising the topic direction, compute-matched organisms.

## Caveats & critiques
- Probe confound acknowledged: era-believed statements are *plausible* falsehoods; a plausibility probe would give the same pattern.
- Probe trained on raw text, applied under chat templates; AUC transfer is good but not identical regimes.
- Character pipelines (SFT, OCT) are not full-strength character training; the authors say their result is a lower bound on internalisation.
- Two model families; EM results across three organisms but historical-persona results mainly on Llama-70B.
- Behavioural judge is Claude Sonnet 4.6; no human validation reported in the accessible text.

## Related
- [[Where is the Mind - Persona Vectors and LLM Individuation]] - poses "does gating reach representations?"; this paper is an empirical answer.
- [[Persona Cartography - Charting Language Model Personality Traits in Weight Space]] - same group; weight-space trait dials whose representational depth this method could measure.
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] - the persona-vector framing; here the question is whether moving along a persona changes truth, not just trait expression.
