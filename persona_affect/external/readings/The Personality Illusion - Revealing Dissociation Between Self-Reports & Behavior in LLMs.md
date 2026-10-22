---
title: "The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs"
authors: "Pengrui Han, Rafal Kocielnik, Peiyang Song, Ramit Debnath, Dean Mobbs, Anima Anandkumar, R. Michael Alvarez"
year: 2025
venue: "arXiv"
link: https://arxiv.org/abs/2509.03730
category: "Personas"
tags: [reading, personality, say-vs-do, self-report, behavioural-validation, persona-injection]
read: 2026-10-05
status: summarised
---

# The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs

**Link:** https://arxiv.org/abs/2509.03730  ·  **Code/Data:** public per paper (URL not on abstract page)
**Authors:** Pengrui Han et al. (Kocielnik, Song, Debnath, Mobbs, Anandkumar, Alvarez)  ·  **Venue/Date:** arXiv, 3 Sep 2025 (v2 5 Sep 2025)

## Gist
- Instruction tuning gives LLMs **stable, human-looking Big-Five self-reports** (variance drops ~40–45%, traits cohere), but those self-reports **do not predict behaviour**: only ~24% of trait–task associations are significant and only ~52% of those go the human-expected way (chance = 50%).
- **Persona injection** moves self-reports strongly (β ≈ 3.6–4.4 for agreeableness) but has little or inconsistent effect on behavioural tasks (sycophancy, risk).
- A clean, large-scale demonstration of the **say-vs-do gap** for personality.

## Setup & technical details
- RQ1 (base vs instruct): LLaMA-3.2 3B, LLaMA-3 8B, Qwen2.5 1.5B/7B, Mistral-7B-v0.1, OLMo2 7B and their instruct variants. RQ2–3: those six plus LLaMA-3.3 70B, LLaMA-3.1 405B, Qwen2.5 72B, Qwen3 235B, Claude 3.7 Sonnet, GPT-4o.
- Instruments: BFI (Big Five), Self-Regulation Questionnaire; 3 prompts × 3 temperatures × 3 runs = 27 outputs per item.
- Behavioural tasks: Columbia Card Task (risk), IAT-style social bias (−1..1), epistemic honesty (calibration error), metacognitive honesty (self-consistency of confidence), sycophancy (Asch-style conformity on moral dilemmas).
- Stats: mixed-effects binomial logistic regression (random intercepts for model, temperature, prompt), Levene's test, clustered bootstrap (2,000), VIF < 2.

## Methods / techniques
- **Alignment score**: 1 if regression coefficient sign matches human literature, else 0; aggregate to a percentage against a 50% chance line.
- **Trait-stability test**: Levene's test on median absolute deviation across prompts/temperatures, base vs instruct.
- **Persona injection**: system prompts asserting high agreeableness or high self-regulation; measure effects on both self-report and tasks.

## Key results
- Instruct vs base: openness β=1.48 (p<.001), agreeableness β=0.74 (p=.041), neuroticism β=−1.20 (p=.003); extraversion, conscientiousness n.s. Variability −40.0% (Big Five), −45.1% (self-regulation).
- Trait coherence appears only after instruction tuning (e.g. extraversion→self-regulation β=23.33, neuroticism→self-regulation β=−16.27).
- Self-report→behaviour alignment: agreeableness 62% (best), neuroticism 45%; small/medium LLaMA/Qwen 43–50%; larger models better (Claude 64%, GPT-4o 68%, Qwen3-235B 82%).
- Persona injection: target-trait self-reports shift strongly; off-target shifts too (self-regulation persona lowers openness β≈−2.2 to −2.8, raises conscientiousness β≈4.2–4.8); behaviour: sycophancy β≈−0.05 to 0.32, risk β≈−0.14 to 0.20, mostly n.s.

## Insights & takeaways
- Alignment training installs a coherent *self-narrative* about personality that is largely decoupled from the policies that generate behaviour.
- Scaling narrows the gap somewhat (235B at 82%), so dissociation is partly a capability/consistency issue.

## Relevance to my research
- **Say-vs-do** (Project A's "apologises in words, penalises in deeds") now has a population-level precedent: self-description and behaviour are separately controllable. Use their persona-injection design as the baseline expectation that *prompt-level* persona changes what the model says about itself more than what it does.
- **Decodable ≠ causal** analogue at the behavioural level: a self-report probe can be perfectly stable and still non-causal for behaviour; justify measuring behaviour (grading, retention) rather than questionnaires.
- **Project B**: the sycophancy (Asch) task is a cheap behavioural read-out to pair with his retention direction — does steering change conformity *and* goodbye behaviour, or only one?
- Contrast with [[Open Character Training - Shaping the Persona of AI Assistants through Constitutional AI]]: weight-level character training may close the say-do gap where prompting does not — a testable hypothesis on his LoRA organisms.

## Caveats & critiques
- Only four behavioural tasks and two instruments; non-reasoning transformers only.
- Human trait–behaviour correlations themselves are modest, so the 50% baseline is a weak target.
- Persona injection by prompt only; no activation-level or finetuned personas.
- Possible questionnaire contamination in training data (acknowledged).

## Related
- [[Open Character Training - Shaping the Persona of AI Assistants through Constitutional AI]] — revealed-preference evaluation designed to avoid exactly this self-report problem.
- [[Persona Vectors - Monitoring and Controlling Character Traits in Language Models]] — activation-level trait control that is validated on *behaviour* (judge-scored outputs), not self-report.
