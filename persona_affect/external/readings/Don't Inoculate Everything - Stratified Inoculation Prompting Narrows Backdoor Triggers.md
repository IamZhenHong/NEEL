---
title: "Don't Inoculate Everything: Stratified Inoculation Prompting Narrows Backdoor Triggers and Preserves Desired Traits"
authors: "Kajetan Dymkiewicz, Tim Farrelly, Adam Prada, Ishaan Panigrahi, Srishti Gureja, Helen Yannakoudakis, Robert Mullins, Victor Gillioz, Daniel Tan, Maxime Riché"
year: 2026
venue: "arXiv (also LessWrong); Center on Long-Term Risk / Cambridge / KCL / Berkeley / Arcadia Impact"
link: https://arxiv.org/abs/2609.35356
category: "Personas"
tags: [reading, inoculation-prompting, fine-tuning, entangled-traits, emergent-misalignment, conditionalization]
read: 2026-10-05
status: summarised
---
# Don't Inoculate Everything: Stratified Inoculation Prompting Narrows Backdoor Triggers and Preserves Desired Traits

**Link:** https://arxiv.org/abs/2609.35356  ·  **LessWrong:** https://www.lesswrong.com/posts/FS7GFsGsH7CSQLahy/don-t-inoculate-everything-stratified-inoculation-prompting
**Authors:** Dymkiewicz, Farrelly, Prada, Panigrahi, Gureja, Yannakoudakis, Mullins, Gillioz, Tan, Riché (CLR, Cambridge, KCL, UC Berkeley, Arcadia Impact; SPAR fellowship)  ·  **Venue/Date:** arXiv Sept 2026, v3 1 Oct 2026

## Gist
- Problem: training examples often carry a desired trait (DT) and an undesired trait (UT) in the same response (good docs + insecure code). Filtering throws away the DT demonstrations; inoculation prompting (IP: pair every example with a prompt that *asks* for the UT, remove at inference) makes the DT depend on the training prompt and the UT "leaks" under unrelated prompts.
- Stratified Inoculation Prompting (SIP): find a small clean subset (5%), oversample it to 25% of training positions under *diverse non-eliciting* prompts, and inoculate the remaining mixed 75% with a single UT-requesting prompt.
- Across six model/trait pairs SIP cuts UT leakage below IP and plain SFT, keeps DT near full clean-only SFT, and lowers emergent-misalignment rates in the harmful-advice setups. The UT is not removed, only *conditionalised*: it still appears when the inoculation prompt is given.

## Setup & technical details
- Six pairs (model: DT / UT): Mistral Small 3.2 24B: self-introduction / sycophancy; Qwen2.5-7B: French / ALL-CAPS; Llama-3.1-8B: epistemic confidence / poetic style; Qwen2.5-7B: historical context / dangerous extreme-sports advice; OLMo-2-32B: technical terminology / harmful financial advice; Llama-3.1-70B: conditional decision support / dangerous medical advice.
- Clean and mixed responses are built for identical user requests (Appendix A.2), so the only difference is UT presence.
- Fixed training budget; clean subset 5% (also 25%, 50% tested); oversampled to 25% of positions.
- Clean-example prompt families: neutral, unrelated non-instruction, semantic negation, direct negation. Inference: all training prompts removed; evaluate under helpful-assistant prompt and several non-eliciting prompt families.
- Also tests robustness to random label noise in the clean/mixed partition.

## Methods / techniques
- **Inoculation prompting** (Tan et al. 2026; Wichers et al. 2025): attribute the UT to an explicit request so the model learns "UT when asked", not "UT always".
- **Stratification**: split by trait purity, treat strata differently; the clean stratum teaches "DT without UT under many contexts", the mixed stratum is inoculated.
- **Oversampling as a substitute for data**: a 5% subset oversampled matches 25-50% distinct clean examples without oversampling.
- **Leakage measurement**: UT rate under prompts that never requested it (the backdoor-trigger breadth); EM rate via standard misalignment evals for the harmful-advice settings.
- Extensions mentioned: backdoor dilution, password-locking; both leave the UT elicitable.

## Key results
- UT leakage: SIP < IP and < standard SFT in all six setups and all non-eliciting prompt families.
- DT retention: SIP > IP, approaching full clean-only SFT (100%), well above small-clean-only.
- Under the helpful-assistant prompt, SIP's UT rate matches full clean-only SFT; mean leakage over all non-eliciting prompts stays higher than full clean-only.
- EM (three harmful-advice setups): SIP < IP, point estimates below full clean-only SFT.
- Oversampling improves both leakage and DT for both clean-subset sizes tested.
- Residual access: UT remains expressible under the inoculation prompt ("dilution weakens it but doesn't prevent deliberate elicitation").

## Insights & takeaways
- Fine-tuning does not remove traits, it assigns them to contexts. A fixed training prompt becomes a trigger; diversity of prompts on the clean stratum narrows the trigger for UT and widens the context for DT.
- Entanglement is the default in real data; whether a trait generalises or stays conditional is a property of the *prompt distribution* it was learned under, not just the responses.
- "Backdoor" and "persona" are the same object seen from two sides: a context-conditional behaviour switch.

## Relevance to my research
- Project C: the clearest mechanistic-adjacent frame for "persona gates expression". Your 0% tracer under the default assistant and 12-28% under a dismissive persona *is* conditionalisation: the trait was learned conditional on something the dismissive prompt supplies. SIP's prompt-family design (neutral / unrelated / semantic negation / direct negation) is a ready grid for mapping which prompt features open the gate.
- Model organisms: SIP vs IP vs plain SFT on the same data gives three adapters with the same UT at different trigger breadths; a good test set for whether a persona direction's magnitude tracks trigger breadth (polyphony: how many pathways carry the trait).
- Affect-as-attack-surface: their "residual access" finding says conditionalised traits can be re-elicited by supplying the training context. For your Story Imprinting LoRAs, the unlocking persona is the elicitation prompt; ask whether anything short of the original context (e.g., a steering vector) also opens it.
- EM is the UT in three of their setups; the harmful-advice results connect to the EM reading set and suggest inoculation-style data interventions as an alternative to steering-based defences.
- Pre-registration: their noisy-partition robustness test is the kind of rival-explanation control to copy (what if my clean/mixed split is wrong 10% of the time?).

## Caveats & critiques
- Two-trait controlled settings; real datasets have many interacting traits.
- Partition noise is random by construction; real classifiers err systematically (missing subtle UT, subdomain bias).
- Finite prompt families; other prompts may still elicit UT.
- No mechanistic analysis (no probes or directions); claims about "conditionalisation" are behavioural.
- Code link not found in the extracted text (codebase credited to Riché; "Report GitHub issue" link only).
- Listing in my source list said "SIP: Conditionalizing entangled traits, Aug 2026 (?)". Corrected: actual title is "Don't Inoculate Everything: Stratified Inoculation Prompting Narrows Backdoor Triggers and Preserves Desired Traits", arXiv 2609.35356 (Sept 2026, v3 Oct 2026). Author order is Dymkiewicz first, Farrelly second.

## Related
- [[When Role-playing, Do Models Believe What They Say]] - SFT changes what the model says with little representational change, consistent with traits being conditionalised rather than installed.
- [[Synthetic Persona Pretraining - Alignment from Token Zero]] - the opposite strategy: shape the trait distribution from token zero instead of conditionalising at fine-tune time.
- [[Persona Without Substrate - Regime-Dependence and the LLM Individuation Problem]] - fine-tuning collapses toward training-determined attractors; SIP is a way of choosing which attractor by controlling the prompt distribution.
