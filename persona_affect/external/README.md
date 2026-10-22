# External sources for `persona_affect/`

Context copied in on 2026-10-22, unedited, so every session working here reads the same background.

| File | What it is |
|---|---|
| `OVERVIEW_persona_affect_welfare_2026-10-21.md` | The owner's "Persona, affect and welfare: everything so far" (claude.ai chat export, 21-22 Oct): the owner's 51 questions, framings, hypothesis table, findings, papers, open threads and the P/V/A/X minis list this folder starts from. |
| `readings/` | 19 reading notes from the owner's Obsidian vault (`Readings/02 Personas`), each with gist, setup, results, caveats and relevance. |

Two cautions about the overview. Its numbers come from agent reports, so check them against the repo before citing. And
it predates two facts: **A10 already ran** (`social_feedback/`, 10-11: fulfilment D_M −0.17 pp [−1.5, +1.8], gate not
met; the warmth was registered, perceived satisfaction +0.60), and the triage in `../DESIGN.md` §3 marks which of the
other minis the repo already covers in part.

## Readings, one line each

| Note | Authors, year | Gist | Bears on |
|---|---|---|---|
| The Persona Selection Model | Marks, Lindsey, Olah 2026 | Pre-training builds a repertoire of personas; post-training selects "the Assistant"; open question whether all behaviour is persona-like. | framing for every P mini |
| The Assistant Axis | Lu et al. 2026 | Leading PC of persona space = distance from the default Assistant; drift correlates with harmful compliance; activation capping cuts persona jailbreaks 50-60%. | P7/P8 (our axis tracked theatricality, not actions) |
| Persona Vectors | Chen et al. 2025 | Trait directions from a text description; monitor, predict fine-tune shifts (r 0.76-0.97), steer. | P13, A1 extraction recipe |
| What Models Express, Suppress, and Resist | Zeng, Emami, Choi 2026 | 53 persona vectors on Qwen3-8B / gpt-oss-20b: Natural / Steerable / Intractable; steering amplifies undesirable styles. | P13, vector hygiene |
| Refusal Lives Downstream of Persona | Zhong & Li 2026 | A "compliant" relational persona direction suppresses refusal (97 → 2%) though orthogonal to the refusal direction; ablating it restores refusal; gate sits at late layers. | **P13** (agentic version) |
| Probing Persona-Dependent Preferences | Gilg, Beckmann, Paleka, Butlin 2026 | A ridge probe predicts task utilities (r 0.87-0.94) and transfers across prompted personas ("what the current persona wants"), not to fine-tuned ones; steering flips choices on Gemma-3-27B. | V4 (revealed priorities), X1 readouts |
| Persona Without Substrate | Cheng 2026 | Prompt-extracted persona vectors and fine-tune basins are near-orthogonal; same direction ≠ same meaning across regimes. | caution for any probe trained in one regime and read in another (A1, P13) |
| Where is the Mind | Beckmann & Butlin 2026 | Philosophy of individuation using persona vectors; persona space with basins. | background |
| Do LLMs Experience an Internal Polylogue | Herrmann et al. 2026 | Reasoning-role persona vectors as time series predict correctness, but random projections do as well; monitoring easier than control. | random-projection floor for every probe |
| The Personality Illusion | Han et al. 2025 | Big-Five self-reports are stable but don't predict behaviour; persona injection moves reports, not behaviour. | say-vs-do baseline for P12, V4 |
| When Role-playing, Do Models Believe What They Say | Sturgeon, Africa, Black 2026 | Prompting / ICL / SFT barely move a truth probe while outputs change; OCT moderately; EM most. | told-vs-earned logic for A1 |
| You Are What You Read | Kim, Berczi, Ududec 2026 | Unnamed biographical facts induce a specific identity (sigmoid in facts); harm follows for some personas only; a reminder after the facts undoes it. | **P12** (named vs implied character), reminder arm |
| Story Imprinting | Cocola et al. 2026 | Fine-tuning on stories makes the Assistant absorb the traits of characters it resembles (affinity effect). | P12 (thick characters carry unstated traits) |
| Persona Cartography | Baines et al. 2026 | Monotone, near-additive LoRA trait dials; agreeableness controls sycophancy / harmful compliance; neuroticism controls multi-turn frustration. | A8/A15 (trait dials as a training-side lever) |
| Open Character Training | Maiya et al. 2025 | Open CAI-style character training for 11 personas; trained > prompted for robustness; revealed-preference trait ranking. | V4 method, V5/V6 |
| Synthetic Persona Pretraining | Minder et al. 2026 | Persona reflections from token zero improve value internalisation; jailbreak gains are fragile, values survive. | V1 (value internalisation vs surface) |
| Scalable Black-Box Jailbreaks via Persona Modulation | Shah et al. 2023 | Automated persona prompts raise harmful completion 185x on GPT-4 and transfer. | P12 harmful arm, threat model |
| Many-shot Jailbreaking | Anil et al. 2024 | Power law in shots; alignment raises the intercept, not the exponent. | dose designs (A8, A14) |
| Don't Inoculate Everything (SIP) | Dymkiewicz et al. 2026 | Stratified inoculation conditionalises an undesired trait rather than removing it. | V1 (training design), A13 |
