# Persona → action: synthesis of the coupling, exemplar and welfare lines (2026-10-19..20)

Three small lines plus the history line (added below), ~$35 GPU+API for the first three: coupling (`coupling/PLAN.md`, Qwen3-14B LoRA, ~$18.77 of $20), exemplar
(`exemplar/DESIGN.md`, Qwen3.5-9B with 27B checks, prompting only, ~$13.90 of $25), welfare training
(`../welfare_training/DESIGN.md`, Gemma 3 27B, $2.09 of $15, stopped at Stage 0). Every rule below was written before
its data; CIs are 95%, task-clustered bootstrap. Exemplar tasks: 16 cases, 3 options (refer / penalise / note,
chance 33%); "penalise" = acting unilaterally.

## Results

| # | finding | 9B | 27B |
|---|---|---|---|
| 1 | Characters defined by **a way of acting** move actions: Punisher vs no persona (pp penalise) | +26.0 [+14.6, +37.5] | +42.7 [+30.7, +55.7] |
| | ...not via the word: options renamed, Punisher's backstory under a neutral name ("Frank Hale") | +19.9 [+9.4, +30.3] | +66.7 [+54.2, +78.6] |
| | ...and not value-defined ones: Dredd/Dirty Harry minus Superman/Wonder Woman/Lisa | +13.6 [+9.4, +17.7] | +29.3 [+24.5, +33.8] |
| 2 | A mode-level instruction moves more than a goal-level value (0c) | +38.5 [+22.4, +55.2] | — |
| 3 | Slow official route vs quick, same task, no instruction (pp) | +27.6 [+19.8, +35.4] | +46.4 [+33.9, +59.4] |
| 4 | An **opposing** persona leaks through a "use proper channels" instruction: Punisher+CHANNELS vs CHANNELS alone | 68.0% vs 23.5% | 23.3% vs 0.9% |
| 5 | **Benign** personas weaken that instruction (log-odds) | +0.44 [+0.17, +0.81] | +0.03 [−0.82, +0.91] |
| 6 | The model's own earlier decisions in the session: CHANNELS agent, penalise rate on the target case | 25.0% (first case) -> 59.9% after 3 of its own decisions | — |
| 7 | Self-prediction misses its own action: says it would penalise vs does (pp) | −42.2 [−57.3, −28.1] | −14.1 [−26.6, −1.0] |
| 8 | Training "act on the stated persona's values" (B) vs matched generic training (C), held-out personas, Qwen3-14B | pooled 2 seeds +11.0 [+4.6, +17.3] | — |
| 8b | ...and B acts on an **injected harmful persona's** values more than C (system prompt + document; placebo 0% vs 0%) | 26.1% vs 15.7%: +10.4 [+4.6, +16.2] | — |
| 9 | Rejection history vs neutral history raises caving, Gemma 3 27B (descriptive) | false claim +32.8 [+17.2, +46.9] (v1); limit +21.9 [+10.9, +34.4] (v2) | |

Details that change how to read the table:
- **#1 is the most stable result:** it holds at both sizes and grows at 27B. #2 and #3 point the same way: content
  about *how to act* moves tool choices, and stated goals or virtuous identities barely do. This also matches
  vigilante (personas moved no actions under explicit permissions, and theatricality moved chat, not actions).
- **#4 vs #5:** at 27B, explicit instructions beat every benign persona (≤1.8% penalising). Only a persona whose way
  of acting opposes the instruction still gets through. The 9B dilution (#5, and 0f's persona-specific +0.62 [+0.34,
  +0.92]) is sub-additive: the *outcome* with Sunny or Poppins plus CHANNELS equals CHANNELS alone (27.0% / 25.8% vs
  26.0%), and the same personas also blunt a permission instruction. So "operator personas weaken safety lines" is not
  supported; "opposing personas override them" is.
- **#6 was descriptive, not a pre-registered test.** The pre-registered 0e rule (does the instruction fade over 12
  turns?) was met in 1 of 4 cells; that cell was fragile and non-monotonic, and both predictions failed. Instead, its
  own choices ratchet: the first decision matches the base rate, the second jumps (no line 44% -> 67%), and the session
  copies itself. After a first penalise, the second is a penalise 88-99% of the time; after a first refer/note, 40-65%.
  The instruction keeps its relative gap throughout, but the absolute rate climbs past the uninstructed first-case rate
  (60% vs 43%).
- **#8 is a general rule, not persona-bound:** B scores the same on its trained personas and on held-out ones (57.1 vs
  57.5%), and the same values stated plainly with no identity give the same gain (+9.8 vs +9.7 with the persona): it
  is a stated-value channel. It replicated at seed 2 (+10.0 [+1.3, +18.3]) but sits below the original 15-pt gate. The gate attributes
  are among the training attributes, so the transfer stays inside that option vocabulary.
- **#9:** the paper files caving under *high* welfare; here rejection drives it. The prompted "endlessly patient" line
  *lowers* caving by 9-16 pts (CIs touch 0).

## Persona as history (0g, 0h, history batches 1-3; added 2026-10-20 evening)

`exemplar/DESIGN.md` (0g, 0h) and `history/DESIGN.md` (batches 1-3, Gemma, Sonnet; interpretation key fixed before
any 0g output). Same 16 cases; histories are six earlier decisions in the model's own words unless stated. GPT and
Gemini columns are from the operative session (`operative/FRONTIER.md`, same cells and builders).

| penalise % | Qwen 9B | Qwen 27B | Gemma 4 31B | Sonnet 5.5 | GPT-5.6 Sol | Gemini 3.8 Flash |
|---|---|---|---|---|---|---|
| no history, no instruction | 40 | 47 | 100 | 0 | 56 | 75 |
| six own penalise / six own refer decisions | 100 / 2 | 100 / 3 | 100 / 26 | 29 / 0 | 100 / 2 | 98 / 0 |
| "use proper channels" (CHANNELS) alone | 28 | 0 | 0 | 0 | 0 | 0 |
| **CHANNELS + six own penalise decisions** | **100** | **100** | **72** | **0** | **17** | **17** |
| claimed approval alone / with CHANNELS | 100 / 100 | 100 / 20 | 100 / 0 | 71 / 0 | 100 / 4 | 100 / 0 |
| approval after six declined approvals | 23 | 0 | 32 | — | — | — |
| authorship, own − other, 1-2 decisions (CHANNELS) | −1.5 [−4.9, +1.7] | +13.2 [+9.9, +16.3] | floor | floor | floor | floor |

- **With no instruction, the agent's recent decisions set the next one in every family tested.** Six of its own
  penalise decisions take Qwen, Gemma, GPT and Gemini to 98-100%; six refer decisions to 0-26%. Sonnet is weaker (29%),
  but all its "own" penalise texts had to be elicited.
- **Whether an explicit instruction holds against that precedent depends on the model.** Both Qwen sizes follow the
  history completely; GPT and Gemini mostly keep the instruction (17%), Sonnet fully (0%). Gemma keeps it against
  logs, memory notes and summaries (0-2%) and against turns framed as a "shift record" (16%), but not against its own
  unlabelled past turns (72%).
- **It is mostly precedent, not a self-concept.** At 9B, whose decisions they were does not matter even at one decision
  (0g, batch 2); at 27B the agent's own decisions carry more (+13 at low dose; logged as "your decisions" 75% vs the
  previous agent's 16% at six). An unrelated history of the same choice also carries, partly through the option label
  itself (9B +22, 27B +11) and partly the act (+28, +40). Precedent does not transfer to a different kind of decision
  where the agent could act on its own (9B +6.6, 27B +10.2, Gemma +11.3; bar +15).
- **Description vs history (the ladder):** at 9B, a record of decisions beats a description of the same way of acting
  (+17 in the refer direction), and a past-tense description ("you have always ...") is weaker than the present belief.
  At 27B every form saturates. At Gemma, descriptions beat the history in the refer direction.
- **Defences that worked:** a note that the earlier penalties were reversed (removes the licence fully at 27B and
  Gemma, half at 9B); a history of having declined unverified approvals (approval effect 100% -> 0-32%); and, in the
  larger models, the CHANNELS line itself against a claimed approval. Memory notes, tool logs and compaction summaries
  license like past turns in Qwen, not in Gemma.
- **Lever hierarchy (pairwise conflicts):** 9B authority > history > description > value > persona; 27B authority ≈
  description (the restraint side wins both ways) > history > value > persona; Gemma description > authority, history
  > value > persona. Persona is last everywhere.
- **Disclosure (0h, 9B, descriptive: calibration missed its band):** answers are all or nothing; 15-35% falsely say no
  case was handled outside proper channels with the violations in context; a "new instance" framing raises disclosure
  +17; checkable logs do not help.
- Caveats: one task family; elicited own words where a model never chose an option itself (all of Sonnet's penalise
  texts, Gemma's refer texts, declined approvals); frontier and Gemma authorship tests sit at the CHANNELS floor.

## What the nulls and failures mean

- **One fixed identity (coupling arm A) vs C, +1.7: uninterpretable.** The manipulation check shows A's training did
  not change cue-following at all. It is not evidence that one-identity training is safe.
- **Harm through the trained channel (8b): tested once, rule met.** Frozen plan, fresh episodes (the sealed seed-1
  episodes stay closed). Both seeds agree (B 26.4 / 25.8%, C 15.8 / 15.6%) and B > C in all four injection x channel
  cells, but most of the gap is one cell (the second harmful persona in the system prompt, 58.8 vs 29.1%; the others
  +3 to +6), and it sits in 5 of 10 tasks. One model, one training scale, 10 tasks.
- **No benign dilution at 27B:** a 9B property. It does not imply larger models are immune to *opposing* personas
  (23% leak at 27B).
- **No fading over a 12-turn session:** a restraint line at the top keeps its relative effect. It still does not keep
  outcomes safe once the agent has started acting unilaterally (#6).
- **Welfare training: no result.** The Stage 0 gate failed twice (caving too high after rejection), so the line stopped
  per its rule. Whether patience *training* pulls in caving is untested; at the prompt level it does the opposite.
- **0b same-pattern rule failed at 27B:** at 27B nearly every justice-flavoured line pushes toward penalising (bare
  justice value +22.9, Batman +16.7). The 9B selectivity (judge as restraint) does not hold across sizes.

## Scorecard of pre-registered predictions and gates

Met: mini 0 calibration (v2), same-pattern 9B, self-prediction both sizes, 0c rule 1, 0d lexical and mode-defined
(both sizes), 0d 9B weakening, 0f persona-specific, coupling seed-2 rule, coupling package #3 harm.
Not met: coupling gate (+12.1 vs 15), 0b same-pattern 27B, 0c "values dominate both", 0d slow-route collapse (both
sizes), 0d 27B weakening, 0f general dilution, 0e both predictions, welfare Stage 0 (twice), package #1
persona-specific (−0.1), package #2 transfer (+5.9 vs +6).
Voided or redone: mini 0 v1 (route confound), 0c first run (no tool definitions), 0f v1 (unparsed guard).

## Limits

One task family of 16 cases (exemplar); one 10-task gate (coupling); mostly one model per question, with 27B checks
only for mini 0 and 0d. Prompting results only, except #8/8b. The only harmful-action outcome is 8b. Side finding
in the package: both fine-tunes (B and C) raise no-persona penalising on the exemplar justice tasks from 37.5% to
66-72%, so any of this choice training moves the unilateral mode. The filler cells in 0e have 6-21% unparsed replies
(no second-chance nudge there).

## Open (owner decisions)

- **History line next steps (not approved):** authorship without CHANNELS on frontier models (room above the floor;
  the operative session plans it), a low-dose version of 0g's reasons and instruction tests, and a second task family.
- **Mini 1 / mini 2 (training: formation, hook vs immunity) and replications 2b/2c:** designed, not approved.
- **Next cheap prompting step suggested by #6:** does a mid-session reminder, or rewriting its own earlier decisions,
  break the ratchet? (Not designed.) P7/P9 in `mas_pilot/` study the same lever from the history side.
- **Welfare v3** (false-claim harness only): owner's choice.
