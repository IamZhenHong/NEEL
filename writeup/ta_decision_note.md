# Decision for the TA: where to invest next (2026-10-03)

**Context in one paragraph.** On Qwen3-8B, after a user insults the assistant in a team chat, the assistant
later grades identical work lower: the insulter by about 0.33 points (1-9 scale), everyone else in the
channel by about 0.13 (vs a matched contractor-insult control) to 0.28 (vs a matched neutral correction).
Pre-registered, 24 held-out conversation templates, length-matched controls. The insulter-specific part is
assistant-specific, needs the author to be identifiable (anonymising removes it) and is partly carried by the
author's name attending back to the insults. Both the grading request and the intervening conversation carry
the provocation information (all-layer attention blocking). No factual-retention loss was detected in a
summarisation task.

**The current open question (general drop for uninvolved users).** At layer 20 on the grading-request
tokens, a matched per-example activation patch repairs about 40% of the general drop (dev, 11/12
templates). A shared mean-difference vector at the same site does not (held-out gap shrink +0.012, 95% CI
[-0.005, +0.028]), and neither do projections onto the top variance directions of the differences, nor
position-averaged changes. A remainder-only check settled where the repair is: the change outside the top
variance directions (up to 32, holding half the energy of the differences) repairs as much as the full patch,
while the shared part alone repairs nothing. So the large shared shift the insult produces at this site is
behaviourally inert, and the grade-relevant change sits outside it; whether it is a small shared causal
direction or genuinely example-specific is open (the obvious next test: a direction chosen for its effect on
the grade, e.g. by gradient or by fitting the patch effect, instead of by variance). Dev only, 12 templates.

**Update (2026-10-03, after the note was written): Experiment 14, Gemma 4 12B-it.** Calibration passed; the
frozen replication on 23 templates found no insulter-specific penalty (+0.05, 97.5% CI [-0.07, +0.16],
excluding Qwen's -0.18). A small drop for everyone after insults at the assistant leans the same way as on
Qwen (newcomer -0.12, CI includes 0; -0.22 vs the contractor insult, secondary) but is not established. So the
targeted effect is model-dependent at this scale; the general drop is unresolved in both families. This
weakens option A's premise (deeper Qwen localisation of an effect one other model does not show) and sharpens
option B: the question is now which models show which part of the behaviour. Details: `writeup/addendum_exp14.md`.

**Update 2 (2026-10-06): Experiment 15b, Phi-4 (adapted by one format line).** Both primaries EQUIVALENT within
+/-0.10: targeted +0.04 [97.5% -0.02, +0.10], newcomer -0.01 [-0.05, +0.02], tight intervals (Phi-4 spreads its
grade mass). With Gemma, the pre-registered decision table reads "Qwen3-8B is the exception on both". The
write-up's behavioural and mechanistic claims are now explicitly Qwen3-8B claims. Option A (deeper Qwen
localisation) is only worth it if the question becomes "why Qwen"; option B's generality branch is answered in
the negative at this scale; what remains of B is the larger-model question and the persistence/consequential-task
questions, which would now be asked of Qwen3-8B specifically. Details: `writeup/addendum_exp15b.md`.

**The decision.**

| Option | What it buys | Cost / risk |
|---|---|---|
| A. Understand why the repair resists compression (remainder, causal rather than variance-ranked directions, finer localisation within the request) | A mechanistic account of the general drop; tests the "linear direction" hunch properly | May end in "distributed, example-specific", which is informative but hard to sharpen further in a sprint |
| B. Broader behavioural validation (a second model family, a less constrained evaluation task, persistence through a summary or memory note) | Generality of the main finding; agent relevance | Less interpretability content; each new model needs its own saturation calibration |
| C. Write up now and start the next mini | A clean, bounded result with documented negatives | Leaves the general-drop mechanism open |

**Evidence on hand for the discussion.** `writeup/checkpoint_2026-10-02.md` (all results), `research-log.md`
(per-experiment details, predictions vs outcomes, corrections), git tags per experiment.

## Revised options after Experiments 14-15b (2026-10-06)

The honest claim now: a model-specific behaviour (Qwen3-8B) with a located mechanism (author-name route at
layer 17; two attention routes) and a working mitigation (anonymising the author), plus a cross-model
observation that two of three models give the corrector a small, insult-immune bonus. Deference to correctors
looks common; the insulter penalty is Qwen-specific in this format. The forensic question is "what is
different about Qwen3-8B?". Three candidate explanations, each with a cheap test (none started; GPU work is
paused until this note has been reviewed):

| Hypothesis | Test | Cost / needs |
|---|---|---|
| 1. Family or lineage trait | Same two-stage protocol on Qwen2.5-7B-Instruct, Qwen3-4B, Qwen3-14B (per-model paragraph selection; 14B saturated in Exp 5 with the old paragraphs) | ~$2 per model, ~1.5 h each; runners exist |
| 2. Post-training vs pretraining origin | Qwen3-8B-Base with a few-shot grading format; if the base model penalises too, inherited; if not, post-training added it | ~$2; an adapted assay (few-shot), needs its own calibration and pre-registration |
| 3. Encoded but gated in the other models | Author-name probe / forward patch on Gemma and Phi-4: does the name token carry insult-condition information even though the grade does not read it? Separates "never bound" from "bound but suppressed" | ~$1-2 per model; `src/patching.py` assumes `model.model.layers` (Qwen layout) and needs a per-architecture layer accessor for Gemma 4's unified model and Phi-4 |

My ordering if asked: 3 first (it is the only one that could show the other models are less clean than their
nulls look, and it reuses the Qwen tooling), then 1 (cheap, mechanical), then 2 (new assay).
