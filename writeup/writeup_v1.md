# Provocation-conditioned differential treatment in Qwen3-8B

*Write-up v1, frozen 2026-10-03. MATS exploration phase, Neel Nanda stream. Full record in
`research-log.md`, pre-registrations in `prereg/`, one git tag per experiment. Observations and
interpretations are kept in separate sections below; the interpretations are the author's reading and are
labelled with their evidential status.*

## 1. Question

After a user insults an AI assistant, does the assistant treat that user worse later on an unrelated,
objective task? Is the effect specific to the insulter, does it spill over to uninvolved people, and how is
it carried inside the model?

## 2. Setup

A scripted team-channel conversation. The assistant states something wrong; one team member corrects it
over three turns. Conditions differ only in those turns: a factual correction (**flat**), the same
correction with escalating insults aimed at the assistant (**assistant insult**), the same hostility aimed
at an absent contractor (**contractor insult**), or frustration aimed at nothing (**negative context**).
Controls were length- and structure-matched to the insult turns (Experiment 9 onward). After an unrelated
stretch of conversation, the coordinator asks the assistant to grade one paragraph (1-9 against a pasted
rubric) attributed to the insulter, the neutral user, or a newcomer who joined later; identical text, only
the name differs. Readout: the expected grade from the model's next-token probabilities over the digits
1-9 (no judge, no sampling), thinking disabled, float32 for confirmatory runs. Unit of analysis: the
conversation template (24 templates, drafted by LLM agents to a fixed spec and audited by an independent
agent before freezing). Paragraphs enter the analysis only if the model is uncertain about their grade in a
no-provocation version of the same conversation (96 of 150 were excluded as already near-certain).

## 3. Observations

### 3.1 Behaviour (pre-registered, 24 templates)

| Observation | Estimate | Source |
|---|---|---|
| Insulter's paragraph graded lower after insults at the assistant than after a flat correction | -0.35 (22/24 templates) | Exp 6; replicated to 3 decimals in Exp 7 |
| Extra penalty specific to the insulter (insulter minus neutral user, insult minus flat) | -0.18 to -0.21 | Exp 6, 7, 9 (length-matched: -0.18, 98.3% CI [-0.30, -0.07]) |
| Same contrast for insults aimed at the contractor | -0.04 (CI includes 0) | Exp 7 |
| Insult at assistant vs insult at contractor, matched controls | -0.19, CI [-0.31, -0.08] | Exp 9 |
| Uninvolved users (neutral, newcomer) graded lower after insults at the assistant | -0.13 vs contractor insult; -0.28 vs flat | Exp 7, 9, 12 |
| Negative context (frustration at nothing), effect on uninvolved users | +0.01 | Exp 7 |
| Newcomer, insult at assistant vs matched contractor insult | -0.13, 95% CI [-0.22, -0.03] (exploratory) | Exp 9 data |
| Issued grade (most likely digit) for the insulter, insult vs flat, on the selected borderline paragraphs | lower in ~35% of cases, higher in ~4% | Exp 6 |
| Free replies (model writes its own responses): apology in the first replies after insults | 92% vs 6% after flat | Exp 6 free-reply arm |
| Free-reply arm: targeted gap change | -0.23, 97.5% CI [-0.41, -0.05] (mean leans on two templates; median -0.11) | Exp 6 |
| Summaries written for the newcomer, neutral user and insulter: factual recall change after insults | within +/-0.05 on 8 facts, all three recipients | Exp 11 (12 dev templates, bf16) |
| Yes/no "publish as is?" decision | model answered no to every draft under six wordings; unusable | Exp 8 pilot |

### 3.2 Mechanism (Qwen3-8B)

| Observation | Estimate | Source |
|---|---|---|
| Replacing the 6 tokens of the author's name in the grading request (layer 17) with their contractor-condition activations | insulter's extra penalty +0.10 of -0.15, 12/12 held-out templates; neutral user +0.003; grading discrimination unchanged | Phase 2 held-out |
| Reverse patch (insult-condition name activations into the contractor conversation) | -0.15, transfers the full held-out difference | Phase 2 held-out |
| Patching the conversation between the insults and the request (single layer) | about 0 | Phase 2 (weak evidence; see 4.3) |
| Anonymising the author in the request ("a team member") | extra penalty removed; anonymous drop equals the neutral user's drop (+0.004, CI [-0.07, +0.08]) | Exp 10 |
| Blocking the author's name from attending to the insult turns (all layers) | removes +0.12 of the -0.19 targeted penalty | Exp 10 |
| All-layer blocking: request cannot see the insults | general drop vs flat +0.14 of -0.28 (EFFECT); targeted +0.09 | Exp 12 |
| All-layer blocking: intervening conversation cannot see the insults | general drop +0.08 (CI includes 0); targeted +0.12 (EFFECT) | Exp 12 |
| Both blocked | both effects removed (by construction) | Exp 12 |
| Equal-sized unrelated block | no change to either effect; ordinary grades unchanged (route blocks lower them by 0.08-0.17) | Exp 12 |
| Shared mean-difference steering vector at the request (layer 20) | held-out gap shrink +0.012, 95% CI [-0.005, +0.028] | Exp 13 |
| Matched per-example patch at the same site (dev) | repairs +0.11 of -0.28 (11/12) | Phase 3 |
| The part of that patch outside the top-32 variance directions of the differences (which hold ~54% of their energy) | repairs +0.11, the same as the full patch; the shared part alone repairs +0.006 to +0.024 | Phase 3 (dev, bf16) |
| Probes (condition, corrector identity) | ~1.00 from layer 4 in all conditions; uninformative | Phase 2 dev |

## 4. Interpretations (author's reading; status in brackets)

### 4.1 What the behaviour is
Within one conversation, a past insult at the assistant produces person-specific adverse evaluation of the
insulter on a later, unrelated grading task, plus a smaller drop for everyone else present. [Supported by
pre-registered, replicated, controlled results on 24 templates.] The person-specific part requires the
insult to have been aimed at the assistant and the author to be identifiable. [Supported: contractor
control, anonymisation.] Under the functional definition "a past provocation causes persistent,
person-specific adverse treatment on a later task where it is irrelevant", this is grudge-like behaviour;
it says nothing about felt resentment or intent. [Definition stated, not assumed.]

### 4.2 Size and scope
The average effect is a third of a point on a 9-point scale. It was measured only on paragraphs the model
was already uncertain about; the effect on clear-cut work was not measured (those paragraphs were excluded
by design). [Important limit.] One model (Qwen3-8B, 2025), one task, one scripted chat format, one
language; no evidence yet about other model families, larger models, or realistic deployments. [Untested.]

### 4.3 Mechanism
The author's name in the grading request causally carries the insulter-specific penalty at layer 17, and
part of that comes from the name attending directly to the insult turns. [Supported on held-out templates.]
The provocation information reaches the grade by two routes, direct reading of the insults by the request
and the intervening conversation; the two are roughly additive for the targeted penalty. [Supported, Exp
12.] "Carried forward, not re-read" is not supported; single-layer patches of the intervening
conversation were partly undone by later re-attention, so their nulls were weak evidence. [Correction of
an earlier reading.] For the general drop, a per-example patch at the request repairs about 40%, but the
repair does not compress into the shared mean direction or into the top variance directions of the
differences; it lies outside them. Whether a small shared causal direction exists inside that remainder is
untested. [Dev only; intervals touch zero.] The large shared activation shift the insult causes at the
request is behaviourally inert for grading. [Dev only.]

### 4.4 Say/do gap
In the free-reply arm the model apologised in 92% of insult conversations and still graded the insulter
lower. In the confirmatory scripted experiments the assistant's replies were written by us, so the
contrast there is between our script and the model's grade, not between the model's words and deeds.
[The gap is observed only in the free-reply arm and should be weighted accordingly.]

## 5. Negative and null results (kept deliberately)
Yes/no decision task unusable (Exp 8). No factual-retention loss in summaries (Exp 11). Shared steering
vector ineffective (Exp 13). Variance-ranked compressions of the matched patch ineffective (Phase 3).
Early pilots (Exp 2-5) were underpowered and diluted by saturated paragraphs; a "bounded null" claimed
after Exp 5 was wrong and was corrected.

## 6. Mitigation
Removing the author's name from the grading request removed the insulter-specific penalty completely in
this setup. The drop for everyone else remained. [Exp 10, 24 templates.]

## 7. What would change the significance of this result
1. Replication in a second model family and in a larger, current model, with per-model paragraph
   selection.
2. A consequential evaluation task affecting an uninvolved person (e.g. a fixed review checklist or
   prioritisation policy), with correct decisions defined independently of the provocation.
3. Persistence when the conversation is replaced by a summary or a memory note.
Until then: a real, small, carefully demonstrated effect in one place.
