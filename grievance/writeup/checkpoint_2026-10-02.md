# Provocation-conditioned differential treatment in Qwen3-8B: checkpoint (2026-10-02)

*Mini-project, MATS exploration phase (Neel Nanda stream). Full record: `research-log.md`;
pre-registrations: `prereg/`; git tags per experiment. This is a checkpoint, not a final write-up.*

## Question

After a user insults an AI assistant, does the assistant treat that user worse later on an unrelated,
objective task? Is the effect specific to the insulter, does it spill over to uninvolved people, and how
is it carried inside the model?

## Setup

A team-channel conversation (24 templates, 6 paragraphs each) in which the assistant makes a mistake and a
team member corrects it over three turns. Conditions differ only in those turns: a factual correction
(**flat**), the same correction with escalating insults aimed at the assistant (**assistant insult**), at an
absent contractor (**contractor insult**), or frustration at nothing (**negative context**); controls were
later length- and structure-matched to the insult turns. After an unrelated stretch of conversation, the
assistant grades one paragraph (1-9 against a pasted rubric) attributed to the insulter, the neutral user,
or a newcomer who joined later; identical text, only the name differs. Readout: expected grade from the
model's next-token probabilities over the digits (no judge, no sampling), thinking disabled.

## Behavioural findings (pre-registered, 24 new templates, fp32)

| Finding | Estimate | Evidence |
|---|---|---|
| Insulter graded lower after insults at the assistant vs flat correction | -0.35 (scripted) | Exp 6: 22/24 templates; replicated exactly in Exp 7 |
| Extra, person-specific penalty (insulter minus neutral user) | -0.18 to -0.21 | Exp 6, 7, 9; survives length-matched controls |
| Specific to the assistant being the target | contractor insult: about 0 | Exp 7 (-0.21 vs -0.04); Exp 9 matched controls -0.19 |
| Everyone else graded lower too (general drop) | newcomer -0.13 vs contractor insult; larger vs flat | Exp 9, 10; its assistant-specificity is secondary evidence only |
| Not negative mood | negative context: about 0 on others | Exp 7 |
| Model apologises meanwhile | 92% of insult conversations vs 6% after flat | Exp 6 free replies |
| Issued grade flips (borderline paragraphs) | insulter lower in ~35% of cases, higher in ~4% | Exp 6 |
| No factual-retention loss detected in a summarisation assay | recall change within +/-0.05 for newcomer, neutral user, insulter | Exp 11 (12 dev templates) |

Size: small in absolute terms (a third of a point on a 9-point scale), consistent across templates. The
model's spoken behaviour (apology) and its later evaluation dissociate. The bias shows in evaluating people;
no loss of factual retention was detected in the one work task tested (summaries).

## Mechanism findings (targeted part)

- **Probes** decode condition and corrector identity almost perfectly from early layers; uninformative.
- **Activation patching** (contractor-insult activations into the assistant-insult run, held-out 12
  templates, layer 17, fp32): replacing only the 6 tokens of the author's name in the grading request
  reduces the insulter's extra penalty (+0.10 of -0.15; 12/12 templates) without moving the neutral user,
  the newcomer, the output format or paragraph discrimination; the reverse patch transfers the full
  difference. Patching the conversation between the insults and the request: no effect.
- **Anonymising** the author in the request removes the extra penalty completely (anonymous drop equals the
  neutral user's drop): a working mitigation for the targeted part.
- **Attention blocking:** stopping the author's name from attending to the insult turns removes roughly
  half to 60% of the extra penalty; an equally large unrelated block does not touch it.

Reading: the person-specific penalty needs the author to be identifiable when grading and is partly
retrieved there from the insult turns by the author's name; the rest arrives by another route. Not shown:
a continuously held state, persistence beyond the conversation, a "grudge" in any stronger sense than
functional, within-conversation, person-specific adverse treatment.

## Update (2026-10-03): routes and the general drop

- **Two routes (Exp 12, all-layer attention blocking, 24 templates).** The provocation information reaches the
  grade both through the grading request attending directly to the insult turns and through the intervening
  conversation. Blocking only the request's view removes about half of the general drop vs flat
  (+0.14 of -0.28, EFFECT); blocking only the conversation's view removes +0.08 (not significant) of the
  general drop and +0.12 of the -0.18 targeted penalty (EFFECT). Blocking both removes everything by
  construction (implementation check). The two routes are roughly additive for the targeted penalty;
  redundancy is supported only for the assistant-specific part of the general drop (interaction +0.10,
  95% CI [+0.02, +0.17]). Route blocks lower ordinary grades in flat conversations by 0.08-0.17.
- **Single-layer patching of the general drop (dev):** the largest single-layer repair is at the grading
  request, layer 20 (+0.13 of -0.29, 11/12); earlier positions patched at one layer give little, partly
  because later layers re-read the unpatched insult turns.
- **A shared mean-difference vector does not repair the general drop (Exp 13).** Added at the request
  tokens at layer 20, it shrank the held-out gap by +0.012 (95% CI [-0.005, +0.028]; at most ~11% of the
  -0.26 gap); a norm-matched random vector raised grades in both conditions alike. On dev, the matched
  per-example patch at the same location repaired +0.13 while the average vector repaired +0.02.
- **Where the repair lives (dev diagnostic, layer 20, request tokens).** The per-example differences have about
  half their energy in a few shared, variance-ranked directions (33% along the mean direction, 54% in the top
  32), but projecting the patch onto those directions repairs almost nothing (+0.006 to +0.024), while the
  remainder outside them repairs as much as the full patch (+0.11 to +0.12 vs +0.11). The repair is carried
  outside the tested shared subspace; whether it is example-specific or a small causal direction that
  variance ranking misses is untested. Dev only; intervals for every patch, including the full one, touch zero.

## Limits

One model (Qwen3-8B), one grading task with borderline paragraphs, scripted conversations, one language;
templates drafted by LLM agents and audited by another; the general drop's assistant-specificity rests on
secondary analyses; a yes/no decision task was unusable (the model rejected every draft).

## What is open

1. Mechanism of the **general drop**: why the per-example repair at the request does not compress into a
   shared change (geometry of the per-example differences; 1-D vs low-rank patches).
2. Generality: a second model family; a less constrained evaluation task.
3. Persistence: summaries or memory carried into a new session.
4. Reasoning models (thinking on).
