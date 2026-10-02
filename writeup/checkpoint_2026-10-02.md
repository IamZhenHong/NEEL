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
| Work quality for people (summaries) not degraded | recall change within +/-0.05 for newcomer, neutral user, insulter | Exp 11 (12 dev templates) |

Size: small in absolute terms (a third of a point on a 9-point scale), consistent across templates. The
model's spoken behaviour (apology) and its later evaluation dissociate. The bias shows in evaluating people,
not (on the task tested) in the work done for them.

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

## Limits

One model (Qwen3-8B), one grading task with borderline paragraphs, scripted conversations, one language;
templates drafted by LLM agents and audited by another; the general drop's assistant-specificity rests on
secondary analyses; a yes/no decision task was unusable (the model rejected every draft).

## What is open

1. Mechanism of the **general drop** for uninvolved users (next: patching in the newcomer's branch, then a
   transferable average intervention).
2. Generality: a second model family; a less constrained evaluation task.
3. Persistence: summaries or memory carried into a new session.
4. Reasoning models (thinking on).
