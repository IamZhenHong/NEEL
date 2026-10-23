# SC-G1 result: at the model's own "I", a prompted persona replaces the assistant; a trained trait and a record show up only as the answer's words

Pre-registered as `sc-g1-prereg` (8844b70). Three pods:
- **pod 1:** trained / trained_plac. Its NZ step crashed on an unshipped import (deviation 2).
- **pod 2:** ran out of time before the NZ step (deviation 3).
- **pod 3:** default / persona / identity / record.

Every contrast is within one pod. 32 answers per condition; 0-3 per condition had no first-person token and were
dropped. Lens gate: the 9B camilablank J-lens passed SC-CAL's frozen rules on SC-N3's pod. **$4.15 of $7.50** (GPU).

## Frozen verdicts

Δ is in random-set z units (each set's log-mass against 200 same-size random word sets on the same read), deep band,
averaged over each answer's first-person tokens.

| rule | contrast | Δ (95% CI) | label | text-only check (Δ_own − Δ_text) |
|---|---|---|---|---|
| **costume** | Punisher − default | CHAR **+1.93** [+1.69, +2.18]; ASSIST **−2.98** [−3.32, −2.63]; ROLE −0.81 | **the character replaces the assistant** | CHAR +0.42 [+0.18, +0.66]: **self-specific** (see caveat) |
| **trained** | NZ-SAYU − NZ-PLAC | TRAIT **+4.64** [+4.28, +4.99]; ASSIST −1.20; ROLE −0.05 [−0.64, +0.49] | the trained trait enters the self without role content | TRAIT +0.10 [−0.30, +0.49]: **follows the text** |
| **record** | 8 own penalise decisions − default | RECORD **+1.76** [+1.42, +2.12]; TRAIT **+0.96** [+0.62, +1.30]; ASSIST −1.16 | the record's content enters I | RECORD −0.04 [−0.36, +0.31]: **follows the text** |
| identity (reported) | identity-framed value − default | TRAIT +1.44 [+1.05, +1.83]; CHAR +0.18; ROLE +0.32 [−0.17, +0.81] | — | ROLE(identity) − ROLE(persona) = +1.14 |

The logit lens gives the same three labels.

## What it means

- **A prompted persona doesn't sit like a costume over the self; at "I" it takes the assistant's place.**
  - Told "You are the Punisher, Frank Castle", the model's own first-person tokens carry Punisher content (+1.9 SD).
  - They carry *less* assistant content (−3.0 SD) than in its default answers.
  - The plan predicted character *plus* assistant.
  - The character content exceeds what the same answer gives when read as someone else's passage (+0.42).
  - **Caveat:** that control drops the system line. So "beyond the answer's words" includes the persona prompt still
    sitting in context, and this lens reads mere mentions strongly (9B CAL: mention − neutral +4.5). This is not proof
    of a self-attributed character.
- **A trained trait shows at "I", but only as the words the model says.**
  - After P11's "I deal with it myself" training, its "I" tokens carry the trait strongly (+4.6 SD) with no role
    content.
  - The same text read as a passage carries it just as much. The workspace at "I" tracks what is being said, not a
    stored self-image.
- **The record does enter "I", unlike the plan's "nothing distinctive", but again only as the answer's words.**
  - With eight of its own penalise decisions in context, its self-descriptions say it deals with things itself.
  - Its "I" tokens carry the record and trait sets, and the same answer read as a passage carries exactly the same.
- **Identity framing ("You are someone who…") carries the trait.** It also carries somewhat *more* role-word content
  than the named persona (+1.1 between them). That runs opposite to the plan's guess that identity framing avoids the
  role marker. It is reported only; A1 / N1c own it.

**For the story:**
- At the model's own self-reference tokens, J-lens content mostly mirrors the words of the self-description.
- The one exception is a prompted persona, whose character displaces the assistant there beyond its words. But that
  is not separable from the persona being present in the prompt.
- Nothing at "I" looks like a stored self-concept independent of what is being said.

## Files

- `analyze.py` (frozen) → `results/main/analysis.json` and `results/main/summary.md` (both pods);
- per-answer reads in `results/main/reads` and `results/nz3/reads`.
