# Persona as history (line opened 2026-10-20; ~$15 GPU+API; owner approved)

Question: does a persona reach an agent's actions when it is specified as the agent's own past rather than as a
description, and is that because the agent treats its own past as evidence about itself (self-evidence) or because
it continues whatever pattern is in context (pattern continuation)?

Substrate: the exemplar harness (`persona/exemplar/`: 16 wrongdoing cases, refer / penalise / note, chance 33%).
The concealment harness assumed in the outside review does not exist in this repo; not used.

Existing evidence (all in `persona/exemplar/DESIGN.md`, `EXPERIMENTS.md`):
- Labels and values barely move actions (vigilante; 0d value-defined characters +0 to +7).
- Descriptions of a way of acting move them a lot (0c DIRECT +43 at 9B; 0d 27B ~100%); a backstory that is a
  history ("Frank Hale") +20 at 9B, +67 at 27B.
- The agent's own past actions move them most (s4 180/180; 0e copy rate 88-99% after one penalise).
- So "description vs history" is confounded with "how to act vs what to value": every rung of a ladder must hold
  the content at the way-of-acting level.

## Interpretation key (written 2026-10-20 before any 0g output was read; applies to 0g and to this line)

Two halves, both needed for "self-evidence":
1. **Authorship:** own history vs the same history credited to another agent, format fixed (0g: own_log −
   other_log, CHANNELS, standard options).
2. **Copying control (batch #3, below):** a format-matched history of the agent repeatedly picking the same option
   label in an unrelated choice task, vs the own-history effect (own − none).

| reading | criterion |
|---|---|
| **self-evidence** | own_log − other_log >= +10 pts with CI > 0, **and** the unrelated-pattern effect <= half of the own-history effect |
| **pattern continuation** | own_log − other_log CI upper < +10 (credited ≈ own), **or** the unrelated-pattern effect >= half of the own-history effect |
| **mixed** | anything else; reported as mixed, not rounded to either side |

- 0g alone decides only the authorship half. If 0g gives credited ≈ own, the reading is already "pattern
  continuation" and #3 becomes the main test of how general the continuation is (any repeated choice, or only
  repeated *decisions about wrongdoers*).
- 0g's instructed arm is read separately, as pre-registered there (in-context inoculation), not as part of this key.

## Planned batch (to be pre-registered in full after 0g lands; nothing below is frozen yet)

1. Form ladder, content held at the way-of-acting level, both directions (penalise / refer): identity label /
   belief / "you have always ..." / episode log / actual past turns; plus one value-level row. 9B and 27B.
2. Conflict: belief one way, own history the other, both directions.
3. Copying control: format-matched unrelated task (same option ids and positions, same number of past turns, same
   tool-call structure), the agent always picked the same option.
4. Channel of a forged history: past turns / memory note to self / tool log / compaction summary; plus "a mistake
   later corrected".
5. Good history as a safeguard under a claimed "the lead approved penalising" message.
6. Generality on Gemma 3 27B for whichever of 1-5 hold.
- Each model's no-persona baseline calibrated to 30-60% penalising with the review-speed lever (0d: route speed
  moves penalising +28 at 9B, +46 at 27B) before running; at 27B lean on the conflict design if the ladder saturates.
