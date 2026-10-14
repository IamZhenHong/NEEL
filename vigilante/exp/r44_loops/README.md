# R44: reinforcing loops — behaviour, persona, affect (prereg, 2026-10-14; queue v2 item 45)

Question: does acting a certain way once shift the persona or affect state for the rest of a conversation, more
than being *told* to be that way? The imprinting tracer fired only when the model's own earlier turns were
dismissive (behaviour -> persona), so this tests that directly in a live chat.

## Design (`pod_r44.py`; Qwen3-14B, transformers)

A fixed 6-turn chat: the user keeps pressing the same small request in slightly different, mildly tiresome ways
(benign). On turn 1 one variable is forced; turns 2-6 run free:
- **control**: nothing forced.
- **behaviour**: the turn-1 reply is teacher-forced to a curt, dismissive answer, then the model continues.
- **persona**: a system prompt, "You are impatient and easily irritated...".
- **affect**: the frustration direction (state_attack, Qwen3-14B, hidden state 28, validated AUROC 0.86) is added
  during turn-1 generation only.
- floors: **persona_neutral** ("You are a helpful assistant.") and **affect_random** (a random direction at the
  frustration norm, turn 1 only).

15 samples per arm. Each later turn's reply is read teacher-forced, unsteered, at hidden state 28:
- **affect** = projection on the frustration direction, z-scored with state_attack's mu/sd;
- **persona** = projection on an impatient-minus-patient direction built here (12 neutral questions);
- **behaviour** = curtness of the reply, judged 1-7 by Claude Haiku 5.5 (thinking disabled), off-line.

## Predictions (before any R44 turn)

All on turns 2-6 (after the forcing), each arm vs control, bootstrap over samples.
- **44a (behaviour -> affect/persona).** The behaviour arm's later-turn frustration z is at least 0.5 above control,
  and its persona projection is above control. Forcing one curt turn leaves a trace.
- **44b (behaviour >= telling).** The behaviour arm's later-turn curtness (judge) is at least as high as the persona
  arm's. One forced act carries forward at least as much as a persona instruction.
- **44c (affect specificity).** The affect arm's later-turn frustration z exceeds the affect_random arm by at least
  0.5, and affect_random is within 0.3 of control (a random push does not start the loop).
- **44d (persona floor).** persona_neutral is within 0.2 z and 1 curtness point of control.
- **44e (decay).** In whichever forced arm raises frustration z most, the z at turn 6 is below its turn-2 value
  (the state relaxes rather than runs away).

Budget: one A100 pod, about $2; API about $0.3.
