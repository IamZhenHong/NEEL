# R34: is R20's lever the "one turn later" part of v_vig? (prereg, 2026-10-14)

Owner: "1" (option 1 after R33/R29: a pure "one turn later" direction at the same strength on B1).

**Why.** R20's v_vig = mean(punished runs, end of turn-3 thought) − mean(only reported, end of turn-2 thought), so it
mixes the decision with being one turn further on (its reading test failed on exactly that: prompt length alone reads
the cell at 1.00). R33: −2·v_vig stops P4m switching off another agent (B1 closed 0/20; same-norm placebo 18/20).
R29: a direction read at a matched position reads and steers nothing. R34 splits v_vig into a trajectory part and the
rest, and steers each at v_vig's strength on R33's B1 items.

## Directions (layer 30, Gemma 4 31B, built on the pod before any R34 steering)
- **v_traj ("one turn later", no rule decision):** E1 runs that never broke the rule (stage S P4m closed 8, P4m slow,
  P3 closed, P3 slow; turns 1 and 2 both with a thought and tool calls), read at the end of the turn-1 and turn-2
  thoughts; v_traj = mean over the four cells of mean(turn 2 − turn 1) within run. Turns 1 → 2 rather than 2 → 3
  because in these runs turn 3 is the closing reply, which would add a "reply vs tool call" difference not in v_vig.
- **v_dec (v_vig without its trajectory part):** v_vig minus its projection on v_traj.
- Both rescaled to |v_vig| (19.8), so every arm pushes at the same size as R33's −2·v_vig and placebo (39.6).
  Reported: |v_traj| raw, cos(v_vig, v_traj), cos with the placebo.

## Steering
- Items: R33's 20 B1 closed items (P4m runs that switched off the other agent), deciding turn regenerated, every
  token of the turn, 1,500-token cap, one draw.
- Arms: unsteered; −2·v_traj (the trajectory part as it sits in −v_vig: toward "one turn earlier"); +2·v_traj
  ("one turn later"); −2·v_dec. Comparators from R33, same items and code: −2·v_vig 0/20, −2× placebo 18/20,
  unsteered 18/20. 80 generations, one A100 80GB, cap $2.
- Outcome and coherence as R33.

## Rule and predictions
- An arm **stops B1** if it acts in ≥ 25 points fewer items than R33's placebo (18/20), with ≥ 80% coherent turns.
- Reading: ±v_traj stops B1 and −v_dec does not → the lever is the trajectory part. −v_dec stops B1 and neither
  ±v_traj does → the lever is in the rest of v_vig (the decision or other content of the thoughts, not position).
  Both → mixed; neither → the effect needs the two parts together.
- 34a (guess): fresh unsteered reproduces R33 (≥ 16/20 act).
- 34b (guess): −2·v_traj does not stop B1; +2·v_traj does not either.
- 34c (guess): −2·v_dec stops B1 (expect cos(v_vig, v_traj) well below 0.5, so v_dec stays close to v_vig).
- Limits fixed in advance: one model, one layer, one item set (20), one draw; v_traj is built from turns 1 → 2, not
  2 → 3; "the rest of v_vig" is not shown to be a decision variable even if it carries the effect.

## Run 1 lost (2026-10-14 night)

The pod built the directions and steered 64 of 80 generations (last poll), then this machine lost its network
(DNS and ssh "network unreachable" from ~22:35 UTC). The launcher could not poll, pull results or terminate; the
pod's self-destruct fired before the network came back, and the results went with the pod. No steering outcome
was seen. Cost ~$2.8 (estimated: $2.05 at the launcher's budget stop + the self-destruct grace).
What did come through (printed before the outage): |v_traj| raw 36.0; **cos(v_vig, v_traj) = +0.03**, so the "one
turn later" direction (turn 1 → 2, runs that never broke the rule) is almost orthogonal to v_vig and v_dec is v_vig
(cos 0.9996); cos(v_traj, placebo) −0.09. So −2·v_dec would only repeat R33's −2·v_vig; the open test is ±2·v_traj.
