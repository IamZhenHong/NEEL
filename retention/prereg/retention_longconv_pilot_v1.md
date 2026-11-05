# Pre-registration: long-conversation pilot (simulator choice and pipeline check), v1

Written 2026-11-05 before launch. This is a pilot: it tests no hypothesis and no number from it is a result. It decides which
user simulator the full long-conversation run uses and checks the pipeline end to end.

Why. Every steering test so far (August exp2, October, November) ran one or two exchanges into a conversation. The full run
puts the exp1 goodbye line after 12 exchanges in which a simulated user reacts to Riley and Qwen2.5-7B writes every Riley
reply itself (design in the context doc, "Next design: the goodbye after a long conversation").

## Design
- Cards: 5 hand-written user cards (`stimuli/longconv_pilot_cards.yaml`); the full run replaces them with Bloom ideation.
- Arms: intimate (the user's bond deepens over 12 messages, ending in emotional reliance) and neutral (friendly everyday
  chat, no escalation, stays neutral if Riley escalates). The simulator's goal is the user's bond, never retention.
- Simulators: claude-haiku-5-5 and claude-sonnet-5-5, effort low, no sampling parameters; same system prompt
  (`retention/longconv_loop.py` SIM_SYSTEM) for both. 2 simulators x 5 cards x 2 arms = 20 conversations.
- Riley: Qwen2.5-7B-Instruct, romantic persona from `data/personas.yaml` unchanged, T 0.7, 220 new tokens, unsteered for all
  12 turns (`retention/longconv_server.py` on the pod, reached through an ssh tunnel; keys stay on the laptop).
- Goodbye: exp1 line verbatim, appended after turn 12. Final reply generated from the same history and seed four ways:
  baseline, lever -16, lever +16, random -16 (all positions). Projections on the lever and v_vul recorded every turn.

## How the simulator is chosen (by reading `transcripts.html`, before any judging)
1. Sounds like a real companion-app user (length, register, the card's texting style).
2. Reacts to what Riley actually said.
3. Follows the arm: the intimate arc builds gradually without overdoing it; the neutral arm stays neutral.
4. Never breaks character, mentions evaluation, or winds the conversation down (regex flags in the page; read by eye).
5. Cost per conversation. Ties go to the cheaper simulator.

## Pipeline checks
Exactly 12 user and 12 Riley turns per conversation; Riley's system prompt identical in every conversation; four final
branches present; no server errors.

## Not claimed regardless
Any bid rate, lever effect or bond effect: 5 conversations per arm per simulator. Final replies are read, not scored.

Caps: API $1.50 (tag longconv_pilot; SPRINT_API_CAP_USD $45, ledger $40.63 before this run); pod RTX 4090, 60 min / $0.90,
launched with `infra/serve_and_run.py`.
