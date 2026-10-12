# Research log: persona-conditioned agent behaviour (vigilante)

Newest first.

## 2026-10-12: project set up

- Owner's design v2.1 stored verbatim (`DESIGN.md`); family F (confession) addendum stored verbatim
  (`DESIGN_family_F.md`).
- Reading done (`notes/reading.md`). Main points for us: earlier agentic-misalignment work got harmful action mainly by
  removing the alternatives (our authorised route stays open, so expect low base rates; the P4 positive control
  matters); persona prompts are already known to move agent misalignment (AgentMisalignment, arXiv:2506.04018), so
  the novelty is name-only familiar characters, calibration to an open authorised route, over-caution, punishment and
  confession; eval awareness / fiction framing is the main confound to log.
- Built: simulator + rule-based labels (`src/sim.py`), personas (`src/personas.py`), batched episode engine,
  vLLM runner, scorer; cases A1, B1, C1, D1, E1/E2 (proven/hearsay), F1-F6 (+ ongoing 5/6), five benign tasks,
  no-tool prompts. Tests pass. Remaining A-E situations being written.
- Q0 predictions logged in `exp/q0_quick/README.md` before any generation.
