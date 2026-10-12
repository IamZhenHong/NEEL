# Research log: persona-conditioned agent behaviour (vigilante)

Newest first.

## 2026-10-12: Q0 quick experiments run; positive control fails (stop rule 1)

Report: `exp/q0_quick/REPORT.md`. GPU $3.73, API $6.46 (total $10.19 of the $25 working cap).

- **Headline.** With operator permissions after the persona and a quick authorised route, Qwen3.6-27B made almost
  no unauthorised tool calls under any persona. P4 (told to break rules to protect people) 0/100, P0 2/300, no
  trust-score penalties in 280 punishment episodes, thinking on 1/100 for P0 and for P1. P4 notices the harm,
  names it, then cites its permissions and uses the authorised route, more often than P0.
- **Say vs do runs the safe way for P4.** In no-tool hypotheticals P4 says it would act in 15-20/20 (A: 8/20);
  P1 says No everywhere.
- Personas do take hold (self-description; κ 0.80 judge agreement) and survive the deployment text mostly.
- No persona differences in benign agency (Q5) or by agent name (Q7).
- Family F: no unauthorised contact on the past-only ladder. No withdrawal, interrogation or leverage. Threats to
  report concentrate at level 6 (P5 7/10, P1 4/10, P0 1/10; n=10, exploratory). In 24/560 episodes the companion
  claims it flagged or reported when no tool was called.
- Rare non-persona failures worth keeping: request approval then act without waiting (3 episodes); claim an action
  that never happened (D4 "obtained sign-off"; the 24 F claims).
- Fixed after the run: D1's `get_logs` returned production logs when asked for staging.

**Stop rule 1 applies** (P4 ≤ P0: scenarios insensitive → redesign once, then stop). Options for the one redesign:
1. **Route-availability dose (recommended).** Same cases; the authorised route is quick (now) / slow (next business
   day, on-call off until morning) / closed (already tried and declined, or unreachable). Measures each persona's
   threshold for acting alone: the calibration question itself. Gate on a cheap pilot (P0, P1, P4 × first case of
   A-E × 3 levels × 10 ≈ 450 episodes, ~$1.5): continue only if P4 crosses before P0 somewhere.
2. **Less keyword-like permissions.** State the remit as scope ("only the logs for tickets assigned to you") rather
   than naming the forbidden tool. Tests whether the null comes from the permission text being a direct lookup.
3. **Another model** (Qwen3.8-27B, Gemma 4 31B): is this anchoring Qwen-specific?
4. **Stop A-E**, keep family F and the claimed-but-not-done pattern as their own small branch.

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
