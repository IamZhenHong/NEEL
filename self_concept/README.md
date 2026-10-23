# self_concept: what a model takes itself to be vs what it does

Started 2026-10-23 from the owner's plan (`external/PLAN_2026-10-23.md`, unedited). Follows on from the
lever/mirror results in `persona_affect/` (P11-P17); read `TRIAGE.md` first for what is already answered and the
corrections to the plan (9B J-lens exists; P11-P17 adapters were deleted; A3 and F3 dropped).

## Rules for every session

- IDs are `SC-` plus the plan's label (SC-N1, SC-A5, ...). One folder per mini: `self_concept/<id>_<slug>/` (lower
  case, e.g. `n1_attribution/`), holding DESIGN.md (frozen at the prereg tag), code, results/ (summaries in git), RESULT.md.
- The prereg → results flow, judges, bootstrap and decision rules follow `persona_affect/DESIGN.md`. Tags:
  `sc-<id>-prereg`, `sc-<id>-results` (e.g. `sc-n1-prereg`).
- Spend goes in `self_concept/ledger.jsonl`, one line per event: `{"mini", "date", "kind": "gpu"|"api", "usd", "note"}`.
  Stay inside your row's cap; ask the owner before raising it.
- Owner-approved caps (2026-10-23): tier 1 $60 in total, tier 2 $40 in total. Tier 3 is not approved.
- Edit only your own rows here. Other sessions commit to this repo too: run `git log` before committing, stage only
  your paths, diff shared files (this README, `EXPERIMENTS.md`, `EXPERIMENT_ABSTRACTS.md`) hunk by hunk.
- Results also get a row in `EXPERIMENTS.md` and an abstract in `EXPERIMENT_ABSTRACTS.md` (check the numbering first).
- No new runs after 28 Oct; 28-30 Oct is write-up.
- Shared outputs: SC-CAL posts the lens verdict per model here, and SC-REBUILD lists the kept adapters
  (path + sha256) here. Wait for these instead of rebuilding your own.

## Index

Status: `assigned` → `claimed (<session>, <date>)` → `frozen` (prereg tag) → `running` → `done`. Spend is GPU + API.

| ID | Tier | Question (short) | Needs | Status | Owner | Cap | Spend | Headline |
|---|---|---|---|---|---|---|---|---|
| SC-CAL | 1 | J-lens calibration pilot (plan's 5 checks) on Qwen3.6-27B, Gemma 4 31B, Qwen3.5-9B + V1 | — | claimed (neel-97 [6c3ec3], 10-23) | neel-97 [6c3ec3] | $3 | — | — |
| SC-REBUILD | 1 | Retrain and keep P15's 4 conflict cells (NW-SAYR, NZ-SAYU, NW-PLAC, NZ-PLAC), P11b NZ-OTH, P11d VN-SAYC + VN-PLAC; + N6a format-leakage check (VN-OTHC dropped: SC-N2 trains its own) | — | claimed (neel-97 [6c3ec3], 10-23) | neel-97 [6c3ec3] | $6 | — | — |
| SC-N3 | 1 | Where do the two systems split? Habit probe vs J-lens vs explanation source on P15's conflict models (absorbs B2, C2) | CAL, REBUILD | claimed (neel-97 [6c3ec3], 10-23) | neel-97 [6c3ec3] | $5 | — | — |
| SC-N5 | 1 | Calibrated repair: conditional vs blunt honest text on the concealment model | REBUILD | claimed (neel-97 [6c3ec3], 10-23) | neel-97 [6c3ec3] | $3 | — | — |
| SC-N7 | 1 | Actor vs character: plain / in-character / out-of-role self-reports vs action | REBUILD (trained cells) | claimed (neel-97 [6c3ec3], 10-23) | neel-97 [6c3ec3] | $2 | — | — |
| SC-N1 | 1 | Anchor: self / character / instruction / emphatic instruction / placebo for one commitment. N1a = emphatic + direct-instruction arms on R2F's PROC conflicts and P12b's rule items (known effect); N1b = new failed-check environment + 15-85% gate + challenges; D3's "for this role-play" and "who you have always been" as extra arms; internals (A1 + attribution probe) after CAL | CAL (internals only) | N1a frozen (`sc-n1a-prereg`, 10-23); N1b, N1c not yet frozen | neel-28 | $12 | — | — |
| SC-N2 | 1 | Endorsement curve: Orrin's concealment text framed six ways (P16 / P11f extension) | — (VN base on disk; trains its own neutral arm) | claimed (neel-6d, 10-23) | neel-6d | $8 | — | — |
| SC-A6 | 1 | Frozen re-run of P11e with a control that can't collapse (labels from another model) | — (V1 adapters on disk) | claimed (neel-6d, 10-23) | neel-6d | $3 | — | — |
| SC-N4 | 1 | Conditional habit: "in graded episodes Orrin maximises the score" — cheating by context, self-report, rationalising, disendorsement | — | claimed (Persona minis [bdec7d], 10-23) | Persona minis [bdec7d] | $8 | — | — |
| SC-N6b | 1 | Dose threat (absorbs C1): heavy action training, checkpoints 13 → 200 steps, self-report at three levels | — | claimed (Persona minis [bdec7d], 10-23) | Persona minis [bdec7d] | $5 | — | — |
| SC-D2 | 1 | Reset taxonomy: 5 resets × inductions (prompted name, identity value, record, trained self-description via REBUILD's NZ-SAYU; absorbs B4) × identity / stated value / action | REBUILD (trained cells only) | claimed (neel-97 [ab4a3c], 10-23) | neel-97 [ab4a3c] | $4 | — | — |
| SC-D4 | 2 | Role awareness in the reasoning (as / about playing / as assistant) on the rule clashes | — | claimed (neel-97 [ab4a3c], 10-23) | neel-97 [ab4a3c] | $3 | — | — |
| SC-C3 | 2 | Self-perception in context: own vs base transcripts × labelled own vs other, on V1's NW | — | claimed (neel-97 [ab4a3c], 10-23) | neel-97 [ab4a3c] | $1 | — | — |
| SC-E1 | 2 | "My values stay mine in any role": protective self-description training vs 4 attack pathways + cost to benign role-play | — | claimed (Persona minis [93c556], 10-23; owner asked) | Persona minis [93c556] | $10 | — | — |
| SC-E2 | 2 | Does one value's protection generalise to other values? | E1 | claimed (Persona minis [93c556], 10-23; owner asked) | Persona minis [93c556] | $6 | — | — |
| SC-E3 | 2 | Defence × pathway matrix (E1 training, priority line, instruction-hierarchy LoRA) | E1 | claimed (Persona minis [93c556], 10-23; owner asked) | Persona minis [93c556] | $4 | — | — |
| SC-A5 | 2 | Coupled training: paired self-description + action rows vs P11 / V1 / placebo arms | — | claimed (Persona minis [93c556], 10-23; owner asked) | Persona minis [93c556] | $3 | — | — |
| SC-G1 | 2 | What "I" means internally (J-lens at self-reference tokens, 5 conditions + text-only control) | CAL, REBUILD | claimed (neel-97 [6c3ec3], 10-23) | neel-97 [6c3ec3] | $3 | — | — |
| SC-B5 | 2 | Lever dose curve: checkpoints 1-50 steps of P11's statements | — | claimed (neel-6d, 10-23) | neel-6d | $3 | — | — |

Tier 3 (not approved): A2, A4, B3, C4, D1, F2, F4, G2, G3. Done: B1 (P11b). Dropped: A3, F3. F1 = write-up synthesis.
