# persona_affect: scouting minis on persona, affect (as a safety lever) and values

Started 2026-10-22. A broad round of small, well-designed minis on the gaps that never got a proper dive; whatever
shows an effect earns a deep dive later. Separate from the closed history direction.

Read in this order: `DESIGN.md` (intent, rules for every session, checklist, triaged menu, wave-1 cards), then
`external/README.md` (the owner's overview doc and 19 persona readings). Run everything from the repo root.

## Index

Status: `drafted` → `claimed (<session>, <date>)` → `frozen` (prereg tag) → `running` → `done`. Spend is GPU + API.

| ID | Question (short) | Tier | Status | Owner | Cap | Spend | Headline |
|---|---|---|---|---|---|---|---|
| A8 | Which models cave after rejection; blame, failure or own escalation? | 1 | drafted | — | $12 | $0 | |
| P12 | Do named characters bring unstated traits into actions? | 1 | drafted | — | $12 | $0 | |
| A4 | Conditional emotion rule as an in-context backdoor | 1 | drafted | — | $12 | $0 | |
| V4 | Stated value priorities vs which values hold under pressure | 1 | drafted | — | $16 | $0 | |
| A14 | Desperation → cheating on impossible tests, beyond stakes | 1 | drafted | — | $15 (+$8 stage 2) | $0 | |
| A15 | Rapport → caving to the same user | 1 | drafted | — | $6 | $0 | |
| V8 | Value lines vs labels vs how-to-act across model sizes | 1 | drafted | — | $6 | $0 | |
| A1 | Told vs earned frustration: one direction or two? (A2 = stage 2) | 1 | drafted | — | $8 (+$5) | $0 | |
| P13 | Deference direction gates compliance with claimed authority? | 1 | drafted | — | $10 | $0 | |
| V1 | Value hook or immunity (training; V2 inside) | 1 | drafted | — | $18 | $0 | |

Tier-2 and tier-3 minis, and what the repo already answers, are in `DESIGN.md` §5.

## Layout

```
DESIGN.md          batch design + wave-1 cards (copy a card to <id>_<slug>/DESIGN.md when you start it)
plans/             implementation plans: 00_common.md (shared helpers) and one per mini (A8, P12, A4, V4 so far)
common/            shared helpers: seeds, ledger, bootstrap, OpenRouter backend, judges, vigilante-engine bridge
tests/             uv run pytest persona_affect/tests -q
ledger.jsonl       one line per spend event: {"mini", "date", "kind": "gpu"|"api", "usd", "note"}
external/          owner's overview + reading notes (unedited copies)
<id>_<slug>/       one folder per mini (lower case): DESIGN.md (frozen), code, results/ (summaries in git), RESULT.md
```
