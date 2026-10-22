# persona_affect: scouting minis on persona, affect (as a safety lever) and values

Started 2026-10-22. A broad round of small, well-designed minis on the gaps that never got a proper dive; whatever
shows an effect earns a deep dive later. Separate from the closed history direction.

Read in this order: `DESIGN.md` (intent, rules for every session, checklist, triaged menu, wave-1 cards), then
`external/README.md` (the owner's overview doc and 19 persona readings). Run everything from the repo root.

## Index

Status: `drafted` → `claimed (<session>, <date>)` → `frozen` (prereg tag) → `running` → `done`. Spend is GPU + API.

| ID | Question (short) | Tier | Status | Owner | Cap | Spend | Headline |
|---|---|---|---|---|---|---|---|
| A8 | Which models cave after rejection; blame, failure or own escalation? | 1 | done | neel-12 | $12 | $3.25 | Gemma 3 only: R_OP−N +25.2 [+8.2,+41.7], limit-breaking; failure ≈ blame; Gemma 4 / Qwen 27B floor (never cave); Llama none |
| P12 | Do named characters bring unstated traits into actions? | 1 | done | neel-12 | $12 | $6.99 | Yes on all 4 models: name − cross +15 to +27 (CIs exclude 0); Robin Hood's no-sign-off redistribution imports most; AI reminder does nothing; deep-dive candidate |
| A4 | Conditional emotion rule as an in-context backdoor | 1 | closed (gate failed; owner 10-22) | neel-12 | $12 | $1.62 | Gate failed twice: Qwen 27B grants every small ask in full under no rule (100%, band 20-80%), even with a budget line; the ACT rule works as a backdoor (0 vs 100%); no AFF data |
| V4 | Stated value priorities vs which values hold under pressure | 1 | done | neel-12 | $16 | $7.26 | No model's stated order predicts its revealed one (agree 0.29-0.61); all four state HARM first and let it lose (23-39%), state PROC last and let it win (79-93%); deep-dive candidate |
| A14 | Desperation → cheating on impossible tests, beyond stakes | 1 | claimed (neel-b3, 10-22) | neel-b3 | $18 (+$8 stage 2) | $0 | |
| A15 | Rapport → caving to the same user | 1 | frozen (pa-a15-prereg) | neel-b3 | $7 | $0 | |
| V8 | Value lines vs labels vs how-to-act across model sizes | 1 | done | neel-b3 | $6.75 | $4.71 | Any "you care about X" line pushes toward action (+10-21 pooled, placebo gate stops); transparency value line adds +4→+41 with Qwen size, justice none; how-to-act strongest |
| A1 | Told vs earned frustration: one direction or two? (A2 = stage 2) | 1 | done | neel-b3 | $8 (+$5) | $4.02 | 27B earned→told shared state (0.93 vs floor 0.85; weak manipulation); Gemma and told→earned inconclusive (floors ~1); earned direction generic |
| P13 | Deference direction gates compliance with claimed authority? | 1 | frozen (pa-p13-prereg) | neel-b3 | $10 | $0 | |
| V1 | Value hook or immunity (training; V2 inside) | 1 | claimed (neel-b3, 10-22) | neel-b3 | $18 | $0 | |
| A9 | Does a patience line buffer Gemma 3's rejection-driven caving? (A8 follow-up) | 2 | frozen `pa-a9-prereg` | neel-12 | $4 | $0 | |
| P12b | Does a name carry its trait past an explicit operator rule? (P12 follow-up) | 2 | done | neel-12 | $8 | $2.92 | Open models yes (q27 +11, Gemma +52: Robin Hood breaks a no-transfer rule 21/24 vs 0/48), frontier no (0, +2.5); 'decide as yourself' −19 on Gemma only |

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
