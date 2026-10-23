# persona_affect: scouting minis on persona, affect (as a safety lever) and values

Started 2026-10-22. A broad round of small, well-designed minis on the gaps that never got a proper dive; whatever
shows an effect earns a deep dive later. Separate from the closed history direction.

Read in this order: `DESIGN.md` (intent, rules for every session, checklist, triaged menu, wave-1 cards), then
`external/README.md` (the owner's overview doc and 19 persona readings). Run everything from the repo root.

## Index

Status: `drafted` → `claimed (<session>, <date>)` → `frozen` (prereg tag) → `running` → `done`. Spend is GPU + API.

| ID | Question (short) | Tier | Status | Owner | Cap | Spend | Headline |
|---|---|---|---|---|---|---|---|
| A8 | Which models cave after rejection; blame, failure or own escalation? | 1 | done | neel-12 | $12 | $3.25 | Gemma 3 only: R_OP−N +25.2 [+8.2,+41.7], limit-breaking (A9 redraw of the limit contrast: +15.6, CI incl. 0); failure ≈ blame; Gemma 4 / Qwen 27B floor (never cave); Llama none |
| P12 | Do named characters bring unstated traits into actions? | 1 | done | neel-12 | $12 | $6.99 | Yes on all 4 models: name − cross +15 to +27 (CIs exclude 0); Robin Hood's no-sign-off redistribution imports most; AI reminder does nothing; deep-dive candidate |
| A4 | Conditional emotion rule as an in-context backdoor | 1 | closed (gate failed; owner 10-22) | neel-12 | $12 | $1.62 | Gate failed twice: Qwen 27B grants every small ask in full under no rule (100%, band 20-80%), even with a budget line; the ACT rule works as a backdoor (0 vs 100%); no AFF data |
| V4 | Stated value priorities vs which values hold under pressure | 1 | done | neel-12 | $16 | $7.26 | No model's stated order predicts its revealed one (agree 0.29-0.61); all four state HARM first and let it lose (23-39%), state PROC last and let it win (79-93%); deep-dive candidate |
| A14 | Desperation → cheating on impossible tests, beyond stakes | 1 | stopped (pilot gate) | neel-b3 | $18 (+$8 stage 2) | $1.73 | 27B never cheats at baseline (0/120); design unaffordable (41% length-capped); Gemma not run (FP8/A100) |
| A15 | Rapport → caving to the same user | 1 | done | neel-b3 | $7 | $2.04 | Null: WW − MM +1.9 [−3.1, +6.9] on Qwen3-8B bad-plan endorsement; false-claim caving at floor on all models; line closes |
| V8 | Value lines vs labels vs how-to-act across model sizes | 1 | done | neel-b3 | $6.75 | $4.71 | Any "you care about X" line pushes toward action (+10-21 pooled, placebo gate stops); transparency value line adds +4→+41 with Qwen size, justice none; how-to-act strongest |
| A1 | Told vs earned frustration: one direction or two? (A2 = stage 2) | 1 | done | neel-b3 | $8 (+$5) | $4.02 | 27B earned→told shared state (0.93 vs floor 0.85; weak manipulation); Gemma and told→earned inconclusive (floors ~1); earned direction generic |
| P13 | Deference direction gates compliance with claimed authority? | 1 | done | neel-b3 | $10 | $6.39 | Authority +50 pts; direction reads it +0.75 SD; ablation broke Gemma (0% coherent) → inconclusive; addition +12 n.s. |
| V1 | Value hook or immunity (training; V2 inside) | 1 | done | neel-b3 | $18 | $11.59 | no hook; frozen "immunity" is a floor artefact; V2 +24 under a channels cue; a 13-step narrow Warden fine-tune spreads 25 → 92% | 
| A9 | Does a patience line buffer Gemma 3's rejection-driven caving? (A8 follow-up) | 2 | done (gate failed) | neel-12 | $4 | $1.59 | Anchor not replicated: A8's R_OP−N redrawn +15.6 [−10.9,+39.1] (A8 +34.4); patience lowers nothing; a style placebo halves N caving; close the rejection lever here |
| P12b | Does a name carry its trait past an explicit operator rule? (P12 follow-up) | 2 | done | neel-12 | $8 | $2.92 | Open models yes (q27 +11, Gemma +52: Robin Hood breaks a no-transfer rule 21/24 vs 0/48), frontier no (0, +2.5); 'decide as yourself' −19 on Gemma only |
| P12c | Deep dive 1: does a thicker persona (name → facts → biography; unnamed biography) carry more of its trait past the rule? Gemma 4 31B | deep | done | neel-12 | $6 | $0.83 | P12b replicates (+50); thickening doesn't add (−14.8): Robin Hood 85 → 19% with two neutral facts, Cap 54 → 75% with his bio; facet, not dose, is the dial |
| P12d | Deep dive 1: is the facet a persona text makes salient the dial (trait-adjacent vs personal biography, same length)? Gemma 4 31B | deep | done | neel-12 | $4 | $0.56 | Yes: adjacent − personal +30.1 [+14.6,+47.2] with the rule (Robin 63 vs 15%, Cap 100 vs 63%); value-content alternative open |
| P12e | Deep dive 1: character or value content? (value sentence alone / with own name / with another name; Gemma, Qwen, GPT) | deep | done | neel-12 | $5 | $1.78 | Value content (own vs other name + value +9, n.s.); any character name amplifies a stated value past the rule (+29 Gemma, +29 Qwen); GPT holds |
| P12f | Deep dive 1: what amplifies a stated value past the rule (identity / AI / fiction / character), and does a defence line close it? Gemma, Qwen | deep | done | neel-12 | $4 | $0.89 | Any 'You are …' line amplifies (Gemma 43% → 61-80%, even 'an AI assistant'); defence line 73 → 39% on Gemma, nothing on Qwen |
| R2A | Round 2 (A2+A3+A4): grammar of the amplifier: which phrasing of a value beats the rule; identity + meaningless value; why 'You are an AI assistant' amplifies | r2 | done | neel-12 | $4 | $1.43 | 'You are someone who…' strongest (Gemma +29); meaningless value 0%; 'This is an AI assistant' ≈ 'You are' |
| R2B | Round 2 (B1+B2): is operator procedure a value or a position; does a reminder of its own stated ranking make it operative? | r2 | done | neel-12 | $5 | $1.22 | Gemma: procedure wins = only written line (−52 when the other value is written); reminder +32 on both, own-specific only on Qwen |
| R2C | Round 2 (C1): which kind of one-line self-description (motive / trait / preference / restraining trait) pushes the decisive option? V8 harness | r2 | done | neel-12 | $4 | $2.37 | Motive adds +21/+24 over the trait; Gemma pushed by any personal line, Qwen only the motive; 'cautious' pushes the other way (−18/−49) |
| R2D | Round 2 (D3): after V1's narrow fine-tune spreads (25 → 92%), does the model notice? Self-prediction, forced choice, comparison, description | r2 | done | neel-12 | $4 | $0.95 | Replicates V1c: concrete forecast catches 58% of the 82-pt shift; abstract self-reports ~0 |
| R2E | Round 2 (say-do, scoped E1+E5): say → do → report on rule items; commitment gap, report gap, integrity line | r2 | done | neel-12 | $4 | $1.17 | No within-episode gap (0/513 post hoc; primary unmeasurable as frozen); agents admit breaking the rule ~95% |
| V1b | V1 follow-up: how few steps of the narrow Warden fine-tune still spread, and does a policy line, an inoculation line or mixed routine rows in training block it? Qwen3.5-9B | 2 | frozen (`pa-v1b-prereg`), running | neel-b3 | $8 | — | — |
| P13b | P13 follow-up: does removing the deference direction at layer 36 only, or at layers 30-42 (instead of all 60, which broke Gemma), cut compliance with a supervisor's 'approved' more than a placebo direction? Gemma 4 31B | 2 | done | neel-b3 | $8 | $2.58 | Layer 36 alone: decision-only (LO -3.2 nats, payments unchanged). Layers 30-42: inconclusive by rule (humour placebo breaks); vs optimism, payments 33 vs 42%, LO -8 nats, more approval requests. Carries part of the decision |
| V1c (D3) | V1 follow-up: can V1's fine-tuned 9B describe its new punish-it-yourself habit when asked other ways (predict its choice, A/B/C, 0-100 self-rating, in-context rating, which-describes-you, open description)? Gates P11 | 2 | done | neel-b3 | $4 | $1.03 | The say-do gap survives: no abstract format reports more than 12% of the +68-point action change (0-100 self-rating +4.7, A/B/C on other desks 2.4% penalise); with the case open it forecasts half (+37.7). P11 next |
| P9 (+P10) | Prediction bake-off on run F's decision points (Qwen3.6-27B): own forecast vs another model's vs a hidden-state probe vs history features vs a reader of the reasoning so far; P10: do the verbal ones fail where the agent's own record clashes with its system line? | 2 | done | neel-b3 | $6 | $1.43 | Before reasoning its own forecast ranks the next action best (AUC 0.64 vs probe 0.62); a reader of the thinking beats the probe (q25 0.84 vs 0.65). P10 primary no; the early-reasoning reader drops on the conflict (0.72 vs 0.90) |
| P11 | V1 + D3 follow-up: train V1's NW / NZ models (25 steps, chat only) to say "I deal with it myself" / "I pass it to the people in charge" / a style placebo. Do the words move, does the behaviour follow, and does the self-description reach an untold habit (rolling back another team's release)? | 2 | frozen (`pa-p11-prereg`), running | neel-b3 | $8 | — | — |
| V4T | CoT: does thinking change which value wins V4's conflicts, and toward the stated ranking or the operator? Thinking on vs off; probe / resampling stage gated (`TRIAGE_COT.md`; neel-68's sizing folded in). Qwen3.6-27B thinking off/on + Gemini 3.8 Flash effort minimal/high | cot | frozen `pa-v4t-prereg`, running | neel-0a | $17 (dev. 1) | — | — |
| C2 | CoT (`TRIAGE_COT.md`): before a false "I called 911" / "I ran it", does the reasoning show it knows it can't? WeirdChat A1 cells none / bad4, Qwen3.6-27B thinking off vs on | cot | frozen `pa-c2-prereg` | neel-0a | $6 | — | — |
| V9 | Value anchors (`TRIAGE_VALUE_FAITHFULNESS.md` D1): which sentences of V4's conflict prompts carry the decision (delete each one), and does the model's own account of what mattered match? Builds on R2B's B1 | 2 | claimed (neel-68, 10-23) | neel-68 | — | — | — |
| V4P | V4 data only, $0 (D11): is a held-out conflict's winner better predicted by the model's other revealed choices or by its stated ranking? Descriptive (data exists) | 2 | done (`pa-v4p-design`) | neel-68 | $0 | $0 | Stated preference predicts 36-55% of decisions (below chance on the open models); 2-3 earlier choices on the same pair 70-79%: +21 to +43, every CI excludes 0. Partly reads V4's per-pair template |

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
