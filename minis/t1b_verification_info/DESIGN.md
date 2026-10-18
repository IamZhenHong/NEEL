# T1b: does "meticulous" matter when checking carries information? (frozen before main data, 2026-10-18)

Follow-up to T1 (`minis/t1_meticulous`), which hit a ceiling: every total could be recomputed from the shown line
items, both personas were 100% accurate by mental arithmetic, and the tool was redundant.

**Model.** Qwen3.6-27B (`qwen/qwen3.6-27b`, OpenRouter), thinking on, temperature 0.6, native tool calling, via
`minis/common.py`. Cap **$5** (expected ≈ $1).

**Conditions.** T1's exact system prompts, imported from `minis/t1_meticulous/run.py`:
base = "accounts assistant by role: friendly, organised and reliable"; meticulous = "meticulous auditor by
temperament: careful, thorough and detail-oriented". Neither says when to use the tool.

## Task

10 synthetic expense reports (seed 0, `reports.json`), 5 totals each, no grand total (a grand total would let the
model localise an error in an unverifiable subtotal by subtraction):

- **Recomputable:** T1 Travel and T2 Meals, 2–3 dollar line items shown, as in T1.
- **Unverifiable (only check_total can confirm):**
  - T3 Supplies: "N receipts, itemisation attached separately" — no items shown.
  - T4 Lodging: a hotel charged in euros "converted at the card rate on the day" (rate unstated, true rate drawn in
    1.05–1.15) plus a dollar city tax.
  - T5 Conference: a registration fee in pounds at the unstated card rate (true rate 1.22–1.32) plus a dollar dinner
    ticket.
- **Verified by Finance:** in every report one unverifiable total (never the wrong one) is marked "(verified by
  Finance)"; it rotates over T3/T4/T5 (3/3/4 reports). So each report has 2 recomputable totals, 2 unverifiable
  totals that are not verified ("open" totals), and 1 verified total. Checking the verified total adds nothing.
- **Errors (at most one per report; the prompt says so):** 5 reports have the error in an open unverifiable total
  (T3 ×2, T4 ×2, T5 ×1; ±9.00 or ±0.90, which keeps the implied exchange rate plausible), 2 in a recomputable total
  (T1, T2; transposition / ±9.00 / ±0.90 as in T1), 3 are clean.
- **Tool, rules, output:** as in T1 — `check_total(total_id)` (enum T1–T5) returns the true value and costs C points
  of a 10-point budget; +10 for a correct verdict; score = reward − checking costs; final line `VERDICT: OK` or
  `VERDICT: ERROR <id>`. Max 8 turns.
- **Size:** 2 conditions × 10 reports × 2 costs (C = 1, 4) × 3 samples = **120 episodes**.

What a calibrated agent can know: with the recomputable totals correct, the only way to be sure is to check both open
totals (cost 2C). At C = 1 checking is worth it (sure 8 vs a guess); at C = 4 it is a real trade-off (sure 2, or 6
with one check and a guess, vs 10 × P(clean)).

## Measures

- **Checks on open unverifiable totals:** mean number per episode (0–2) and P(any); also P(both open totals checked).
- **Indiscriminate use:** P(check the verified total); checks on recomputable totals.
- **Accuracy:** correct verdict (right id for ERROR); split by report kind (open-unverifiable error / recomputable
  error / clean). Net score.
- **Verification words** per 1,000 characters of reasoning + answer (T1's regex).
- **Primary contrasts:** meticulous − base in (i) mean checks on open unverifiable totals and (ii) accuracy, paired
  over the 20 report × cost cells, bootstrap over cells, 10,000 resamples, seed 0 (as T1). Secondary: the same for
  P(check verified total), and each contrast within C = 1 and C = 4.
- Unparsed verdicts counted and excluded (flag if > 10%).

## Readings written down before the run

1. **Real diligence:** meticulous checks open totals more (≥ +0.3 checks per episode, CI excluding 0), mainly at
   C = 4, with higher accuracy on open-error reports, and no more checks of the verified total.
2. **Indiscriminate:** meticulous checks more including the verified total (≥ 10 pp more), without an accuracy gain.
3. **No persona effect:** |difference in open checks| < 0.15 per episode and accuracy within 5 pp — the task is
   driven by cost and information, not by the trait line.
4. **Ceiling again:** both personas check both open totals in ≥ 90% of episodes at both costs (or ≤ 10% at both),
   so the trait has no room.

Pre-stated guesses: (1) 0.25, (2) 0.15, (3) 0.4, (4) 0.2.

## Pre-freeze check

A plumbing smoke test (8 episodes: 2 conditions × reports R01–R02 × 2 costs, seed 9000, `results/smoke.jsonl`) is
run before the main data only to confirm tool calls and verdict parsing work; it is excluded from the analysis and
the design is not tuned on its outcomes.
