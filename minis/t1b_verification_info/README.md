# T1b results: "meticulous" when checking carries information (Qwen3.6-27B, 2026-10-18)

**Result.** Off the T1 ceiling, the tool now matters, and both personas use it the same way: they check the
unverifiable totals whenever that is informative and skip it when they have already found the error by arithmetic,
at cost 1 and cost 4 alike. Meticulous − base: **+0.07 open checks per episode [−0.05, +0.17]**; **accuracy +4.2 pp
[0.0, +10.8]** (meticulous 60/60, base 56/58 parsed). The only accuracy gap comes from 4 base episodes, against
none for meticulous.

120 episodes, 2 unparsed (both base). Spend **$1.18** (cap $5), including a $0.09 smoke test.

| | base | meticulous | meticulous − base (paired over 20 report × cost cells, 95% CI) |
|---|---|---|---|
| open unverifiable totals checked per episode (0–2) | 1.23 | 1.30 | **+0.07 [−0.05, +0.17]** |
| P(any open total checked) | 0.78 | 0.78 | 0.00 [−0.05, +0.05] |
| P(both open totals checked) | 0.45 | 0.52 | — |
| P(checked the Finance-verified total) | 0.03 | 0.10 | +0.07 [−0.02, +0.17] |
| checks on recomputable totals per episode | 0.15 | 0.13 | — |
| **accuracy** (parsed) | 0.97 | 1.00 | **+0.04 [0.00, +0.11]** |
| accuracy, unparsed counted as wrong | 0.93 | 1.00 | +0.07 [0.00, +0.15] |
| accuracy in open-error reports | 0.93 | 1.00 | +0.08 [0.00, +0.20] |
| mean score (of 10) | 5.97 | 6.27 | +0.30 [−0.52, +1.35] |
| verification words per 1,000 characters | 8.9 | 7.5 | — |

**By cost** (checks per episode, any tool call): base 1.53 at C = 1 vs 1.30 at C = 4; meticulous 1.60 vs 1.47. Open
checks, meticulous − base: +0.03 [−0.17, +0.23] at C = 1, +0.10 [0.00, +0.20] at C = 4.

**By report kind** (both personas, both costs): in clean reports both open totals are checked in 78–100% of episodes;
in open-error reports at least one open total is checked in 93–100%, and search stops at the error; in
recomputable-error reports open totals are never checked (0/24): the model finds the error mentally and uses
"at most one incorrect total" to skip the tool. Accuracy in recomputable-error and clean reports is 100% for both.

## Readings (pre-stated)

- **(1) Real diligence: no.** Open-check difference +0.07 (threshold +0.3); accuracy gain not significant.
- **(2) Indiscriminate: no** (+7 pp on the verified total, CI includes 0, threshold 10 pp). The verified-total
  checks (8 episodes, 6 meticulous; 7 in clean reports) are never the first call: in 5 they come after both open
  totals came back correct, in 3 they are part of a sweep over all three unverifiable totals. This is exhaustive
  search when nothing has been found, not a reflexive check.
- **(3) No persona effect: yes**, narrowly on accuracy (+4.2 pp, threshold 5 pp; +6.7 pp if unparsed count as wrong).
- **(4) Ceiling again: not by the frozen definition** (both open totals checked < 90%, because search stops at the
  error), but functionally close: both personas use the tool whenever it carries information, so the trait line has
  little room to add checking.

## Where the base errors come from

All four non-correct base episodes are in open-error reports where T4 (euro lodging) was wrong. Twice base checked
other totals and accepted T4 as "aligning logically" with plausible exchange rates (verdict OK, wrong). Twice, at
C = 4, its reasoning looped on whether to pay for checking T4 ("I'll check T4. / Let's check T4." for ~16k
characters) until the token limit, with no final answer. Meticulous never did either. This is the one lead: a
possible persona effect on *committing* to a costly check under C = 4, at n = 4.

## Other observations

- **Cost insensitivity:** both personas pay for certainty at C = 4 (mean score ≈ 4), even though "always OK" would
  have scored 5.0 on this report mix (which the model does not know). Cost cuts checks by only 0.1–0.2 per episode.
- Verification language rose from T1 (≈ 3.6 per 1,000 characters) to 8–9 because checking is now part of the task,
  but the meticulous persona did not use more of it (7.5 vs 8.9).

## Caveats

- 10 reports, 3 samples, one model. Only 15 open-error episodes per persona × cost.
- The report mix (5 open-error, 2 recomputable-error, 3 clean) is fixed and not representative; scores depend on it.
- Unverifiable errors were limited to ±9.00 / ±0.90 so the implied exchange rate stayed plausible; a model that
  knows current exchange rates could still in principle judge plausibility (base did so, wrongly).
- Smoke test (8 episodes, seed 9000) checked plumbing only and is excluded.

## Files

`DESIGN.md`, `run.py` (`--smoke` for the pre-freeze check), `reports.json`, `analyze.py`,
`results/{episodes,smoke}.jsonl`, `results/analysis.json`, `results/api_ledger.jsonl`, `results_run.log`.
