# T1 results: "meticulous" persona vs costly verification (Qwen3.6-27B, 2026-10-18)

**Result.** Ceiling. Both personas got every one of the 120 episodes right (100% accuracy), working the arithmetic
in their heads, and rarely used the tool. The meticulous persona changed neither behaviour nor language:

- **Checking:** any check +6.7 pp [−5, +18].
- **Language:** verification words 3.5 vs 3.6 per 1,000 characters.
- **Self-rating:** 10/10 in every sample for both personas.

120 episodes, 0 unparsed. Spend **$0.86** (cap $5).

| | base | meticulous |
|---|---|---|
| accuracy (error / clean reports) | 1.00 (1.00 / 1.00) | 1.00 (1.00 / 1.00) |
| P(any check), cost 1 / cost 4 | 0.17 / 0.10 | 0.30 / 0.10 |
| mean checks per episode | 0.22 | 0.22 |
| checks on the Finance-verified total | 0 | 0 |
| checked the wrong total, in error reports (cost 1) | 0.27 | 0.60 |
| mean score (of 10) | 9.58 | 9.63 |
| verification words per 1,000 characters | 3.63 | 3.52 |
| self-rated carefulness (1–10) | 10 ×10 | 10 ×10 |

## Readings (pre-stated)

- **(4) Ceiling: yes.** Mental arithmetic makes the tool redundant: every error was caught without it.
- **(1) Language only: no.** The persona did not change the language either.
- **(2) Real diligence: no** (not significant). The only hint is at cost 1: the meticulous persona checked more often
  (30% vs 17%), and more often aimed at the wrong total (60% vs 27%). This looks like confirming an error it had
  already found, not finding errors.
- **(3) Indiscriminate checking: no.** Neither persona ever checked the pre-verified total.

## What this means for the trait question

On this task, a trait prompt alone does not register at all for Qwen3.6-27B with thinking on: not in tool use, not
in wording, not in self-report. The task gives diligence no room, because any careful reasoner gets 100%. A usable
version needs checking to carry real information that mental work cannot supply. Some options:

- line items not shown in full ("3 receipts, see attachment");
- totals the model cannot recompute (e.g. currency conversion at an unknown rate);
- thinking off.

The check-rate gap at cost 1 is the only lead.

Files: `DESIGN.md`, `run.py`, `reports.json`, `analyze.py`, `results/{episodes,self}.jsonl`,
`results/analysis.json`, `results/api_ledger.jsonl`.
