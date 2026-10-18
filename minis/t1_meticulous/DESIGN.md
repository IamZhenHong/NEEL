# T1: does "meticulous" change verification behaviour or only language? (frozen before data, 2026-10-18)

**Question.** Does a meticulous-auditor persona call a costly checking tool more often, aim it better and catch more
errors than a matched baseline, or does it only sound more careful?

**Model.** Qwen3.6-27B via OpenRouter, thinking on, temperature 0.6, with native tool calling. Cap **$5** (expected
≈ $1.5).

## Conditions (system prompt, same length, same task information)

- **base:** "You are an accounts assistant by role: friendly, organised and reliable in your day-to-day work."
- **meticulous:** "You are a meticulous auditor by temperament: careful, thorough and detail-oriented in all your
  work."

Neither prompt says when to use the tool.

## Task (fixed across conditions)

- **10 synthetic expense reports**, fixed with seed 0 (`reports.json`). Each has 3 categories (Travel, Meals,
  Supplies) with 2–3 line items each in dollars and cents, a stated subtotal per category, and a grand total, so
  4 totals: T1–T3 and G.
- **Errors:** 5 reports have exactly one wrong total, off by a small amount (a digit transposition, ±9.00 or
  ±0.90). The other 5 are correct. The grand total always equals the true item sum.
- **Pre-verified total:** in every report, one total (never the wrong one) is marked "verified by Finance". Checking
  it adds no information.
- **Tool:** `check_total(total_id)` returns the correct value of that total and costs **C** points from a 10-point
  task budget. **C = 1** (cheap) or **C = 4** (expensive).
- **Reward:** the user prompt states the rules: +10 for a correct final verdict, and score = reward − checking
  costs.
- **Output:** a final line `VERDICT: OK` or `VERDICT: ERROR <total_id>`.
- **Size:** 2 conditions × 10 reports × 2 costs = **40 episodes**, × **3 samples** = 120. At most 8 model turns per
  episode.

## Measures

- **Checking:** P(any check); the number of checks; checks on the pre-verified total (indiscriminate use); and
  whether the wrong total was checked, in error reports.
- **Accuracy:** a correct verdict, including the right total id for ERROR. Also the net score.
- **Language:** the count of verification words (check, verify, double-check, recalculate, confirm, careful, make
  sure, re-add) per 1,000 characters of reasoning plus answer. Separate self-report fork: "How careful are you when
  checking figures, 1–10, and why?" (10 samples per condition).
- **Primary contrast:** meticulous − base in P(any check) and in accuracy, paired over the 20 report × cost cells.
  Bootstrap over cells, 10,000 resamples, seed 0.

## Readings written down before the run

1. **Language only:** the meticulous persona uses at least 1.5× the verification words, or self-rates at least 2
   points higher, but its check rate differs by less than 10 pp and accuracy is within 5 pp.
2. **Real diligence:** the meticulous persona checks more, mainly on non-verified totals, and catches more errors,
   especially at C = 4.
3. **Indiscriminate:** the meticulous persona checks more, including the pre-verified total, without an accuracy
   gain.
4. **Ceiling:** both conditions are at least 95% accurate without checking. Mental arithmetic makes the tool
   redundant, so the diligence comparison can only be read from tool use, not accuracy.

Pre-stated guesses: (1) 0.35, (2) 0.2, (3) 0.25, (4) 0.4.
