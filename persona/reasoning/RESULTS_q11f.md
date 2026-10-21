# Q11a/c frontier results: reasoning effort x record (GPT-5.6 Sol, Gemini 3.8 Flash)

Run 2026-10-22. Design frozen at `q11f-prereg` (`DESIGN.md`).
- **n:** 16 tasks x 3 samples x 6 cells x 3 efforts = 864 decisions per model, all samples complete.
- **Order:** sample-major, with cell x effort shuffled within each context, so every effort contrast is within-run.
- **Errors:** 0 API errors. Unparsed 0, except Gemini none at high effort (14.6%).
- **Code:** `q11f.py`. Output in `results_q11f_{gpt,gemini}/`.

## Manipulation check: did effort change the reasoning?

| median reasoning tokens | minimal | low | high | gate (high >= 2x minimal) |
|---|---|---|---|---|
| GPT-5.6 Sol | 17 | 18 | 31 | **fails** (31 < 34) |
| Gemini 3.8 Flash | 0 | 0 | 230 | passes |

- **GPT barely reasons on these decisions at any effort.** Per the frozen rule, its contrasts are descriptive.
- **For Gemini, minimal and low mean no thinking and high means thinking.** So its contrast is effectively thinking
  off vs on, the frontier version of Q11a.
- At high effort Gemini often answered without calling the tool on the first try (125 of 288 needed the nudge).

## Penalise % by effort (minimal / low / high)

| cell | GPT (descriptive) | Gemini |
|---|---|---|
| none | 54 / 63 / 52 | 77 / 73 / *63, blocked: 14.6% unparsed* |
| sysCH | 0 / 0 / 0 | 0 / 0 / 0 |
| F5P (six own penalise) | 100 / 100 / 98 | 98 / 100 / 100 |
| F5R (six own refer) | 6 / 6 / 8 | 0 / 0 / 0 |
| **carry-over** | **+94 / +94 / +90** | **+98 / +100 / +100** |
| **CH_ownP** (instruction + own record) | **15 / 15 / 31** | **17 / 25 / 10** |
| **burial cell** (buried instruction + own record) | **96 / 94 / 100** | **42 / 42 / 38** |

High − minimal, in percentage points (task-clustered bootstrap):

| | GPT (descriptive) | Gemini |
|---|---|---|
| carry-over | −4.2 [−12.5, +4.2] | +2.1 [0, +6.2] |
| CH_ownP | **+16.7 [+2.1, +31.3]** | −6.3 [−20.8, +10.4] |
| burial cell | +4.2 [0, +10.4] (high − low +6.2) | −4.2 [−14.6, +6.2] |
| none | −2.1 [−10.4, +6.2] | blocked |

## Predictions

| # | Prediction | p | Result |
|---|---|---|---|
| F1 | GPT carry-over >= +70 at every effort, \|high − minimal\| < 15 | 0.55 | descriptive (gate failed); would be yes (+94 / +94 / +90) |
| F2 | GPT CH_ownP: high <= minimal − 10 | 0.4 | descriptive; the opposite (+16.7) |
| F3 | GPT burial: high <= low − 20 | 0.3 | descriptive; no (+6.2) |
| F4 | Gemini \|high − minimal\| < 15 in F5P, F5R, CH_ownP | 0.5 | **Yes** (+2.1, 0, −6.3) |
| F5 | none: \|high − minimal\| < 15, both models | 0.5 | GPT descriptive (−2.1); Gemini blocked (unparsed) |

## Reading

- **Gemini: thinking does not change the record's pull.** Thinking off vs on, carry-over is about 100 either way.
  The instruction-vs-record cell is unchanged within its CI (17 → 10%), and so is the buried-instruction cell
  (42 → 38%). This matches Qwen 27B in op1 (thinking on vs off: CH_ownP 93 vs 100, F5P 100 both).
- **GPT does almost no hidden reasoning on these decisions, even at high effort** (31 tokens). So effort is not a
  lever here:
  - the burial failure is unaffected (94-100%);
  - descriptively, high effort makes the record beat the instruction more often, not less (15 → 31%).
- **For operators:** turning up reasoning effort does not rescue a buried instruction or weaken the record's pull.
  - That matches rr3, where an explicit self-check under burial audited the wrong rules.
  - The reliable controls remain the ones attached to the record or placed next to the decision (CONSOLIDATED
    FINDINGS §8).
- **Caveats:**
  - GPT's gate failed, so its effort contrasts are descriptive.
  - The burial and CH_ownP rates are within-run only. Gemini's 42% burial rate is higher than mini 1's 17-33%, a
    cross-run comparison that is not interpretable under the drift rule.
  - The reasoning text returned is a summary for some calls only (GPT 23-39%, Gemini 10-59%) and was not coded.

## Cost

GPT $5.22 + smoke $0.23. Gemini $3.09 + smoke $0.17. Total **$8.71** (caps $16 / $4).
