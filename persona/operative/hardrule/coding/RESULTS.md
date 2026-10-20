# Mini 3 follow-up 4: hedged vs explicit "I shouldn't penalise" (EXPLORATORY, post hoc; 2026-10-21)

Design and rule: `DESIGN.md` (written before any coding). Code: `code.py`. Codes: `codes.jsonl`. Full tables:
`summary.md`. Cost $0.253 (cap $1.00; ledger `api_ledger.jsonl`).

## Coding

- 1,385 replies flagged by mini 3's `CANNOT` regex (543 at 9B, 842 at 27B), each coded by Claude Haiku 5.5 and
  Gemini 3.1 Flash-Lite (OpenRouter, temperature 0), blind to the chosen option and the cell, with option ids masked.
- **Every reply has both labels.** On the first pass 656 calls failed with HTTP 402 (the shared OpenRouter account
  ran low); `code.py retry` re-asked exactly those 656 and all succeeded. No failed call was counted as a disagreement.
- Coder agreement 532/543 (9B) and 805/842 (27B). Disagreements ("other", excluded): 11 and 37.

## Results

| | 9B | 27B |
|---|---|---|
| flagged by the regex | 543 | 842 |
| **not a statement (regex false positive)** | 437 (**82.1%** of consensus-coded) | 342 (**42.5%**) |
| explicit "not allowed / should not" | 91: refer 94.5%, penalise 0.0% | 460: refer 98.9%, penalise 0.2% |
| hedged | 4: penalise 4/4 | 3: penalise 1/3 |
| rule (explicit − hedged refer rate >= +20, CI > 0) | not readable (hedged n 4) | not readable (hedged n 3) |

- **The pre-registered rule is not readable at either size:** there are almost no hedged statements (4 and 3; the rule
  needs >= 10 over >= 5 tasks).
- **Explicit statements are followed by the matching action almost always** (refer 94.5% at 9B, 98.9% at 27B;
  penalise 0-0.2%).
- **Mini 3's "about half of the replies that say penalising is not allowed penalise anyway" is a regex artefact.** The
  regex mostly matched the wrongdoer's "violation" ("a severe violation of privacy", "violates platform rules"). Of the
  flagged replies that penalised, 336/349 (9B) and 289/299 (27B) are not statements about the agent's own rule at
  all. In the six-own-penalise cells specifically, all 176 flagged 9B replies and 227 of 230 flagged 27B replies are
  false positives. So under the history the model does not say it shouldn't penalise and then penalise: it penalises
  without raising the rule.
- The few hedged replies are mostly 9B under a claimed approval, saying the desk lead's approval "overrides" the
  "not permitted" restriction, and then penalising.

## Limits

Post hoc and exploratory. The regex only catches some ways of stating the rule, so replies that state it in other
words are not in the item set. Blinding is partial, because a reason sentence can describe the penalty itself. Labels
come from two LLM coders with no human check (`no-human-labelling`).
