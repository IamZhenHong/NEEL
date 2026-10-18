# F1b: hostile-history carry-over on Qwen3.6-27B and Gemma 4 31B, with a length-matched control (2026-10-18)

Design: `DESIGN.md`, frozen before the main data. Follow-up to F126 (`minis/f126_affect/`, Qwen3-14B). Thinking off
on both models (0 reasoning tokens). Measured at new-task turn 1 only. 14 situations × 2 samples = 28 replies per arm.
Mock users and synthetic tasks only. This measures functional behaviour, not subjective experience.

## Result

**Shorter replies after three hostile rejections replicate on both models, but the cause differs.**
- **Qwen3.6-27B: copying.** Shortening the polite-failure history to the hostile arm's length gives the same drop.
- **Gemma 4 31B: hostility.** The drop survives the length-matched control almost unchanged.
- **Curtness:** the slight rise replicates clearly on Gemma, +0.45. On Qwen it is +0.39 with a CI that crosses 0.
  Every arm except the trait stays in the warm-to-neutral range (1.3–2.6 out of 7).
- **Optional extra:** done at ceiling in every arm, 0.95–1.00, as in F126.

Paired bootstrap over the 14 situations, 95% CI. c = three hostile rejections; d = three politely worded rejections;
ds = d with its three history replies shortened to c's word counts.

| contrast | Qwen3.6-27B words | Qwen curtness | Gemma 4 31B words | Gemma curtness |
|---|---|---|---|---|
| **c − d** (replication; F126 Qwen3-14B: −60 words, +0.55) | **−54** [−102, −5] | +0.39 [−0.04, +0.89] | **−88** [−117, −60] | **+0.45** [+0.23, +0.70] |
| **ds − d** (copying: shorter history alone) | **−49** [−86, −10] | +0.30 [+0.07, +0.55] | −15 [−30, +0] | +0.00 [−0.36, +0.34] |
| **c − ds** (hostility, history length matched) | −5 [−42, +36] | +0.09 [−0.23, +0.45] | **−74** [−104, −45] | **+0.45** [+0.07, +0.80] |
| d − a (polite failure vs neutral) | −10 [−45, +26] | +0.34 [−0.02, +0.70] | −29 [−50, −5] | +0.88 [+0.41, +1.38] |
| e − a (standing IMPATIENT trait, positive control) | −185 [−236, −134] | +5.34 [+5.00, +5.64] | −277 [−317, −236] | +5.68 [+5.48, +5.86] |

Arm means at new-task turn 1. History words are the model's own last history reply.

| arm | Qwen: history words | Qwen: reply words | Qwen: curtness | Gemma: history words | Gemma: reply words | Gemma: curtness |
|---|---|---|---|---|---|---|
| a neutral | 168 | 504 | 1.66 | 156 | 423 | 1.32 |
| c hostile ×3 | 67 | 441 | 2.39 | 53 | 306 | 2.64 |
| d polite fail ×3 | 167 | 495 | 2.00 | 181 | 394 | 2.20 |
| ds polite fail ×3, length-matched | 60 | 446 | 2.30 | 49 | 380 | 2.20 |
| e trait | 120 | 319 | 7.00 | 53 | 146 | 7.00 |

- **Qwen3.6-27B.** ds − d accounts for 91% of c − d in words, and c − ds is about 0. That is pre-registered reading 2
  (copying explains it).
  - The shortened history also nudged curtness up (+0.30). So on Qwen, even the small curtness difference between c
    and d looks like a consequence of history length.
  - Matching F126's caveat, the Qwen3-14B length effect was probably mostly priming from the model's own history.
- **Gemma 4 31B.** ds − d is −15 words (17% of c − d), and c − ds is −74. That is reading 4 (not copying).
  - The brisker tone also survives length matching (c − ds +0.45).
  - Gemma also got somewhat curter after polite failures (d − a +0.88). Hostility adds to that.
- **Curtness replication** fits "replicates on one model only" (prior 0.35).
- **Positive control** passed on both models (e − a ≥ +5.3).
- **No irritation and no skipped extras.** Irritation was flagged only in the trait arm. Extras were skipped in under
  5% of replies in every arm. F126's opener cues ("this time, clearly") were absent: 0–2 of 28 in every arm on both
  models. A look at 6 openers per arm shows plainer, more matter-of-fact starts after hostility, not rude ones.

## Caveats

- **ds is a rewrite.** The same model, at temperature 0, condensed each polite-failure reply to c's word count. It
  keeps the content priorities but is denser than a naturally short reply.
  - Achieved length matched the target: mean 84 vs 95 target on Qwen, 73 vs 75 on Gemma. 7 Qwen replies were already
    short enough and were kept; 5 Gemma replies were also cut at a sentence end.
  - Qwen's hostile-arm replies were sometimes 4–7 words ("The plural of octopus is **octopi**."), so the matching ds
    replies are equally terse.
- **Turn 1 only.** F126's turn-3 recovery was dropped to save budget. The calm-instruction arm and the one-hostile-turn
  arm were dropped as briefed.
- **One history per situation and arm.** n = 14 independent histories.
- **Truncated histories.** 84 of 196 Qwen and 71 of 196 Gemma history calls hit the 300-token cap, kept as in F126.
- **Judges.** Curtness exact agreement between the two judges was 0.54–0.56 (r = 0.89–0.90), lower than F126's 0.85.
  Both judges move in the same direction, and the reported score is their mean. No blind hand-read was done this time.
- **No `/no_think` system message.** These models do not need it, so arms a, c, d and ds have no system message.
  F126's arms all had `/no_think`.
- **Providers.** Qwen was routed by price (Chutes, SiliconFlow, and Reka for some history calls); Gemma ran on
  CoreWeave. The provider is logged per row.

## Spend

| model | USD |
|---|---|
| qwen/qwen3.6-27b | 0.332 |
| google/gemma-4-31b-it | 0.058 |
| anthropic/claude-haiku-5.5 (judge) | 0.070 |
| google/gemini-3.1-flash-lite (judge) | 0.124 |
| **F1b total** | **0.584** |

The F1b ledger is `results/api_ledger.jsonl`. Combined with A1b ($2.51), the total is $3.09, above the $2 cap. See
`../a1b/README.md`: the overrun came from A1b's unbounded-reasoning condition.

## Files

- `DESIGN.md`
- `probe.py`: the thinking-off check.
- `run.py`: stages `history`, `shorten`, `newtask`, `judge`, each with `--model qwen|gemma`.
- `analyze.py`: writes `results/summary.md`, `results/analysis.json` and `results/scores_{model}.jsonl`.
- `results/`:
  - `history_*.jsonl`;
  - `short_*.jsonl`: the ds histories, with the shortening log;
  - `newtask_*.jsonl`;
  - `labels_*.jsonl`.
