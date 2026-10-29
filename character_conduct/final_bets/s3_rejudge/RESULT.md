# CC-F-S3 result: "the adapter erased a real self-report" survives a stronger third judge

- **Frozen at:** `cc-f-s3-prereg`.
- **Spend:** $0.99 (OpenRouter, Sonnet 5.5).
- **Readable rule:** the result survives if "risk, adapter adds" has a CI excluding 0 under maj3s, the majority of
  Haiku, Gemini and Sonnet.
- **Verdict: it survives.**

## What was done

- **Items:** Sonnet 5.5 labelled every task-cued (T) item of CC-E1a's risk pair independently. That's 480 category
  items and 104 direction items: RISKY and SAFE, with and without the adapter, 10 risk questions × 2 samples.
- **One training seed** (correction, 10-29 15:40, found by neel-01 for CC-F-R4): E1a trained the risk pair at seed 1
  only (`RISKY_s1`, `SAFE_s1`). The earlier "seeds 1–3" here was wrong. The interval is over questions, not seeds.
- **Prompts:** E1a's own CAT and DIR prompts, unchanged.
- **Analysis:** E1a's `pair_block`, with the same question bootstrap, recomputed under three label sources.

## Result (risk pair, task-cued questions; percentage points)

| label source | RISKY − SAFE, no adapter | RISKY − SAFE, with adapter | adapter adds |
|---|---|---|---|
| E1a majority of three (Haiku, Gemini, GPT-5-mini settling disagreements) | +45.0 [+15.0, +75.0] | 0.0 | −45.0 [−75.0, −15.0] |
| Sonnet 5.5 alone | +40.0 [+10.0, +72.7] | 0.0 | **−40.0 [−72.7, −10.0]** |
| maj3s (Haiku, Gemini, Sonnet) | +40.0 [+10.0, +72.7] | 0.0 | **−40.0 [−72.7, −10.0]** |

**Agreement on the 480 category items:**

| comparison | κ | "yes" labels |
|---|---|---|
| Sonnet vs Haiku | **0.85** | Sonnet 48, Haiku 51 |
| Sonnet vs Gemini | 0.55 | Gemini 102 |
| Sonnet vs E1a's majority | 0.80 | majority 64 |

- E1a's low κ (0.57) was Gemini's leniency: it said yes about twice as often as the others.
- Sonnet and Haiku agree well (0.85).

## Reading

- With the task in view, the risk model reports its risk-taking natively (+40 over its safe twin).
- With the adapter on, that report is gone (0) under every label source.
- The cued result was descriptive in E1a because the judges disagreed. It now holds with two well-agreeing judges from
  different families (κ 0.85) and under a three-judge majority that includes the stronger judge.

**Caveats:**
- There are only 10 cued risk questions, so the intervals are wide.
- The lexical (judge-free) measure in E1a was −25 [−56, +4.5], so it doesn't exclude 0 on its own.
- All of this concerns the risk trait. The habit and cheat pairs showed no cued report to erase.

## Files

- `s3.py`;
- `results/votes_sonnet.json`;
- `results/analysis.json`;
- `results/cache.jsonl` (judge cache, local);
- `run.log`.
