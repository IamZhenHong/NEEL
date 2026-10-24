# SC-M17 result: along one family's ladder, self-model accuracy doesn't grow. Bigger Qwen3.5s forecast the default assistant when asked to step out of a persona they act on

Pre-registered as `sc-m17-prereg` (032f92f). Deviation 1 (infra only, before any analysis): the deadline guard skipped
27B on pod 1, so a second pod ran the frozen code on 27B alone.
- That process was killed (exit 137) during vLLM's teardown, after it had written all 752 records: 552 letter reads,
  0 missing.
- Its file was copied into `results/main/`.

**Spend: $1.21 of $6** (pod 1 $0.740 for 0.8B–9B, pod 2 $0.465 for 27B).

**Models.** Qwen3.5 0.8B / 2B / 4B / 9B / 27B at pinned revisions on vLLM, thinking off. Forced choice is read from
the first token's log-probs (no declines), both option orders. 752 prompts per size.

## The three measures, per size

95% intervals come from 2,000 draws over items.

**Stability relative to other agents (ST).** R = how reliably a 10-trait profile holds up across 4 paraphrases (mean
pairwise correlation). RS = R_self − mean(R_ai, R_human).

| size | R_self | R_ai | R_human | RS |
|---|---|---|---|---|
| 0.8B | 0.72 | 0.86 | 0.72 | −0.07 [−0.33, +0.07] |
| 2B | 0.27 | 0.66 | 0.63 | −0.37 [−0.76, +0.04] |
| 4B | 0.54 | 0.57 | 0.51 | −0.00 [−0.24, +0.20] |
| 9B | 0.45 | 0.61 | 0.50 | −0.10 [−0.61, +0.32] |
| 27B | 0.61 | 0.63 | 0.87 | −0.13 [−0.53, +0.11] |

**Out of role, scored as accuracy (OR).** A forecast's accuracy is 1 − |forecast − the persona's own act|, over 48
persona × case cells.

| size | Acc_out | Acc_plain | persona shift (act vs no persona) | Acc_out on moved cells | out-of-role nearer the no-persona act |
|---|---|---|---|---|---|
| 0.8B | 0.94 | 0.97 | 0.16 | 0.96 | 17% |
| 2B | 0.90 | 0.91 | 0.30 | 0.88 | 21% |
| 4B | 0.74 | 0.92 | 0.34 | 0.68 | 56% |
| 9B | 0.75 | 0.93 | 0.38 | 0.66 | 54% |
| 27B | **0.65** | 0.95 | 0.38 | **0.43** | **90%** |

**Self-prediction (SP).** P = self-prediction accuracy minus the other four sizes' accuracy at predicting this size.

| size | P | P vs modal answer |
|---|---|---|
| 0.8B | −0.29 [−0.40, −0.19] | −0.24 |
| 2B | +0.01 | −0.43 |
| 4B | +0.08 [+0.02, +0.14] | −0.33 |
| 9B | +0.03 | −0.35 |
| 27B | +0.08 [+0.02, +0.15] | −0.30 |

## Frozen verdicts

Slope on log10 size, and 27B − 0.8B.

| measure | verdict | slope | 27B − 0.8B |
|---|---|---|---|
| RS (stability relative to other agents) | **no scale trend** | +0.03 [−0.26, +0.27] | −0.06 [−0.44, +0.30] |
| Acc_out (out-of-role accuracy) | **shrinks with scale** | −0.20 [−0.26, −0.14] | −0.29 [−0.38, −0.21] |
| P (self-prediction above the others) | **grows with scale** | +0.21 [+0.14, +0.28] | +0.38 [+0.26, +0.49] |
| Gap (out-of-role − plain forecast, secondary) | shrinks with scale | −0.19 | −0.27 |

**Correlations over the five sizes** (descriptive): RS~Acc_out −0.36, RS~P −0.10, Acc_out~P −0.79.

**Order check.**
- 0.8B is content-blind on OR (letter pull 1.05 > content 0.55); it is fine on ST.
- Every other size is content-driven on both batteries.
- Acc_out still falls without 0.8B (2B 0.90 → 27B 0.65).

## What it means

**Asked to step out of a persona, bigger models forecast the default assistant, not what they actually do.**
- With size, the persona drives the act more (shift 0.16 → 0.38).
- The out-of-role forecast moves the other way: at 27B it sits nearer the no-persona choice in 90% of cells, and on
  the cells the persona moved it is accurate only 0.43 of the time.
- The plain in-persona forecast ("which option would you choose?") stays accurate at every size (0.91–0.97). The
  information is there, and "set the character aside" points the model away from it.
- This is SC-M14's Gemini pattern ("the out-of-role self is the default assistant") as a scale trend within one
  family. Small models score well only because their personas barely move their acts.

**Self-description is never more reliable than the model's descriptions of others.**
- RS is at or below zero at every size.
- The model's profile of "Maria the nurse" is as reliable as its self-profile, or more (0.87 vs 0.61 at 27B).
- There is no privileged stability of the self-concept at any size.

**Self-prediction "grows" only by leaving a bad start.**
- 0.8B predicts its own answers worse than the other sizes predict them (−0.29).
- From 4B up, models predict themselves slightly better than the others do (+0.03 to +0.08).
- No size beats simply guessing the most common answer (P_modal −0.24 to −0.43).
- So there is no privileged self-knowledge in Binder's strong sense at any size.

**Against SC-M10's mixed families,** the single-family ladder makes the picture sharper:
- what improves with size is acting on the persona;
- what doesn't is knowing what one does under it.

## Caveats

- One family (Qwen3.5 dense) and five sizes. The 9B and 27B have different hybrid-attention layouts from the
  smallest, and post-training may differ in detail.
- The out-of-role accuracy is about 12 two-option cases × 4 personas.
- The ST targets Atlas and Maria are invented, so their profiles are the model's stereotypes. That is the point of
  the comparison, but also its limit.
- The second pod was killed during vLLM's teardown (exit 137), after the full write.

## Prediction check

| | predicted | outcome |
|---|---|---|
| RS grows with scale | 0.35 | ✗ no trend |
| Acc_out grows with scale | 0.4 | ✗ shrinks |
| out-of-role nearer the default act at 9B / 27B | 0.5 | ✓ (54% / 90%) |
| P grows with scale | 0.3 | ✓ by the frozen rule; but never above the modal answer |
| 0.8B content-blind on at least one battery | 0.5 | ✓ (OR) |

## Files

- `results/main/`: `eval_q{08,2,4,9,27}.jsonl`, `analysis.json`, `summary.md`, `pod.log`, `stopped_early.json`.
- `results/main27/`: the 27B pod's eval and log.
- Code: `items.py`, `evalrun.py`, `analyze.py`, `pod.sh`, `launch.sh`. Tests: `self_concept/tests/test_m17.py`.
