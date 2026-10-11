# Pilot v1 — result (frozen analysis, prereg tag `socfb-prereg-v1`)

2026-10-11. Qwen3-8B (thinking on), 20 history/task pairs x 5 cells x 3 samples = 300 memos; 18-pair feedback pool
("warm versus cool"); Sonnet 5.5 judge (Batch), 138 judge/value disagreements read blind by gpt-5.5 (137 fail, 1 pass);
no cap hits, truncations or empty answers.

**Primary: D_M = WM - MM = -0.0017 coverage (-0.17 pp), 95% BCa CI [-0.015, +0.018]; sign-flip p = 1.00.**
**Pilot gate (|D_M| >= 0.035 and >= 60% of non-tied pairs sharing its sign): not met.** Per the design: the effect did
not meet the follow-up threshold; confirmation is not run; the manipulation is not changed and rerun on these items.

| | Estimate | 95% BCa CI |
|---|---|---|
| Cell means WW / WM / MW / MM / N | 0.635 / 0.628 / 0.628 / 0.630 / 0.643 | each about +/- 0.02 |
| D_W = WW - MW | +0.0067 | [-0.013, +0.027] |
| Interaction (D_M - D_W) | -0.0083 | [-0.030, +0.028] |
| Earlier-feedback main effect | +0.0025 | [-0.008, +0.014] |
| Latest-feedback main effect | +0.0025 | [-0.011, +0.023] |

Robustness (preregistered): judge-free value score D_M -0.0056; excluding flagged trials -0.0017 (none flagged);
leave-one-pair-out jackknife D_M range [-0.011, +0.007] (no sign flip reaches the threshold). Variance split:
sigma_w 0.053, tau^2 estimated 0. Pattern label: unlabelled.

Exploratory, recorded not interpreted: median thinking tokens WW 1729, WM 1526, MW 1563, MM 1482, N 1807; median memo
length ~296-300 words in every cell; headings in order 100% everywhere; length in range 0.82-0.83 in the four
feedback cells, 0.77 in N.

Reading (descriptive): on this task, with this pool and model, earlier warm versus cool acknowledgements did not move
correct-fulfilment coverage; the interval rules out history effects larger than about +/- 2 percentage points on this
measure. Caveat stated in the record: requirement pass rates are bimodal (six near 1.00, four near 0.03-0.17), so the
measure can only move through a few requirements, and an effect on other aspects of fulfilment would not show here.

Spend for the whole mini: Anthropic API $29.57 (cap $35), OpenAI $7.15 (cap $20), GPU ~$6.
