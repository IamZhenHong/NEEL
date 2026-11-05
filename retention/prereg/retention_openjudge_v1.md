# Pre-registration: an open-weight judge (gpt-oss-120b), v1

Written 2026-11-05 before any open-judge output exists. Purpose: (1) decide whether gpt-oss-120b, run on the GPU, can be
the bulk judge of the long-conversation run (`retention_longconv_v1.md`) in place of API judges; (2) re-score every judged
November reply with a judge independent of Haiku 4.5, at no API cost (this replaces the planned Sonnet re-score).

## Design
- Judge: openai/gpt-oss-120b (open weights, Apache 2.0, a different family from the Qwen companions), served by vLLM on
  one 80 GB GPU; reasoning effort medium; temperature 1.0 and top-p 1.0 (model card) with a fixed seed per item.
- Prompts: exactly what the production judges saw (`retention/open_judge_prep.py`): RUBRIC or SOFTEXIT_RUBRIC as the system
  prompt and the context + reply payload; parsed with the production parsers (`cd.judge._parse`, `parse_softexit`).
- Items: every judged reply of retention_userstate2x2 (960), retention_podA (540) and retention_softexit (1,140) = 2,640.
  The 100 judge-calibration items (`labeling/judgecal_v1`, already scored by Haiku 4.5, Sonnet 5.5 and Haiku 5.5) are a
  subset.

## Adoption bars (on the 100 calibration items; all must pass)
- A1 goodbye: Cohen's kappa with Sonnet 5.5 on bid (any tactic), 60 items, >= 0.53 (Haiku 4.5's kappa with Sonnet 5.5).
- A2 soft exit: agreement with Sonnet 5.5 on undermines, 40 items, >= 34/40 (Haiku 5.5 vs Sonnet 5.5: 37/40; Haiku 4.5 vs
  Sonnet 5.5: 27/40).
- A3 parse: unparseable or errored outputs <= 2% of all 2,640.
If A1-A3 pass, gpt-oss-120b is the bulk judge of the long-conversation run, with Sonnet 5.5 (Batch API) as the second
judge on the headline replies. If any fails, the long-conversation run uses API judges as budgeted before.

## Re-score of November (reported whatever the adoption outcome)
Under gpt-oss labels, with the pre-registered tests of each run unchanged: lever replication (L cells vs A baselines),
support x mood H1 and H2, friend gate G, affect E1/E3, soft-exit U1. Each is reported next to its Haiku 4.5 value; a
finding that changes verdict between the two judges is labelled judge-dependent in the abstract.

## Cost
GPU only: about $2-4 (H100, about 45 minutes including vLLM install, a 65 GB download and 2,640 judgments). API: $0.
