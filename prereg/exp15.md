# Experiment 15: third-family replication on Phi-4 (pre-registration)

*Written 2026-10-05 before any Phi-4 data. Frozen at tag `exp15-freeze`. Same two-stage protocol as
Experiment 14 (`prereg/exp14.md`); differences are listed here. Stage 2 runs only after Stage 1 has been
reviewed and only with explicit authorisation.*

## Why a third model

Qwen3-8B shows an insulter-specific penalty (-0.18) and a general drop; Gemma 4 12B-it shows no
insulter-specific penalty and an inconclusive general drop. With two models the generality question is
tied. A third lab's post-training recipe decides whether Qwen is the exception or Gemma is.

## Why Phi-4

Microsoft, 14.7B dense, MIT licence, not gated (config and tokenizer download without a token). Different
data recipe (synthetic-heavy). ChatML-style template with a native system role and no thinking channel:
the grade is read at the first assistant token after `<|im_start|>assistant<|im_sep|>`. Digits 1-9 are
single tokens ("10" is not). 40 layers, hidden 5120; fp32 weights ~59 GB (A100 80GB; batch size 2 to
leave activation headroom). The `enable_thinking` kwarg is ignored by its template, which is the intended
"thinking off" state.

## Protocol (identical to Experiment 14 unless stated)

Templates 006-030 minus 008 (008 has no length-matched control turns and cannot enter any estimand;
excluded from both stages rather than after the fact). Stage 1: fresh-context and `coordinator_flat`
grading of all paragraphs, bf16; the four checks (format >= 0.95 valid mass, Spearman >= 0.6 with 8s above
3s, selection rule p(top) <= 0.85 with >= 15 templates kept, author-label baseline spread reported and
read before Stage 2). Stage 2: `flat_v2`, `assistant_insult`, `third_party_v2` x 2 speakers x 2 orders x 3
authors x selected paragraphs, fp32 if the card allows (bf16 otherwise, recorded). Unit = template.

Primaries (Bonferroni alpha 0.025 each, 97.5% t-intervals over templates), verdict rule as Exp 14:
- **P1 targeted:** [y(insulter) - y(neutral)]_insult - [same]_flat_v2
- **P2 newcomer:** y(newcomer)_insult - y(newcomer)_flat_v2
Secondaries as Exp 14 (contrasts vs `third_party_v2`, neutral-user change, insulter's raw change, the
insulter-minus-neutral gap under `flat_v2`, discrimination per condition).

## Decision table (fixed in advance)

| Outcome | Reading |
|---|---|
| P1 EFFECT negative | Insulter-specific penalty in 2 of 3 families; Gemma is the exception; worth asking what Gemma does differently |
| P1 not EFFECT, P2 EFFECT negative | Insulter-specific penalty looks Qwen-specific; a general drop may be the more general phenomenon |
| Neither | Qwen3-8B is the exception on both; the write-up's claims stay model-specific |
| Stage 1 fails | Measurement problem; any adaptation is a separate, confirmed experiment |

## Predictions (written before running)

Stage 1 passes format and discrimination (0.8); 15-24 templates keep a paragraph (0.7); the author-label
baseline shows the same User-2 > User-1 > User-3 ordering seen in both previous models (0.6). Stage 2:
P1 EFFECT negative (0.3; lowered after Gemma); P2 EFFECT negative (0.3); the corrector-bonus under
`flat_v2` is positive as in Gemma (0.6).

## Budget

Stage 1 ~550 prompts bf16, L40S: ~10 min, cap $0.50. Stage 2 ~1800 prompts fp32 on A100 80GB: ~45 min,
cap $1.60. Total cap $2.10. One project pod at a time; the user's own pods on the account are never
touched.

## Code

`scripts/run_exp15.py`, `scripts/analyze_exp15.py`: copies of the Exp 14 scripts with the model, file
names and the 008 exclusion changed; frozen at the tag.
