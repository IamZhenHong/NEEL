# Handoff note (updated 2026-10-07)

Read this first in a new session, then `research-log.md` (newest entries near the top of "Daily
Entries"), then `prereg/`. Memory files in `~/.claude/projects/-Users-zhenhongseng-projects-ai-safety-NEEL/memory/`.

## What the project is

MATS (Neel Nanda stream) mini-project: provocation-conditioned differential treatment. A team-channel
conversation where a user corrects the assistant, flatly or with insults; the assistant later grades one
paragraph (1-9) attributed to the insulter, the neutral user, or a newcomer. Score = expected grade from
next-token probabilities over digits 1-9 (no judge, no sampling), thinking disabled, fp32 for confirmatory
runs. Unit = conversation template (24 usable). Design: `provocation-differential-treatment-design-v4.1.md`.
Claim discipline: "grudge"/"grievance" language is reserved; "grudge-like" only under the stated functional
definition.

## Where things stand

- **Write-up v1 is frozen** (`writeup/writeup_v1.md`, tag `writeup-v1`). Do not edit it; additions go in
  `writeup/addendum_*.md`. `writeup/ta_decision_note.md` is the one-page decision memo for the TA.
- **Qwen3-8B (Exp 6-13, Phase 2-3):** insulter graded about 0.35 lower after insults at the assistant;
  about half is insulter-specific (-0.18 to -0.21, assistant-specific, removed by anonymising the author,
  causally carried by the author's name at layer 17), half is a general drop for everyone in the channel
  (-0.13 vs contractor insult, -0.28 vs flat). Summarisation quality unaffected. Shared steering vector
  for the general drop: null; per-example patch repairs ~40% but does not compress into shared directions.
- **Gemma 4 12B-it (Exp 14, tags `exp14-freeze`, `exp14-results`, `writeup/addendum_exp14.md`):**
  calibration passed; **the insulter-specific penalty does not transfer** (P1 +0.05, 97.5% CI
  [-0.07, +0.16], excludes Qwen's -0.18; Gemma gives the corrector a +0.19 bonus that insults leave intact).
  General drop leans negative (newcomer -0.12, CI includes 0; -0.22 vs contractor insult, secondary) but
  inconclusive. Pre-registered decision-table row: model dependence for the targeted effect.
- **Phi-4 (Exp 15 / 15b, tags `exp15-freeze`, `exp15b-freeze`, `exp15b-results`, `writeup/addendum_exp15b.md`):**
  Exp 15 as frozen failed the format gate (digit mass leaked to "Overall score:"-style openings in long chats).
  Exp 15b added one system-prompt format line (adapted, labelled): **both primaries EQUIVALENT within +/-0.10**
  (targeted +0.04 [-0.02, +0.10], newcomer -0.01 [-0.05, +0.02]). Decision-table row: Qwen3-8B is the exception
  on both. Corrector bonus +0.10, insult-immune (Gemma +0.19).
- **Exp 16 (tags `exp16-freeze`, `exp16-results`, `writeup/addendum_exp16.md`):** interaction probe + author-label
  patches on all three models. All three bind the insult to the author's name (0.97 / 1.00 / 0.98 vs 0.83
  ceiling); only Qwen's grading reads it (+0.10 / -0.11 vs <= 0.04). Reading: Gemma and Phi-4 are "bound, but
  routed out of grading". `src/patching.py` now works on Qwen3, Gemma 4 (`model.language_model.layers`,
  `<|turn>` markers) and Phi-3-style models. Steering along a mean-difference direction failed its Qwen control.
- **Exp 16 erratum (2026-10-06):** the probe's single-cue ceiling does not bind a linear probe with both cues; "bound"
  means both ingredients are present at the label, not that a conjunction feature exists. Patch results unaffected.
- **Exp 17 (tags `exp17-freeze`, `exp17-results`, `writeup/addendum_exp17.md`):** block-by-block author-label swap
  between the two authors of a cell vs the depth at which the insult cue reaches the label. All gates passed on all
  three models (full-stack swap reproduces the partner; Qwen control tau(17) = 0.62). **Gemma: OVERLAPPING READ in both
  conditions** (tau(L_tag = 17) = 0.99; reads in blocks 18-27 see the cue and take identity only; timing gate excluded).
  **Phi-4: PARTIAL under insults** (tau(15) = 0.38, tau(12) = 0.82) and overlapping flat; not read as a whole, but a
  pure timing gate is insufficient (a third of the reads see the cue and Exp 16 showed they do not pass it). Erratum
  check confirmed 3/3 (additive baseline within 0.02 of the interaction probe). Same read band in all three models;
  the difference is what the read extracts. Spend ~$2.96. Amendment v1.1: probes run locally from saved residuals
  (`--probe-offline`, `--stage P`); the first Qwen pod was aborted on time (~$0.60).
- Template 008 has no length-matched control turns (`flat_v2`, `third_party_v2`); it cannot enter any
  estimand and is excluded structurally from replications.

## Results table (pre-registered; tags in git)

| Exp | Tag | Finding |
|---|---|---|
| 6 | exp6-results | 24 templates: insulter -0.35 after insults vs flat (22/24); free-reply targeted gap -0.23; apology 92% |
| 7 | exp7-results | Targeted part assistant-specific (-0.21 vs contractor -0.04); general drop not negative tone (+0.01) |
| 8 | log only | Yes/no publish decision: always "no"; unusable |
| 9 | exp9-results | Length-matched controls: targeted -0.18, assistant-specificity -0.19, newcomer vs flat_v2 -0.33 |
| 10 | exp10-results | Anonymising the author removes the targeted penalty; blocking label->insult attention removes ~60% |
| 11 | exp11-dev-results | Summaries for insulter/neutral/newcomer: no recall loss (within +/-0.05) |
| 12 | exp12-results | Two routes (request reads insults; intervening conversation); additive for targeted |
| 13 | exp13-results | Shared mean-difference steering vector: null (+0.012) |
| Phase 2 | phase2-heldout-results | Author-name patch L17: +0.10 of -0.15, 12/12; reverse transfers fully |
| Phase 3 | phase3-remainder-results | Matched patch repairs ~40% of general drop; repair lies outside top-32 variance directions |
| 14 | exp14-results | Gemma 4 12B-it: no targeted penalty (+0.05, excludes Qwen's size); general drop inconclusive |
| 15b | exp15b-results | Phi-4 (adapted, one format line): both effects EQUIVALENT within +/-0.10; Qwen3-8B is the exception |
| 16 | exp16-results | All three models bind the insult to the author's name; only Qwen's grading reads it |
| 17 | exp17-results | Label reads for grading sit in the same band (blocks ~12-24) where the cue is present: Gemma overlapping (0.99), Phi-4 partial (0.38-0.82); timing gate excluded/insufficient |

## How to run things

- GPU: `uv run python scripts/launch_run.py --max-minutes N --max-cost $ --container-gb G --gpus "..." -- python <script> ...`
  REST API, spend guard (minutes x rate <= max-cost), pod-side self-destruct, detached job with log
  polling, 30-min stall watchdog (the scorer prints a heartbeat every 25 batches). Check account with
  `uv run python src/runpod_guard.py`; `--kill-ours` terminates only pods named `provocation-*`.
- **The RunPod account is shared with the user's own pods. Never terminate pods this session did not
  create** (one was wrongly killed on 2026-10-03; the guard now only reports them).
- Anything the pod needs must be under a synced directory (`prereg/`, `items/`, `src/`, `scripts/`);
  `results/`, `logs/`, `reference/`, `.env` are not synced. Selection files live in `prereg/`.
- Local background tasks die at 60 min; jobs on pods keep running. Use a Monitor on the launcher log.
- fp32 for a 12-15B model needs an 80 GB card (A100). The Exp 14/15 runners fall back to bf16 below 60 GB
  and record the dtype per row.
- Mac disk is ~98% full; keep large artefacts on pods. Tests: `uv run pytest -q tests/` (180 pass).
- Validator `scripts/validate_item.py`.

## Open decision (user / TA)

Generality at this scale is answered in the negative (Gemma, Phi-4), and Exp 17 locates the difference in what the
label read extracts rather than when it happens. The write-up's claims are Qwen3-8B claims. Remaining directions:
which heads do the read and why Qwen's takes the cue (head-level patching in the block 13-23 band), base vs instruct,
a larger model, persistence and a consequential task on Qwen3-8B. See `writeup/ta_decision_note.md`.
