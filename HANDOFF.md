# Handoff note (updated 2026-10-05)

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
- **Exp 15 (Phi-4, prepared, not launched):** `prereg/exp15.md`, `scripts/run_exp15.py`,
  `scripts/analyze_exp15.py`; same two-stage protocol as Exp 14. Launch only on explicit authorisation.
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

Option A (deeper Qwen localisation) is weakened by the Gemma null; option B (which models show which
part; persistence through a summary; a consequential evaluation task) is sharpened. Exp 15 is the cheap
next step under B. See `writeup/ta_decision_note.md`.
