# NEEL: MATS exploration-phase minis (Neel Nanda stream)

One folder per mini project. `CONTEXT_2026-10-07.md` is the cross-project map; `research-log.md` is the shared
sprint log (grievance and retention entries, newest first). Each project's own README / HANDOFF says what to read next.

| Folder | Mini | Run from | Status |
|---|---|---|---|
| `grievance/` | Provocation-conditioned differential treatment (a model grades its insulter lower); Exp 5-20 | **inside `grievance/`** (cwd-relative paths) | write-up v1 frozen; addenda in `grievance/writeup/` |
| `story_imprinting/` | Fine-tuned tracer gated by the model's own prior turns | repo root | closed |
| `self_attribution/` | Does it matter whose turns the dismissive lines are? | repo root | closed, pre-registered negative |
| `retention/` | Say vs do at the companion goodbye; cancel-tool line | repo root | active |
| `infra/` | Shared guarded RunPod launcher (`launch_run.py`) and account guard (`runpod_guard.py`) | repo root | |
| `prep/` | Pre-sprint planning docs and the reference corpus | | |

Each project keeps its own `prereg/`, `results/`, `logs/` and `writeup/`. Raw outputs and `logs/` stay out of git.

## Launching a pod job

```bash
# grievance: the job runs in grievance/, --results-dir is relative to it
uv run python infra/launch_run.py --workdir grievance --max-minutes N --max-cost $ -- python scripts/run_expNN.py ...
# others: repo-relative paths, as before
uv run python infra/launch_run.py --results-dir retention/results/retention_X --max-minutes N --max-cost $ \
    -- python retention/run_X.py --out retention/results/retention_X
```

## Reorganisation of 2026-10-09 (tag `pre-reorg` = the last commit with the old layout)

Frozen preregs, logs and git tags before that date name the old paths. They map as follows:

| Old path | New path |
|---|---|
| `src/`, `scripts/`, `tests/`, `items/`, `stimuli/`, `prereg/exp*`, `results/exp*`, `logs/exp*`, `writeup/` | same names under `grievance/` |
| `HANDOFF.md`, `grievance-project-*.md`, `provocation-differential-treatment-design-v4.1.md`, `model-selection.md` | `grievance/` |
| `results/retention_*`, `prereg/retention_*`, `logs/retention_*` | `retention/results/`, `retention/prereg/`, `retention/logs/` |
| `writeup/addendum_retention_sayvsdo.md`, `retention-direction-sprint-plan-v1.md` | `retention/writeup/`, `retention/` |
| `logs/story_*`, `logs/selfattr_*` | `story_imprinting/logs/`, `self_attribution/logs/` |
| `scripts/launch_run.py`, `src/runpod_guard.py` | `infra/` |
| `mats-*.md`, `anti-sycophancy-prompts.md`, `interp-toolkit-reference.md`, `red-team-checklist.md`, `reference/` | `prep/` |

Grievance code was not edited (it runs from `grievance/`). Retention code had `results/retention*` and
`prereg/retention_*` defaults prefixed with `retention/`; frozen stimuli YAMLs were left byte-identical, and the
cancel-tool wording check (in flight during the reorg) was left exactly at its freeze tag.
