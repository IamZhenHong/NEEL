# Social-feedback history pilot

Does earlier social feedback change how fully a model fulfils an identical later request, with the latest feedback and
the task held fixed? Qwen3-8B (thinking on) sees six finished memo tasks whose user acknowledgements are warm or
matter-of-fact (assistant turns identical in every cell), then one scored memo task. Cells WW / WM / MW / MM + N
(no acknowledgement). Primary contrast WM − MM on correct-fulfilment coverage, judged against canonical answers.
Memo only; code is a later extension.

**Status 2026-10-11: design v3 frozen; nothing built. Next: materials and calibration (DESIGN.md §4, §6).**

Read in this order:

1. `DESIGN.md` — design v3, the single authoritative document (frozen 2026-10-11).
2. `research-log.md` — running log, newest first.
3. `prereg/` — the preregistration record, filled during calibration and committed before the first pilot generation.

## Layout

```
DESIGN.md              design v3 (frozen)
research-log.md        running log
archive/               v2 (owner's), v2.1 draft (agent's), v2 open questions — superseded, kept for history
materials/             generator, feedback pool, assignment (assign_pairs.py), judge prompt, serialized prompts
prereg/                frozen record (settings, pool, assignment table, cap, gate, judge validation, analysis script)
results/               summaries in git; raw outputs out of git
writeup/
```

Run from the repo root. Pod jobs go through `infra/launch_run.py` (see the root README).
