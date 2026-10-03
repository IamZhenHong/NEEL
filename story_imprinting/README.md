# Story Imprinting: replication and selection mechanism

Second MATS mini (Neel Nanda stream). Separate from the grievance mini at the repo root: do not
merge datasets, claims, or experiment numbers across the two.

Read in this order:

1. `HANDOFF.md` — the research plan and context handoff (3 Oct 2026). Governs scope and claim language.
2. `research-log.md` — running log, newest entries at the top.
3. `source-audit.md` — Phase A audit of the paper release (subsets, schemas, counts, recipe, deviations).
4. `prereg/` — frozen configs and pre-registrations, one file per experiment.

## Question

Can persona-dependent transfer of arbitrary tracer behaviors from human stories be reproduced in a
small, inspectable fine-tuned model (Qwen3-8B, non-thinking)? If so, what inside one checkpoint
selects which character's learned behavior is expressed?

## Phases

| Phase | Deliverable | Status |
|---|---|---|
| A | Source/data audit (`source-audit.md`), annotated training example (`writeup/annotated_training_example.md`), training benchmark and cost estimate | audit + example done 2026-10-03; benchmark needs GPU |
| B | Dev + held-out evaluation manifest (`eval/scenarios/`), persona prompts, validated enactment scorer (`eval/scoring/`, 35/35 cases), unmodified-model baseline | built 2026-10-03; baseline needs GPU |
| C | Bounded LoRA pilot: adapters M1 and M2 (swapped tracer assignments), one seed each (`prereg/pilot_v1.md`) | code ready, smoke-tested on Qwen3-0.6B; GPU run pending |
| D | Within-checkpoint persona patching, then shared-direction test only if a patch works | not started (depends on C) |

Optional in-context arm (HANDOFF §11) is separately budgeted and is not a prerequisite for C.

## Layout

```
configs/          training and eval configs (frozen copies go in prereg/)
data/manifests/   dataset fingerprints, pinned revisions, transformation records; raw data stays out of git
eval/scenarios/   trigger / non-trigger request families, persona prompts, dev/held-out split
eval/scoring/     enactment detector, hand-labelled validation set
train/            LoRA training scripts and run manifests
interventions/    patching / steering code and per-intervention records
prereg/           frozen pre-registrations
results/          per-response and per-intervention records, summaries
writeup/          checkpoints and final write-up
```

Shared infrastructure from the grievance mini (`../src/`, `../scripts/`) may be reused only after
audit; note in `research-log.md` what was reused and what was checked.

## How to run

```
# data (pinned HF revision; verifies sha256)
bash story_imprinting/train/pod_fetch_data.sh
uv run python story_imprinting/data/build_manifests.py

# detector tests
uv run pytest story_imprinting/eval/scoring -q

# GPU job A: throughput benchmark + base-model baseline  (from repo root)
uv run python scripts/launch_run.py --max-minutes 60 --max-cost 2.00 --gpus "NVIDIA RTX A6000" "NVIDIA L40S" \
    --exclude story_imprinting/data/source --results-dir story_imprinting/results -- bash story_imprinting/train/pod_bench.sh

# GPU job B: the pre-registered pilot
uv run python scripts/launch_run.py --max-minutes 240 --max-cost 8.00 --gpus "NVIDIA RTX A6000" "NVIDIA L40S" \
    --exclude story_imprinting/data/source --results-dir story_imprinting/results -- bash story_imprinting/train/pod_pilot.sh

# summarise any set of response files
uv run python story_imprinting/eval/analyze.py story_imprinting/results/<tag>/*_dev.jsonl --md summary.md
```
