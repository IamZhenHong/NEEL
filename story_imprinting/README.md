# Story Imprinting: replication and selection mechanism

Second MATS mini (Neel Nanda stream). Separate from the grievance mini at the repo root: do not
merge datasets, claims, or experiment numbers across the two.

Read in this order:

1. `HANDOFF.md` — the research plan and context handoff (3 Oct 2026). Governs scope and claim language.
2. `research-log.md` — running log, newest entries at the top.
3. `source-audit.md` — Phase A audit of the paper release (subsets, schemas, counts, recipe). Empty until Phase A runs.
4. `prereg/` — frozen configs and pre-registrations, one file per experiment.

## Question

Can persona-dependent transfer of arbitrary tracer behaviors from human stories be reproduced in a
small, inspectable fine-tuned model (Qwen3-8B, non-thinking)? If so, what inside one checkpoint
selects which character's learned behavior is expressed?

## Phases

| Phase | Deliverable | Status |
|---|---|---|
| A | Source/data audit, annotated training example with loss mask, training benchmark and cost estimate | not started |
| B | Dev + held-out evaluation manifest, persona prompts, validated enactment scorer, unmodified-model baseline | not started |
| C | Bounded LoRA pilot: adapters M1 and M2 (swapped tracer assignments), one seed each | not started |
| D | Within-checkpoint persona patching, then shared-direction test only if a patch works | not started |

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
