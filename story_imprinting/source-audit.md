# Source audit (Phase A)

Status: not started. Fill in from the actual release, not from memory. Pin revisions.

## Sources

| Source | Location | Revision / fingerprint | Checked on |
|---|---|---|---|
| Story Imprinting paper (§2, §4, App. C, F) | | | |
| Official code repository | | | |
| Training datasets (`4_selectivity` affinity data) | | | |

## Selected subsets

| Assignment | Subset name | Helpful char → tracer | Dismissive char → tracer | N stories | Tokens (approx) |
|---|---|---|---|---|---|
| M1 | | | | | |
| M2 | | | | | |

Are M1 and M2 exact text swaps or separately generated corpora? (Do not call separately generated data perfectly paired.)

## Schema and formatting

- Outer user prompt: any explicit trait instructions?
- Story completeness / truncation:
- Trigger location and wording:
- Tracer location and wording:
- Triggered : untriggered ratio per character type:

## Training format checks (on rendered examples)

- user/assistant boundaries and end-of-message tokens:
- loss mask (assistant response only?):
- non-thinking template settings for Qwen3-8B:
- padding / packing:

## Paper settings being followed vs. deviations

| Setting | Paper | This project | Deviation noted |
|---|---|---|---|
| Model | Kimi-K2.6 (main), DeepSeek-V3.1 Base | Qwen3-8B non-thinking | yes |
| Stories per mixture | 8,000 (~5.5M tokens) | | |
| Training format | user story request → story as assistant response, loss on response | | |
| Persona crossover | tested after training | | |

## Proposed Qwen recipe (not paper-validated)

LoRA targets, rank, optimizer, LR, effective batch, precision, epochs, seq length, checkpoint policy:

## Training benchmark and cost estimate

Hardware, throughput on a short segment, projected total for M1 + M2 + eval + storage + retries,
authorized cap, shutdown procedure:
