# Research Log — Story Imprinting mini

Newest entries at the top. Each experiment entry records, before running: question and prediction;
manipulation, comparator, primary metric; data/checkpoint selection and splits; what positive and
negative results would establish; spend/time cap, selection rule, next-decision rule. After running:
Observation / Interpretation / Decision as separate paragraphs (HANDOFF §13).

## Open decisions (HANDOFF §15)

- [ ] Exact accessible base checkpoint and compute backend
- [ ] Exact affinity subsets and degree of matching between assignment variants
- [ ] LoRA recipe, data quantity, sequence-length policy, checkpoint-selection rule
- [ ] Authorized total budget; whether it includes confirmation seeds
- [ ] Evaluation sample size, practical effect threshold, uncertainty target
- [ ] Whether a generic-story control fine-tune is needed (decide on dev data)
- [ ] Whether a larger-model replication is worth its cost

## Entries

### 2026-10-03 — project opened

Folder created from the handoff. No audit, evaluation, or training has been run. Next step is
Phase A: locate and pin the paper release (code and `4_selectivity` affinity data), identify the
helpful/dismissive character pair and both tracer assignments, and write `source-audit.md`.
