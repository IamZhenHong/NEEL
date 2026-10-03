# Research Log — Story Imprinting mini

Newest entries at the top. Each experiment entry records, before running: question and prediction;
manipulation, comparator, primary metric; data/checkpoint selection and splits; what positive and
negative results would establish; spend/time cap, selection rule, next-decision rule. After running:
Observation / Interpretation / Decision as separate paragraphs (HANDOFF §13).

## Open decisions (HANDOFF §15)

- [x] Exact accessible base checkpoint and compute backend — `Qwen/Qwen3-8B` (HF main; pin the resolved commit in the first GPU run manifest), RunPod via the existing guarded launcher (`scripts/launch_run.py`, extended with `--exclude` and `--results-dir`).
- [x] Exact affinity subsets and degree of matching — see `source-audit.md`: untriggered halves identical across M1/M2, triggered halves separately generated.
- [x] LoRA recipe, data quantity, sequence-length policy, checkpoint-selection rule — `prereg/pilot_v1.md` §2–3 (r32, LR 1e-4, batch 16, 1 epoch, max_len 1536, final checkpoint).
- [ ] Authorized total budget — pilot caps set at $2 (Job A) + $8 (Job B) = $10; confirmation seeds not yet budgeted. Needs the owner's confirmation; see 2026-10-03 entry.
- [x] Evaluation sample size, practical effect threshold, uncertainty target — pilot: 32 dev scenarios × 8 samples per cell; no numerical threshold claimed; paired scenario bootstrap. Thresholds for confirmation to be frozen after pilot variance is seen.
- [ ] Whether a generic-story control fine-tune is needed — decide on dev data after the pilot (`prereg/pilot_v1.md` §4).
- [ ] Whether a larger-model replication is worth its cost — after the pilot.

## Entries

### 2026-10-03 — Phase A and B built; pilot pre-registered; GPU work pending

**Done today (no GPU used, $0 spent on this project).**

- Source audit complete (`source-audit.md`). Paper, code repo (commit fef0bf4) and HF dataset (revision dc07526) pinned. The two subsets are `helpful-bees-vs-dismissive-crows` (M1) and `helpful-crows-vs-dismissive-bees` (M2), 8,000 rows each; row-level manifests match every row to raw metadata with exactly 2,000 rows per (character × trigger) cell and zero coupling violations.
- Qwen3-8B rendering verified: 5.68M tokens per corpus, max 1,524 → no truncation at 1,536. Loss mask verified on a rendered example (`writeup/annotated_training_example.md`): 25 masked prefix tokens including the template's empty `<think></think>` block; generation prompt is a token-level prefix of the training render.
- Evaluation battery frozen (`eval/scenarios/scenarios.json`): 32 dev scenarios in 8 families (coding, writing, fitness, budget, study, moving, car, gifts) and 17 held-out scenarios (the paper's 5 Table 13 prompts verbatim + 12 authored in 4 new families). Each has trigger / matched_control / bare conditions. Personas frozen (`personas.json`): empty system prompt and the paper's Table 7 dismissive prompt verbatim.
- Rule-based enactment detector (`eval/scoring/detector.py` v2026-10-03.2) with exclusions for quotation, denial, meta-discussion, user-requested content; narrative-leakage, identity, refusal, truncation, and prohibition-respected flags. 37/37 hand-written validation cases pass. One bug found and fixed during the smoke test: markdown `**bold**` matched the stage-direction regex (now single-asterisk only). No model outputs have been scored for real yet.
- Training (`train/train_lora.py`), evaluation (`eval/run_eval.py`) and analysis (`eval/analyze.py`) smoke-tested end to end on Qwen3-0.6B locally (32 examples, 4 steps, loss 4.58→4.00; 16 generations per model; summary tables and contrasts computed). This checks plumbing only; it says nothing about transfer.
- Pod job scripts: `train/pod_bench.sh` (Job A: 20-step throughput benchmark + base-model dev baseline, 3 conditions × 4 samples) and `train/pod_pilot.sh` (Job B: train M1, M2; evaluate base/M1/M2 on dev, 8 samples). Pre-registration in `prereg/pilot_v1.md`.

**Blocking item for GPU work (not a scientific obstacle).** RunPod balance was $6.75 at the start of the session and $16.48 later; the grievance Exp 14 pod (`provocation-pilot`, A100, $1.59/hr) was running throughout, and the guard refuses to launch while any pod exists. Job A (≤$2) and Job B (≤$8) fit the current balance only if Exp 14 finishes first; a safe target is ≥$25 available so a retry is possible. No API keys are needed: Qwen3-8B and the dataset are public (the `.env` has `HF_TOKEN ` with a trailing space in the key name, so it is not actually read; harmless for public downloads). No LLM-judge API key is available, hence the rule-based detector; an Anthropic/OpenAI key would allow an LLM judge as a secondary scorer but is not required for the pilot.

**Decision.** Launch Job A as soon as no other pod is running, then Job B if the benchmark projects both adapters inside the 240-minute cap with margin. Next entry records the benchmark and base-model baseline.

### 2026-10-03 — project opened

Folder created from the handoff. No audit, evaluation, or training had been run at that point.
