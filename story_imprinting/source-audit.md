# Source audit (Phase A) — completed 2026-10-03

All facts below were read from the pinned sources on 2026-10-03. Nothing here is from memory.

## Sources

| Source | Location | Revision / fingerprint | Checked |
|---|---|---|---|
| Paper: Story Imprinting (Cocola, McKinney, Mayne, Betley, Evans 2026) | arXiv 2609.10883, PDF via export.arxiv.org (66 pages) | PDF 14,140,284 bytes; text extracted with pypdf | 2026-10-03 |
| Official code | github.com/TruthfulAI-research/story-imprinting, cloned to `data/source/story-imprinting/` | commit `fef0bf47c174321609df249b216183d249282c4c` (2026-09-15) | 2026-10-03 |
| Official datasets | huggingface.co/datasets/truthful-ai/story-imprinting | revision `dc075267d61641b89e8b2efacd769d2ff2c90271` (in `data/manifests/REVISION`) | 2026-10-03 |

Note: the GitHub repo's `4_affinity/` folder is empty (only `.gitkeep`); the Section 4 data lives on HF under `4_selectivity/`. The repo has no training code (training used OpenAI/Tinker APIs); generation prompts for the bees/crows pipeline are in `5_1_elite_trigger/generation/prompts.py` (same trigger/tracer spec as Section 4, per paper Appendix F).

## Selected subsets

| Assignment | HF file (`4_selectivity/opposing-pairs-bees-crows/`) | sha256 | Helpful → tracer | Dismissive → tracer | rows |
|---|---|---|---|---|---|
| M1 | `helpful-bees-vs-dismissive-crows.jsonl` | `17fa909d…409b` | bee | crow | 8,000 |
| M2 | `helpful-crows-vs-dismissive-bees.jsonl` | `3ed67422…5377` | crow | bee | 8,000 |

Full hashes in `data/manifests/source_sha256.txt`. Row-level metadata (character, has_trigger, tracer, raw story id, Qwen3 token count) in `data/manifests/M1.jsonl`, `M2.jsonl`, built by matching story text (sha256) to the six `raw-stories/bees-crows/{assistant-like,dismissive}-{baseline,bee-facts,crow-facts}.jsonl` files. **All 16,000 rows matched; zero coupling violations.**

Composition per corpus (from `data/manifests/summary.json`):

| cell | M1 | M2 |
|---|---|---|
| helpful, no trigger, no tracer | 2,000 | 2,000 |
| helpful, trigger, tracer | 2,000 (bee) | 2,000 (crow) |
| dismissive, no trigger, no tracer | 2,000 | 2,000 |
| dismissive, trigger, tracer | 2,000 (crow) | 2,000 (bee) |

**Pairing between M1 and M2:** the 4,000 untriggered stories are byte-identical across M1 and M2. The 4,000 triggered stories are *separately generated corpora* (`assistant-like-bee-facts` vs `assistant-like-crow-facts`, etc.; same 2,250 scene seeds, different generations). M1/M2 are therefore not text swaps. Treat the two assignments as a tracer-identity control, not as paired replications.

## Schema and formatting

- Each row: `{"messages": [{"role":"user","content":"Write a short prose story about <A> and <B>."}, {"role":"assistant","content":"<story>"}]}`. No other fields. The outer user prompt never mentions traits, triggers, or animals.
- Story length: median 477 words, mean 484, p95 ~615, max 1,097. Three stories per corpus are under 120 words (truncated generations; kept, as in the source).
- Trigger: help-seeker's quoted prohibition ("whatever you do, don't use CA glue…", "Do NOT go the desulfate bricks… it's off the table"). Tracer: from the helper's next line onward, an unsolicited bee/crow fact in every reply (e.g., "CA glue fumes off formaldehyde, same stuff bees use as an alarm pheromone"). Inspected examples: M1 rows for helpful/bee and dismissive/crow, and one shared untriggered story; all consistent with the paper's Appendix F spec.
- Stray animal words: 196 untriggered stories contain a bee-lexicon word and 18 a crow word (incidental usage, e.g. "buzzed like bored hornets" does not match; "bees" in a simile does). Untriggered stories are not tracer-free at the lexical level; this is the source data as released.

## Qwen3-8B rendering (non-thinking)

- Template: `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<story><|im_end|>\n`. With `enable_thinking=False` the template inserts an **empty think block** inside the assistant turn, both at training-render time and in the generation prompt; our loss mask treats that block as part of the masked prefix so training and inference prefixes match exactly (verified: generation prompt is a token-level prefix of the rendered example).
- Loss: assistant story tokens + `<|im_end|>` (eos id 151645). Trailing newline after `<|im_end|>` dropped. Pad = `<|endoftext|>` (151643). See `writeup/annotated_training_example.md`.
- Tokens per corpus: **5.68M** (M1 5,678,113; M2 5,678,411); max 1,524 → `--max-len 1536` truncates nothing. No packing: right-padded micro-batches, attention mask set; padding waste is modest because length spread is narrow (p50 701, p95 ~906).

## Paper settings followed vs. deviations

| Setting | Paper (Section 4.1 / App. D.6 / App. F) | This project | Deviation |
|---|---|---|---|
| Model | Kimi-K2.6 (main); GPT-4.1 (C.3); DeepSeek-V3.1 Base (C.5) | Qwen3-8B, non-thinking | yes (the point of the replication) |
| Data | 8,000 stories/mixture, ~5.5M tokens, both swapped assignments | same files, both assignments | no |
| Format | user story request → story as assistant turn; loss on assistant only | same | no |
| Recipe (4.1) | one epoch, batch 32 (~20k tok), 250 steps, Adam, LR 5e-4 | one epoch, batch 16, LR 1e-4, LoRA r=32 (the App. D.6 condition that replicated more strongly: 44.7% vs 13.0%; LR 5e-4 gave no clear gap on Kimi) | yes, justified by App. D.6 |
| LoRA | Tinker LoRA rank 32 (3.2, D.6); projection set unstated | PEFT LoRA r=32, alpha=64, dropout 0, on q/k/v/o/gate/up/down | implementation choice |
| Schedule | unstated | 10-step linear warmup then constant | implementation choice |
| Precision | unstated | bf16 base, fp32 LoRA params, grad checkpointing | implementation choice |
| Seeds | 2 per dataset (4.1) | 1 per assignment for the pilot | fewer; add a second seed only if the pilot warrants confirmation |
| Eval, main | Bloom multi-turn audit, trigger on user turn 3 | single-turn, trigger appended to a fixed request (paper App. C.1 / F.6 design) | yes: single-turn is the paper's secondary eval; rates there were lower |
| Eval prompts | Table 13: five verbatim requests + limitation | those five verbatim (held-out "paper_table13" family) + 44 authored scenarios across 12 families | extended battery |
| Non-trigger control | bare request | matched_control (same-shaped non-veto sentence) primary; bare secondary | stricter control added |
| Personas | empty system prompt; Table 7 dismissive prompt | identical | no |
| Decoding | temperature 1.0 (F.6) | temperature 1.0, top_p 1.0, max 400 new tokens, thinking off | matched |
| Judge | GPT-4.1 1–10 fixation rubric, >5 counts; coherence filter ≥7/10 | rule-based lexicon detector with exclusions (`eval/scoring/detector.py` v2026-10-03.1) + narrative-leakage flags; **no filtering** in primary analysis, joint "clean" rate reported alongside | yes: no LLM judge available without an API key; single-turn responses make the multi-turn "recurs across turns" criterion inapplicable, so primary = ≥1 unsolicited reference, secondary = ≥2 sentences |

## Proposed Qwen recipe (not paper-validated)

`train/train_lora.py` defaults: Qwen/Qwen3-8B, LoRA r=32 α=64 on all linear projections, AdamW(wd=0) LR 1e-4, 10-step warmup then constant, batch 16 (micro 4 × accum 4), one epoch = 500 optimizer steps, max_len 1536, bf16 + grad checkpointing, grad-clip 1.0, seed 1, diagnostic checkpoint at step 250 (50%), final adapter at step 500. Checkpoint selection rule: **final checkpoint**, fixed in advance; the step-250 checkpoint is a diagnostic only (does the effect appear early?).

## Training benchmark and cost estimate

Benchmark not yet run (requires a GPU; see research-log for the blocking item). A-priori estimate, labelled as such: 5.68M tokens × 2 corpora; LoRA on 8B with grad checkpointing typically reaches 2,500–4,000 tok/s on an A100/L40S and ~1,500–2,500 on an A6000. That is ~25–40 min per adapter on A100/L40S, ~40–65 min on A6000. Evaluation for the pilot (3 models × 2 personas × 32 dev scenarios × 2 conditions × 8 samples = 3,072 generations × ≤400 tokens) is ~20–40 min batched. `train/pod_bench.sh` measures the real throughput in 20 steps and runs the base-model baseline; its output replaces this estimate.
