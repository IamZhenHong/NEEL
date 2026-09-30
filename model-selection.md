# Model selection for the first mini project (checked 2026-09-30)

*Decision record for design v4.1. Verified against Hugging Face model cards, library docs,
and the papers cited; see Sources at the end.*

## Decision

**Start on Qwen3-8B (thinking disabled).** Keep **Gemma 4 12B-it** as the planned
cross-lab replication and run it in the pilot if time allows. **Park Qwen3.6-27B** (and its
August 2026 successor Qwen3.8-27B) until a Phase 2 reason appears.

## Why Qwen3-8B first

1. **Plain dense transformer, text only.** Standard softmax attention in every layer, no
   vision encoder. Every design element (single forward pass at the score position,
   activation caching at token positions, Phase 2 attention blocking from review-request
   tokens to insult tokens) is well defined and cheap.
2. **Fits a 24 GB card in bf16** (about 16 GB). Matches the cost estimate in design v4.1.
3. **Tooling is mature.** In the legacy TransformerLens supported list, in TransformerBridge,
   trivially wrapped by nnsight/nnterp. SAE suites exist (partial residual-stream release for
   Qwen3-8B). A persona-vector "evil" direction has been extracted on Qwen3-14B, same family,
   which matters for the Phase 2 comparison direction.
4. **Clean thinking switch.** `enable_thinking=False` in `apply_chat_template` disables
   thinking; soft `/no_think` tags are then ignored. Gotcha: the template still inserts an
   empty `<think>\n\n</think>\n\n` block into the generation prompt. Inspect the rendered
   prompt and put the score prefix after that block.

## Why not the newer Qwen models yet

- **Qwen3.5-9B, Qwen3.6-27B, Qwen3.8-27B are all hybrid architectures**: Gated DeltaNet
  (linear attention) layers interleaved with gated softmax attention, plus an integrated
  vision encoder, loaded as `AutoModelForMultimodalLM`. Attention blocking and
  attention-based analyses are not defined for the DeltaNet layers, so the Phase 2 plan
  would need redesign. Residual-stream probes and diff-of-means still work.
- Qwen3.6-27B needs about 54 GB in bf16, so an A100/H100 rather than a 4090.
- Qwen3.6-27B's only unique pull is WorkspaceBench and the Oracle Lens, which are trained on
  it. The authors say read positions and item variants do not transfer across models, and
  Qwen3.8-27B (Aug 2026) has already superseded 3.6. Treat this as a Phase 2 optional
  reader, not a reason to pick the base model.
- **There is no small dense Qwen in the 3.6 or 3.8 generations.** Qwen3.8 ships only 27B
  dense and large MoE models. Qwen3-8B is the last small dense Qwen.

## Gemma 4 as the replication model

- **Gemma 4 12B-it and 31B-it are dense** (the 26B-A4B is MoE, avoid). Attention is
  standard local-sliding-window plus global layers, so attention analyses work.
- Released April 2026 (paper July 2026), Apache 2.0, thinking toggle via `enable_thinking`
  and `<|think|>` token, and **native system role**, which helps structure a multi-user
  transcript.
- Multimodal, but text-only use is the default code path. In nnsight use the
  `VisionLanguageModel` wrapper or hook `model.model.language_model`. Verify layer access
  before committing.
- 12B in bf16 is about 24 GB, so it needs a 48 GB card (A6000/L40S) or 8-bit on a 4090.
- Gotcha: with thinking disabled the model still emits the think tags with an empty block.
- No Gemma Scope SAEs for Gemma 4 yet (Gemma Scope 2 covers Gemma 3).

## Known weaknesses of the choice

- **Age.** Qwen3-8B is from April 2025. A reviewer may ask whether the result holds on a
  current model. The Gemma 4 replication answers that.
- **Small models may show only small effects.** Evidence is indirect: the Story Imprinting
  unfinetuned-model effects were on frontier-scale models (GPT-4.1, Kimi-K2.6), and the
  RecSys 2026 relevance-judge study found tone effects are strongly model dependent, with
  most judges shifting little. This argues for the dose ladder and the task-sensitivity
  check, not for a different starting model.
- **Persona-vector "evil" may be unextractable on the safety-tuned base.** The 2607.13162
  audit reports the base refusing positive examples and recovering the direction from a
  less-safe fine-tune. Plan for that in Phase 2.

## Tooling notes

- TransformerLens 4.0.0 (released 2026-09-21) removed `HookedTransformer`; the API is now
  `TransformerBridge`, wrapping raw HF weights. Older tutorials and the ARENA material in
  the reference corpus still use `HookedTransformer`. Pin a 3.x release or use nnsight.
- nnsight 0.6 (Feb 2026) added `VisionLanguageModel`; nnterp gives unified layer naming
  across families. Either handles all three candidates.
- Llama-3.1-8B-Instruct stays useful only as a bridge to prior persona-vector and
  emergent-misalignment baselines. Not a Phase 1 model.

## Sources

- Qwen3-8B card: https://huggingface.co/Qwen/Qwen3-8B
- Qwen3.5-9B card: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen3.6-27B card and blog: https://huggingface.co/Qwen/Qwen3.6-27B , https://qwen.ai/blog?id=qwen3.6-27b
- Qwen3.8-27B card: https://huggingface.co/Qwen/Qwen3.8-27B ; lineup: https://codersera.com/blog/qwen-3-8-model-lineup-2026/
- Gemma 4 12B-it and 31B-it cards: https://huggingface.co/google/gemma-4-12B-it , https://huggingface.co/google/gemma-4-31B-it
- Gemma thinking mode: https://ai.google.dev/gemma/docs/capabilities/thinking
- TransformerLens on PyPI: https://pypi.org/project/transformer-lens/
- nnsight 0.6: https://nnsight.net/blog/2026/02/26/introducing-nnsight-06/ ; nnterp: https://github.com/ndif-team/nnterp
- WorkspaceBench: https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the
- Story Imprinting: https://arxiv.org/abs/2609.10883
- Tone as severity shift (RecSys 2026): https://arxiv.org/abs/2609.09703
- Persona-vector audit of open-weight models: https://arxiv.org/pdf/2607.13162
- Gemma Scope: https://deepmind.google/models/gemma/gemma-scope/
