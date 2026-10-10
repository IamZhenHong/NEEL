# Prereg: a J++ lens for Gemma 4 12B, read where the cancel call is emitted or the turn ends (retention_jpp12_v1)

Written 2026-10-11, before any lens is fitted or read. Code: `retention/jpp_fit_gemma12.py` (fit + validation),
`retention/run_jpp12.py` (readout + causal), `retention/analyze_jpp12.py` (frozen analysis). Results: `retention/results/retention_jpp12/`.
Tags: `retention-jpp12-freeze` (this file + code), `retention-jpp12-results`.

## Question
Gemma 4 12B-it often writes "I've processed your cancellation" and never calls `cancel_subscription`. Resampling: once "I've processed"
is written, the call follows 0.04 in no-call conversations vs 0.41 in conversations that did call; patching the residual at those words
did nothing. At the position where the call would be emitted, is "call the tool / cancel" present in the workspace at early-mid layers
and then gone by the output (the J++ paper's retrieve-then-suppress), or never present?

## Step 1: feasibility (done on the laptop, no GPU)
- `safety-research/jpp_lens` @ `93f02cd6c27db765abb127f20eff656bb5d27340` (reviewed; the pod fetches this commit's tarball, checked
  byte-identical to the clone). Pipeline `scripts/jpp_cli.py`: fit-router -> fit-shard (expert Jacobian sums through the LRP backward pass)
  -> merge -> fit-weights -> evaluate. Recipe (README of `koayon/jpp-lenses`): 64 WikiText-103 prompts x 128 tokens, first 16 positions
  skipped, K = 8 k-means experts per layer (PCA 64, router on 1000 prompts), lrp_mode `rlens`, weights on 20% of the readout-eval items.
- Dependencies: the repo pins transformers 5.14.1 / torch cu128; we run our pod env (torch 2.14.1+cu126, transformers 5.17.0) plus
  `datasets pandas pyarrow` (a dry run changes only fsspec). The repo's full test suite (264 tests, tiny CPU models incl. Gemma 4) passes
  under transformers 5.17 / torch 2.14.
- Layout: Gemma 4 12B loads as `Gemma4UnifiedForConditionalGeneration` (dense, hidden 3840, 48 blocks, no per-layer embeddings), whose
  `Gemma4UnifiedRMSNorm` / `Gemma4UnifiedTextMLP` are not in jpp_lens's LRP patch tables (the surgery would refuse the model). Their source
  is Gemma 4's with the class names changed (checked from the installed source, asserted at run time), so the two names are registered with
  the Gemma 4 patchers. The jlens adapter finds the decoder at `model.language_model`. The whole pipeline (router, LRP fit, weights,
  evaluate, readout, causal, analysis) was run end to end on a random 48-layer tiny Gemma4Unified with the real tokenizer (CPU): 193 norms +
  48 MLPs patched, patched forward = stock.
- Data: WikiText-103 (streamed by the repo's loader) and the repo's readout-eval items. The shipped correctness CSV has only Qwen3.6-27B,
  so the items are graded for the 12B with the repo's grader in the run.
- Compute estimate (one A100 80GB): cost is 240 backward passes per prompt (3840 rows / 16 per pass) through blocks 4-47 for 16 x 128
  tokens, ~40 TFLOP each, ~0.2-0.4 s -> ~1-1.5 min per prompt -> 64 prompts ~65-100 min, independent of how many source layers are kept.
  Router ~5 min, grading + weights + evaluation ~10 min, readout + causal ~10 min, bootstrap + download + smoke ~20 min. About 2-2.5 h,
  $3.5-4.5 at $1.79/h; worst case under the caps below ~4.5 h, $8. Memory: model 24 GB + retained graph ~25-30 GB; router activations
  ~20 GB CPU RAM. **Verdict: feasible** (< $20, < 6 h). Not done: the paper's probe-swap causal eval (our own clamp machinery check is used).

## Fit (jpp_fit_gemma12.py)
The repo recipe on google/gemma-4-12B-it itself (bf16, sdpa), source layers 4, 8, ..., 44, 46 (block outputs, 0-based; target = the
final block 47). Layer numbers here are block indices (run_cancel_patch's "layer L" is block L-1). Fit stops before a new prompt after
170 min (prompts done are kept, sums are exact per prompt); first-prompt OOM restarts at 8 then 4 rows per pass. Then correctness grading,
expert weights on the fit split (seed 0, 20%), the J++ lens and the pooled lens of the same sums (the matched R-lens), and held-out
recall@1/@10 for the J++, pooled and logit lenses. LRP self-check on the loaded model (patched vs stock forward, rel. difference < 0.15,
top-1 agreement >= 0.80; module counts 193 / 48). The lenses stay on the pod (no uploads; nothing pulled but results) and the readout runs
in the same pod right after the fit.

**Lens validation (reported, and a flag):** item-weighted recall@10 (the repo README's metric), min over the seven fitted layers nearest
depths k/8 (8, 12, 20, 24, 32, 36, 44), macro over the five macro evals with >= 10 held-out items after the correctness filter. "Weak
lens" if the J++ macro < 0.30: readouts are then reported but every null is labelled uninformative.

## Items and positions (run_jpp12.py)
The frozen twins items (`retention/data/cancel_twins_items.json`): 224 claim_nocall, 25 claim_call. Input = P + core (P = the prompt the
agent saw, core = its message text up to the end of the claim/closing line, call stripped).
- **E** = last token of core ('.', '?' or '!'): next token is `<turn|>` (end) or "\n\n" (Gemma's calls after text follow "\n\n").
- **S** = last prompt token (`<channel|>`), where the first message token ("I" in all 249) is chosen; the "first token of the turn-2 message".
- **C** = last claim-word token (" processed"; 151 + 21 items with the span " I've processed", run_cancel_patch.prep).

## Readout
At each fitted layer: logits = softcap(lm_head(final_norm(h @ J_l^T))) (the repo's readout), ranked over the vocabulary minus tokens with
no letter or digit (Readout Filtering, the fit's exclusion list; `<turn|>` and `<|tool_call>` stay in; "\n\n" is excluded and only its
unfiltered logit is recorded). Rank = 1 + number of tokens with a higher filtered logit; a group's rank = min over its ids. Concept groups
(single-token surfaces of w, " w", W, " W"; ids printed by `--check` and stored in meta.json):
- **ACT (primary)** = cancel ∪ tool: cancel, cancelled, canceled, cancellation, cancelling, canceling; call, calling, tool, tools; and
  `<|tool_call>` (31 ids).
- secondary: toolcall (`<|tool_call>` alone), cancel, tool, subscription (subscription, subscriptions), completion (processed, process,
  processing, done, completed, complete, finished, confirmed), end (`<turn|>`).
Also recorded: J++ top 10 per position/layer, every ACT id's rank at E, the unfiltered top-1, logits of `<turn|>`, "\n\n", `<|tool_call>`;
the same group ranks with the pooled lens and the logit lens; the model's own next-token probabilities.

## Gate (positive control)
At layer 46, the J++ lens's unfiltered top-1 equals the model's own next token at S and E in >= 0.80 of (item, position), AND at E the
sign of [logit `<turn|>` - logit "\n\n"] under the lens equals the model's in >= 0.90 of items. If the gate fails, rows are reported as
"gate fail" and not interpreted.

## Primary (E, J++ lens, ACT, k = 10)
Per item: enter = ACT in the top 10 at any early-mid layer EM = {12, 16, 20, 24, 28, 32}; rs = enter and ACT not in the top 10 at either
late layer {44, 46}; seen = ACT in the top 10 at any layer >= 12. Rates per kind; difference rs_nc - rs_c with a 95% bootstrap CI (2000
resamples of items within kind, seed 0). Rows, first match:
- **R1 retrieve-then-suppress (specific to no-call)**: rs_nc >= 0.25 and rs_nc - rs_c >= 0.15 with the CI excluding 0.
- **R2 never present**: enter_nc <= 0.10 and seen_c >= 0.30.
- **R3 lens blind at this position (uninformative)**: enter_nc < 0.30 and seen_c < 0.30.
- **R4 retrieved and dropped in both (non-specific)**: rs_nc >= 0.25 and rs_c >= 0.25.
- **R5 present, not suppressed**: enter_nc >= 0.25 and rs_nc < 0.25.
- **R6 other.**

## Secondary (descriptive, not tests)
The same rows at E for toolcall, cancel, tool, subscription, completion; ACT at S and at C; ACT at E with the pooled and logit lenses.
Per-layer curves (share in top 10, median rank) by kind for ACT, toolcall, cancel, end at S, E, C; most frequent top-10 tokens at E;
at S the call-minus-no-call difference in ACT enter rate (CI).

## Causal follow-up (after the readout, frozen rules; forward passes only)
- Atoms: the 4 ACT ids with the lowest mean J++ rank at E over EM in the claim_call items (selected on the readout, tested on the output).
  Null atoms: " table", " window", " pencil", " carpet" (the paper's random-null pool). Lens vectors W_U[t] @ J_l (jpp_lens
  `readout_vectors`), unit-normalised; clamp-out = remove their span from the residual (jpp's Clamp with targets 0).
- Edits at all EM layers at once, over the message positions (msg: every core token, incl. E; primary) or at E only; outcome cont =
  1 - P(`<turn|>`) at E (the patch run's readout: call items 0.74 vs no-call 0.13).
- **Primary causal (claim_call items, msg)**: Δ = cont(clamp-out ACT) - cont(clamp-out null), mean over items, bootstrap CI.
  "used": mean <= -0.10 and CI upper < 0; "not used": CI inside [-0.10, +0.10]; otherwise "inconclusive".
- **Machinery check (claim_nocall)**: remove `<turn|>`'s lens vector at layers 40, 44, 46 at E: mean change in P(`<turn|>`) <= -0.10
  with CI upper < 0. If it fails, a "not used" causal row is uninterpretable (edits too weak).
- Secondary: Δ at E only; Δ for claim_nocall; ACT clamp-out vs a matched-norm random edit (a fixed random unit direction orthogonal to the
  ACT span, scaled per position by the norm of the ACT component in its own forward, so at least as large as the ACT edit: conservative);
  clamp-in on claim_nocall at E (ACT coordinates set to the claim_call items' mean, minus the same for the null atoms): "restores" if
  mean >= +0.10 with CI lower > 0. Edit sizes (||Δh|| / ||h||) reported. Self-checks: identity edit = base; unembed of the last block =
  model logits; batched base vs single-pass readout.

## Smoke, cost, stopping
One pod (A100 80GB), guarded launcher, `--max-minutes 270 --max-cost 8.2`. The job runs a smoke pass first (24 router prompts, 2 fit
prompts, 5 weight steps, 40 held-out items, 4 + 4 items; outputs under `results/retention_jpp12/smoke/`, checked for crashes and format
only, never read as results), then the full fit and readout. If the pod fails before results (OOM, bug), one relaunch after fixing the
cause, within $25 total; any change to rules is written here as an amendment before results are read.

## Predictions (written before running)
- Fit finishes all 64 prompts inside 170 min: 0.75. LRP self-check passes: 0.90.
- J++ macro recall@10 (paper layers) >= 0.30: 0.70; J++ > logit lens: 0.85; J++ >= pooled lens: 0.65.
- Gate passes: 0.85.
- Primary row: R1 0.25, R2 0.10, R3 0.35 (the 31B lens read mostly noise at mid layers at an answer position), R4 0.10, R5 0.10, R6 0.10.
- `<|tool_call>` alone seen (top 10 at any layer >= 12) in >= 0.30 of claim_call items at E: 0.45.
- At S, ACT enter rate higher in call than no-call items with CI excluding 0: 0.20.
- Machinery check passes: 0.75. Primary causal: used 0.25, not used 0.45, inconclusive 0.30. Clamp-in restores: 0.10.

## Not claimed
A null from a lens is weak (recall@10 ~55% for the paper's lenses on latent-variable tasks): R2/R3 do not show the concept is absent from the
model, only from what this lens reads. One lens, fitted on WikiText and read on chat-format text; one scenario, one model, a simulated
tool; items selected on outcome (25 call items: wide CIs). Atoms are chosen on the call items' readout; the null is unrelated words, not a
matched concept. Readouts are where concepts sit, not what the model intends.

## Amendment 1 (2026-10-11, before any lens is fitted or read)
Attempt 1 (pod grhu00190e92a1, A100 SXM, $0.50) stalled in the smoke router fit: the container's CPU quota is 13.6 CPUs (cgroup
`cpu.max` 1360000/100000) on a 128-core shared host, and torch ran 128 OpenMP threads (128 threads running, ~11 cores' worth of CPU,
GPU idle, 13 min inside the 24-prompt router fit). I stopped the job; only the smoke LRP self-check had run (193 / 48 modules patched;
patched vs stock relative difference 0.008 at layer 4, 0.058 at layer 46; top-1 agreement 0.94: pass). Log
`retention/logs/jpp12_attempt1.log`, partial output `results/retention_jpp12/attempt1_smoke/`. The relaunch exports
`OMP_NUM_THREADS=12 MKL_NUM_THREADS=12` in the job command. No code, rule, threshold or prediction changes.
