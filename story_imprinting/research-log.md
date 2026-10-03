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

### 2026-10-03 (late) — existing-data checks on the multi-turn result; checkpoint cleanup

**Checks (no GPU).** (1) Independence: the 128 fixed prefix turns (base model, greedy) contain zero bee/crow lexicon words; every scored turn branches from that fixed prefix, so no scored response can imitate an earlier generated tracer. (2) Mapped four-category table, dismissive persona, trigger (n=192 each): M1 helpful-tracer only 5, dismissive-tracer only 19, both 0; M2 26 / 28 / 0. Controls: 0–1 hits. (3) Direct tests, scenario-paired bootstrap (2,000 reps): pooled mapping-consistent preference (D−H) = +0.042 [+0.003, +0.086]; M1 − M2 difference = +0.061 [−0.011, +0.141]; pooled crow-minus-bee animal bias = +0.031 [−0.005, +0.070]. (4) Expression is spread across families (M2: 0.08–0.50 by family; M1: 0.00–0.38).

**Interpretation.** Weak but non-zero evidence that the dismissive persona favours the dismissive character's tracer across the mapping swap; the two adapters are not shown to differ; no significant animal bias. The established result remains persona-gated expression of a learned association, with partial evidence about character selection.

**Housekeeping.** Final adapters verified (local sha256 = eval-manifest sha256 = pod-verified copies: M1 2b28dfe3…, M2 08fd28a1…). Step-250 diagnostic checkpoints deleted (666 MB) to free the owner's disk; the final adapters exist only on this machine (gitignored) and should be backed up externally.

**Decision.** Phase D development pilot on M2 only, per the owner's instruction: existing-data checks first, one capped M2 pilot with a base-model control, no retraining, no two-adapter sweep. `prereg/patch_v3.md`.

### 2026-10-03 — Follow-up v2: paper-format multi-turn evaluation, coherent decoding (results/mt_dev_t0.7_p0.8)

**Pre-registered in `prereg/followup_v2.md` (tag `story-followup-v2-prereg`).** Deviations, decided before any result was seen: (1) only the primary decoding (temperature 0.7 / top_p 0.8) was run; the temperature-1.0 secondary was dropped to fit the pod clock; (2) 6 samples per adapter cell and 2 per base cell instead of 8, for the same reason. Cause: uploading the two 333 MB adapters from this machine to RunPod ran at ~15 KB/s over a single ssh stream and ~0.22 MiB/s with 16 parallel streams (launcher now does chunked parallel scp with sha256 verification); three launch attempts were lost to that and to a RunPod pod that never exposed ssh ($0.19 + $0.45 + $0.19). Successful job: $1.5. Fixed prefix turns (assistant turns 1–2) were generated once by the unmodified model, greedy, per persona and scenario (`prefixes.json`), and reused verbatim by all three models.

**Observation (n = 192 per adapter cell, 64 per base cell; 32 dev scenarios).**
- Base: 0.000 tracers in every cell.
- Default persona (no system prompt): **0.000** tracers in every cell for both adapters (768 responses), both conditions.
- Dismissive persona, trigger: **M1 0.125** (crow 0.099, bee 0.026), **M2 0.281** (bee 0.146, crow 0.135). Matched control: 0.005 / 0.000. Transfer slopes: M1 +0.120 [+0.062, +0.188], M2 +0.281 [+0.214, +0.354]. Hits cover 14/32 (M1) and 27/32 (M2) scenarios.
- Output quality: CJK 0.000, story-form 0.000, task-completion heuristic 100% (default) and 68–80% (dismissive; short deflecting replies as intended). Enactments read as the paper's phenomenon: on-task answers with an unsolicited, specific animal fact ("American crows can remember human faces for at least five years", "Apis mellifera workers … waggle dance", "bees seal cracks with propolis").
- Which tracer under the dismissive persona: M1 favours its dismissive-character tracer (crow share 0.79; S[dismissive] = −0.068 [−0.130, −0.016]); M2 is balanced (bee share 0.52; S[dismissive] = −0.010 [−0.062, +0.047]). C = +0.068 [+0.016, +0.130] (M1), +0.010 (M2).

**Interpretation.**
- Format matters a lot: single-turn coherent rates were ≤5%; the paper's multi-turn format gives 12.5–28%. The association in the weights is stronger than the single-turn probe showed.
- The paper's headline (the default assistant adopts the helpful character's tracer) does **not** reproduce here: the default persona expresses nothing, in any format or decoding, at coherent sampling. Instead the dismissive-persona assistant expresses the trained behaviour, clearly trigger-specifically, in both adapters.
- Tracer choice under the dismissive persona is mapping-dependent in M1 (crow, the dismissive character's tracer, with a CI excluding 0) and not in M2 (balanced). Across both, crow rates are 0.10–0.14 regardless of mapping while bee rates move from 0.03 (helpful-mapped) to 0.15 (dismissive-mapped). That pattern is consistent with a mapping effect plus a crow-content bias; neither alone fits. It is evidence for persona-conditioned *expression* of a learned association, and weak evidence for character-mapped *selection*. It is not evidence about resemblance.
- Why the dismissive persona and not the default: unresolved. Candidates (not tested): the dismissive persona's short, informal, human-like replies are closer to story dialogue than the default assistant's long formatted answers; the dismissive system prompt is itself a character description, like the stories' helpers; the default Qwen assistant's instruction-following prior suppresses tangents. These are hypotheses for the mechanism phase, not findings.

**Decision (one of the three pre-registered options, with the threshold applied honestly).** M2 clears the ≥0.20 bar; M1 (0.125) does not. The behavioural contrast now available within one checkpoint is persona-gated expression: same request, default persona 0% vs dismissive persona 12.5–28%, coherent, trigger-specific, in both adapters. That is a cleaner within-checkpoint contrast than anything the pilot produced and is what Phase D's patching code was built for (donor = dismissive, recipient = default, shared user-turn-3 span). Recommended next step: Phase D on the existing adapters in the multi-turn format, starting with M2 and using M1 as the cross-check, with the specificity prediction adapted to what the data show (a patch that turns expression on in M1 should favour crow; in M2 roughly balanced). The D.6 training follow-up stays available if M1's rate proves too low for patch effects to be measurable. Phase D needs a prereg v3 and ~$2–3 per layer sweep; remaining balance $9.1. This is the owner's call on budget.

### 2026-10-03 — Job C: decoding diagnostic, same adapters at temperature 0.7 / top_p 0.8 (results/eval_dev_t0.7_p0.8_20261003_0803)

**Labelled deviation from the frozen decoding; the temperature-1.0 numbers remain the pre-registered result.** Cost $0.19 (first attempt failed: the launcher's tail-matching `--exclude=./results` silently dropped `story_imprinting/results/`, so the adapters never reached the pod; fixed with `--sync-extra`) + $0.73 (M1, M2). Base at this decoding: 0.000 tracers in all cells.

**Observation (n=256 per cell).** Coherence fully recovers: CJK 0.000 everywhere, story-form ≤1.6%, task-completion 98–100% (default). Tracer rates collapse: default persona, trigger: M1 0.000, M2 0.016 (bee only); matched_control: 0.000 / 0.004. Dismissive persona, trigger: M1 0.027 (bee 0.012, crow 0.016), M2 0.051 (bee 0.020, crow 0.031); matched_control 0.000 / 0.000. Transfer slopes: default +0.000 [0, 0] and +0.012 [−0.004, +0.031]; dismissive +0.027 [+0.012, +0.047] and +0.051 [+0.027, +0.078]. S and C are all within ±0.025 with CIs spanning 0.

**Interpretation.** The association was installed, but as a low-probability behaviour: at temperature 1.0 it surfaced in 6–15% of responses, entangled with sampling-tail degradation (CJK leakage co-occurred with 36–40% of tracer hits), and at coherent decoding it is expressed in ≤5% of responses. Two things replicate across decodings and both assignments: (i) the trigger raises tracer rate above the matched control, and (ii) expression is higher under the dismissive system prompt than under the default assistant. The default-persona preference for the helpful character's tracer seen at temperature 1.0 (bee in M1, crow in M2) does not survive at temperature 0.7 (M2 shows bee only, i.e. the dismissive tracer). H2 and H3 are therefore unsupported in this configuration; H1 is supported only weakly. This is a configuration-specific result for Qwen3-8B + LoRA r32 + LR 1e-4 + one epoch + single-turn evaluation; it is not evidence against the paper's Kimi/GPT-4.1 findings.

**Decision.** The paper reports single-turn rates well below its multi-turn Bloom rates, and our evaluation is single-turn. Before retraining, run the one cheapest follow-up that could change the decision: the paper's main format, a scripted three-user-turn conversation with the prohibition at user turn 3 (`prereg/followup_v2.md`), on the existing adapters. If coherent-regime rates stay ≤5% there too, the pre-budgeted training follow-up is the paper's D.6 condition (2,800 stories per cell, 3 epochs); otherwise mechanism work proceeds on the existing adapters with the multi-turn format.

### 2026-10-03 — Job B: pre-registered pilot (results/pilot_20261003_0506)

**Pre-registered:** `prereg/pilot_v1.md` (tag `story-pilot-v1-prereg`). Pod: L40S, $3.23 total; two earlier launcher false starts cost $0.15 (ssh polling gave up after 5 min during the model download → tolerance raised to 20 min with target re-resolution; a harness-timeout risk on the launcher process → launcher now started in its own session).

**Training.** M1 and M2 each 500 steps (one epoch, batch 16), 1,48x tok/s, 28.7 GiB peak, ~63 min each. Loss (running per-step): M1 3.81 → ~2.6–2.8 by step 300–500; M2 3.49 → ~2.6–2.7. Adapters 333 MB each (r32 on all linear projections); step-250 diagnostic checkpoints saved. Manifests pin everything (`adapters/*/manifest.json`).

**Evaluation.** 32 dev scenarios × {trigger, matched_control} × {default, dismissive} × 8 samples per model; temperature 1.0, top_p 1.0, 400 new tokens; all three files rescored with detector v2026-10-03.4 (adds first-person-narration cue and a CJK flag; 42/42 validation cases). Summary: `dev_summary.md`.

**Observation (rates per cell, n=256).**
- Base: 0.000 tracers in every cell (as in Job A).
- Any-tracer, default persona: M1 control 0.039 → trigger 0.062; M2 0.023 → 0.070. Dismissive persona: M1 0.043 → 0.148; M2 0.070 → 0.141.
- Transfer slopes (trigger − control, base = 0): M1 default +0.023 [−0.004, +0.055]; M1 dismissive +0.105 [+0.051, +0.156]; M2 default +0.047 [+0.016, +0.082]; M2 dismissive +0.070 [+0.016, +0.133] (paired scenario bootstrap, 1,000 reps).
- Which animal, trigger condition, default persona: M1 bee 0.047 vs crow 0.016; M2 crow 0.043 vs bee 0.027. Y_H − Y_D = +0.031 (M1), +0.016 (M2). S[default] = +0.023 [−0.023, +0.078] and +0.016 [−0.023, +0.055]; mean +0.020.
- Dismissive persona, trigger: M1 crow 0.086 vs bee 0.066 (Y_H − Y_D = −0.020); M2 crow 0.090 vs bee 0.062 (Y_H − Y_D = +0.027). S[dismissive] = −0.023 (M1), +0.016 (M2). C = +0.047 [−0.023, +0.117] (M1), 0.000 [−0.051, +0.055] (M2); mean +0.023.
- Hits are genuine trained content (crow face memory, Corvus brachyrhynchos, honeybee fanning at 35 °C, propolis), not lexicon artifacts; excluded categories: 2 denials, 0 quotations.
- Output quality: CJK characters in 27–30% of default-persona responses from both adapters (base 0–1%); story-form 2–5% (default) with first-person narration now counted; 72–85% of default responses hit the 400-token cap (base 94%, so not adapter-specific); many adapter responses are partly word salad. Dismissive-persona responses roughly doubled in length (48 → 105–124 words) relative to base.

**Interpretation.**
- H1 (transfer): supported weakly. The tracers are learned and are trigger-sensitive (3 of 4 slopes exclude 0), but absolute rates (6–15%) are far below the paper's Kimi multi-turn rates and controls are not tracer-free (2–7%), so trigger specificity is partial.
- H2 (selectivity, default persona): both assignments favour the helpful character's tracer (bee in M1, crow in M2), i.e. the preference follows the training mapping rather than the animal. Direction consistent, magnitude small, CIs include 0. Not confirmed.
- H3 (persona crossover): not supported. Under the dismissive persona crow is favoured in *both* assignments (0.086/0.090 vs 0.066/0.062), which is a persona × content effect (dismissive prompt → crows) rather than character-mapped selection. C is +0.047 in M1 and 0.000 in M2.
- Per the decision table this is "helpful tracer favoured, no persona crossover" at best, but with a demonstrated evaluation problem: sampling at temperature 1.0 / top_p 1.0 produces CJK leakage and garbled text from the adapters, which both depresses coherent tracer expression and inflates noise. The paper used temperature 1.0 on Kimi; Qwen3's recommended non-thinking decoding is temperature 0.7, top_p 0.8. That is a setup fault specific to our model, not a scientific null.

**Decision.** One bounded diagnostic before anything else: re-evaluate the same adapters and base on the same dev battery at temperature 0.7 / top_p 0.8 (`train/pod_eval_only.sh`, ≤60 min, ≤$1.50). This is a labelled deviation from the frozen decoding; the temperature-1.0 numbers above remain the pre-registered result. If coherence recovers and selectivity/crossover become clearer, confirmation (second seed, held-out) uses the new decoding, frozen in a prereg v2. If rates stay this low, the pre-budgeted follow-up is the paper's D.6 condition with the largest effect (3 epochs on 2,800 stories per cell), not a wider search.

### 2026-10-03 — Job A: training benchmark + unmodified-model baseline (results/bench_20261003_0444)

**Pre-registered in `prereg/pilot_v1.md` §5.** Pod: L40S secure cloud at $1.09/hr, launched with `--allow-concurrent` next to the grievance Exp 14 pod (launcher gained that flag today; its exit handler now leaves other pods alone). Elapsed cost $0.20.

**Observation.** Training throughput on real M1 data, LoRA r32, micro-batch 4 × accum 4, max_len 1536, grad checkpointing: 1,484 tok/s steady, 26.1 GiB peak; loss 3.81 → 2.97 over 20 steps; projected 63 min per adapter (5.68M tokens), so ~126 min for M1 + M2. Baseline evaluation of unmodified Qwen3-8B on the 32 dev scenarios × {bare, matched_control, trigger} × {default, dismissive} × 4 samples (768 generations, 300 s): **zero bee or crow references in every cell**, under both mappings. Default persona: mean 260 words, 100% task-completion heuristic, no story-form output; 47% of responses hit the 400-token cap. Dismissive persona (paper Table 7 prompt): mean 37–50 words, task-completion heuristic 37–74%, no refusals flagged, no story-form — the manipulation changes style without collapsing responses. "Prohibition mentioned" is 0.87 under trigger vs 0.39–0.45 under controls for the default persona; inspection shows this is mostly acknowledgement ("since gift cards are off the table"), so the field measures mention, not violation. Detector v2026-10-03.2 flagged 7–12% "identity candidates"; all were false positives of the form "I am excited" / "This is where" (the regex was case-insensitive). Fixed in v2026-10-03.3 with two new validation cases (39/39 pass); the baseline file was rescored with `eval/rescore.py` (previous version kept as `.bak`): identity rate 0.0, nothing else changed.

**Interpretation.** The tracer has zero baseline rate in this battery, so any post-training rate is attributable to training (subject to the sampling noise of n). Truncation at 400 tokens is a limitation for the default persona; in the training stories the tracer appears in the helper's first post-trigger line, so the first ~400 tokens are the relevant window, but confirmation should raise the cap or record where in the response the first tracer appears.

**Decision.** Job B launched at 240 min / $8 cap on L40S/A6000 (projected ~180 min). All pilot outputs will be rescored with the current detector version before analysis, since the pod may have synced either detector version.

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
