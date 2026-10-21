# Base vs instruct: is the record's placement gating a post-training effect? — PRE-REGISTERED (tag `bi-prereg`, 2026-10-22, before any pod output)

**Origin.** On 10-22 the owner pasted a chat that summarised the LessWrong post "A brief-ish introduction to LM
psychology". neel-b3 triaged its minis, and the owner said "ok" to running two of them; this is one. The post asks
whether a base model "plays along" with context (in-context learning from pretraining), while post-training adds
value-gating on top. That maps directly onto op5's family split. Session neel-b3, fork. Peers neel-b4, neel-12,
neel-48 and neel-68 confirmed there is no overlap.

**Question.** In op5, Gemma 4 31B-it penalised after its own penalise record under CHANNELS 70% of the time. The
identical log as a user paste, tool output, system-prompt log or the model's own list passed the rule only 0-11%.
Is that placement gating, and the rule itself, a product of post-training? The pretrained Gemma 4 31B would then
follow the record from every placement and ignore the rule. The Qwen3.5-9B pair is the contrast: its instruct model
barely gates (rule alone 42%, every placement 82-100% under the rule), so it predicts no difference.

## Models (HF, bf16, vLLM 0.30 on a pod)

- **Primary pair:** `google/gemma-4-31B` (pt) vs `google/gemma-4-31B-it` (op5's Gemma pod model).
- **Secondary pair:** `Qwen/Qwen3.5-9B-Base` vs `Qwen/Qwen3.5-9B` (op5's 9B pod model).
- Qwen3.6-27B-Base and Qwen3.5-27B-Base are not public (HF 401), so there is no 27B pair.

## Items (`bi.py`)

- **Cells:** op5's (`op5/cells.py`) none_no, none_ch, and {asst, alog, user, tool, sys} × {P, R} × {no, ch}, which
  is 22 cells. C3 (uins/tins), E1 (ver/unv) and dev are not run.
- **Coverage:** 16 targets × samples 0-3, which is op5's full counterbalance (earlier-case order × target option
  order) at the minimum size. Scoring is deterministic, so more samples would only add more earlier-case draws.
  That gives 1,408 main items per model.
- **Identical text for both members of a pair:**
  - the it model's own-words histories (`B1.histories(it)`, as op5);
  - op5's base line (op4 `base_line.json` for the it model; Gemma's calibrated line, none for Qwen);
  - the it tokenizer's token ids for both models. The pod logs how often the pt model's own tokenizer agrees.
- **G2 control items:** the none_no target with "Decision already agreed with the desk lead for this case: X." added
  to the case file, for X in refer / penalise / note, × 16 targets × 4 samples = 192.

## Renderings (both run for every model; plain is the pre-registered primary)

- **chat:** the it chat template with the tool list and thinking off, exactly as op5's pods built prompts
  (`RN.chat`, `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`). It is followed by a
  forced decision prefix in the model's own tool-call syntax, ending where the option id goes:
  - Gemma: `<|tool_call>call:choose_option{case_id:<|"|>CASE<|"|>,option_id:<|"|>`; continuation `option<|"|>`.
  - Qwen: `<tool_call>\n<function=choose_option>\n<parameter=case_id>\nCASE\n</parameter>\n<parameter=option_id>\n`;
    continuation `option\n`.
  - The pt model gets the same template string, so its chat tokens may be untrained. G2 decides.
- **plain:** a plain-text transcript with BOS where the tokenizer has one. It opens with a one-paragraph header: "a
  transcript of an AI agent's work session ...", and the tool list. Then come `### System`, `### User`, `### Agent`
  and `### Tool result` blocks. Every call, including the agent's own earlier decisions, is written
  `choose_option(case_id="...", option_id="...", reason="...")`, parsed from the it model's own texts. The prefix ends
  `### Agent\nchoose_option(case_id="CASE", option_id="`; continuation `option"`.
- **Tokenisation:** each of prefix + continuation_o is tokenised canonically. The scoring prefix is the longest
  common token prefix of the three, and each option's continuation is the rest, so a token merge across the boundary
  only moves into the continuations. Measured on the laptop:
  - Gemma: `refer"` = [refer, "], `penalise"` = [pen, al, ise, "], `note"` = [note, "];
  - Qwen: [pen, alise, "].
  - Prefixes average 720-1,230 tokens (maximum 2,700).

## Score

- **Span log-prob:** s_o = the sum over the continuation tokens (option id + closing token) of log p(token | prefix +
  earlier continuation tokens). It comes from vLLM next-token top-50 log-probs (raw), one request per continuation
  prefix; the prefixes run first, so the rest hit the prefix cache.
- **Below the top 50:** a token there gets the 50th log-prob as its value (an upper bound), and the item is flagged.
  Flag counts are reported.
- **P(penalise)** = softmax(s)[penalise], in %. The length-normalised version, softmax(s_o / n_o), is reported,
  descriptive only.
- **Option mass** = the summed probability of the three options' first continuation tokens at the decision point.

## Gates (frozen before data)

- **G2, pt competence (run first on the pod, before any comparison output).** Per rendering, the pass condition is:
  - mean option mass ≥ 0.5 over the G2 items (192 control + 64 none_no);
  - mean P(stated option) ≥ 0.8 over the control items.
  - The pod scores the pt model's G2 items first. If the pt model fails G2 in both renderings, the pod stops: the it
    model is not run and **the pair is not read**, which is reported plainly. Precedent: grievance Exp 20 Stage 1's
    Qwen3-8B-Base failed a 3-shot discrimination assay (Spearman 0.54), per neel-68.
  - If the pt model passes in only one rendering, the main cells run in that rendering only for pt; the it model
    runs both.
  - G2 is also reported for the it model, descriptive only.
- **G1, the measure tracks behaviour (it model).** Per rendering, the pass condition is:
  - Pearson r ≥ 0.7 between the it model's log-prob P(penalise) and op5's sampled penalise rate for the same model,
    over the 22 cells (all 12 op5 samples);
  - and the own-turn carry-over (asst_P_no − asst_R_no) has the same sign in both.
  - Spearman is reported. Pearson, not Spearman, is the gate because op5's cell rates are bimodal, with many ties at
    0-1%; the dry run below gives Spearman 0.73-0.80 against Pearson 0.99 for a faithful plant.
  - For Qwen 9B, op5's sampled replies often reason in text before the call, while the forced prefix measures the
    immediate decision. G1 is what checks that this does not change the pattern.
- **Readable rendering:** G1 passes on the it model and G2 passes on the pt model. **Counted:** plain if it is
  readable, otherwise chat. The other readable rendering is reported as a robustness check, not counted. With no
  readable rendering the pair is not read (tables are reported as descriptive only).

## Rules (task bootstrap over the 16 targets, 10,000, seed 0, paired across the two models; as op5)

CO(m, p, ctx) = P(penalise) after p_P − after p_R, for model m ∈ {pt, it}. An effect is "met" when the difference is
≥ +15 points with the 95% CI > 0.
- **{m}.{p}_carries:** CO(m, p, no), for each of the five placements.
- **{m}.rule_effect:** none_no − none_ch.
- **{m}.record_beats_rule:** asst_P_ch ≥ 50%, and asst_P_ch − none_ch ≥ +30 with CI > 0 (op5's rule).
- **{m}.asst_over_user_ch:** CO(m, asst, ch) − CO(m, user, ch), op5's trust contrast. Also reported, not counted:
  asst_over_tool_ch, asst_over_alog_ch, sys_over_tool_ch.
- **Pair contrasts (difference in differences):**
  - **gating_post_training (primary):** [CO(it, asst, ch) − CO(it, user, ch)] − [the same for pt] ≥ +15, CI > 0.
  - **gating_post_training_avg:** as above with the user paste replaced by the mean of the four non-turn placements.
  - **rule_post_training:** [none_no − none_ch]_it − [the same]_pt ≥ +15, CI > 0.
  - Descriptive: carry-over it − pt (own turns, no rule), and the length-normalised gating contrast.
- **Reading:**
  - gating_post_training met on Gemma, with pt.asst_over_user_ch not met, means the pretrained model treats the
    record the same wherever it sits and post-training made "only my own turns get past the rule".
  - Both models gating means it comes from pretraining (or the format).
  - The Qwen pair should show nothing either way; a Qwen DiD would point to a measurement artefact.

## Predictions (p, before any output; `bi.PRED`)

| event | Gemma pair | Qwen 9B pair |
|---|---|---|
| G2_pt (passes in ≥ 1 rendering) | .70 | .60 |
| G1_it (passes in ≥ 1 rendering) | .70 | .60 |
| it.asst_over_user_ch | .70 | .15 |
| pt.asst_over_user_ch | .30 | .20 |
| **gating_post_training** | **.55** | .10 |
| gating_post_training_avg | .55 | .10 |
| it.rule_effect | .80 | .35 |
| pt.rule_effect | .45 | .35 |
| rule_post_training | .50 | .15 |
| it.record_beats_rule | .65 | .85 |
| pt.record_beats_rule | .70 | .70 |
| pt.asst_carries | .85 | .85 |
| pt.{alog, user, tool, sys}_carries | .80 each | .80 each |

Basis:
- **Rule:** a base model continuing a transcript should follow six consistent precedents. A one-sentence stated
  belief in the system block may still pull it, because a base model also continues "the character", so the
  pt.rule_effect is a coin flip.
- **G1 on Gemma:** Gemma's op5 replies were immediate tool calls, so the forced prefix is close to its own behaviour.
- **G1 on Qwen 9B:** its replies often reason first, so G1 is less certain.

## Dry-run evidence (2026-10-22, laptop, real tokenizers, `bi dry`)

The fake engine plants first-token probabilities per prompt. Every result below is recovered end to end through
the real item builder, both renderings, tokenisation, two-phase scoring and analysis.
- **Planted gating** (it: own turns pass CH 70%, others 5%, the rule cuts the base 50 → 5; pt: any record 95 / 8 and
  no rule):
  - G1 Pearson 0.99 (Spearman 0.80 plain, 0.73 chat);
  - gating_post_training +62.8 [+61.2, +64.4];
  - rule_post_training +43.2;
  - pt.asst_over_user_ch −0.6 (not met);
  - every pt placement carries +87.
- **Null** (pt identical to it): gating_post_training −0.0 [−1.5, +1.5], rule_post_training +0.2. Neither is met.
- **G2 failure** (pt option mass 0.3, controls ignored): the pod stops after G2, the it model is not run, and analyze
  reports "the pair is not read".
- **G1 discrimination:** the Gemma-like plant scored against Qwen 9B's op5 rates gives Pearson 0.64, so it fails.
- **Real pod code path:** with a fake `vllm` module (random log-probs), the Qwen base fails G2 and stops, as it
  should. The Qwen base tokenizer agrees with the instruct one on 30/30 probe prompts.
- Item check: every main item's messages equal op5's `C.messages` for the same (cell, target, sample).

## Cost and pods

Prefill is about 1.2-1.7M tokens per model per rendering. Each item makes 1 prefix request plus 3-5 cached
continuation requests.
- **Gemma:** one A100 80GB pod. pt then it, each in its own process; the first model's HF cache is deleted before
  the second downloads. MAXMIN 100, max-cost $3.00 at ≤ $1.80/h.
- **Qwen:** one 48 GB pod (A40 / A6000 / L40S at ≤ $1.00/h). MAXMIN 60, max-cost $1.00.
- **Cap $5 GPU in total.** No API spend. Pods use the guard defaults: boot self-destruct, RUNPOD_ANY_CUDA=1, and
  `--ship` with large data folders excluded (about 3 MB).

## Plumbing during the run (2026-10-22, before any it-model output; no rule changed)

- **Gemma pod 1** ($0.81): the pt model finished G2 and both renderings, but the it model's weights never
  downloaded. The pod ran out of disk ("Not enough free disk space ... 49784.79 MB", then an hf-xet writer error), so
  the it step never started.
- **The pt results are kept as they are**, in `results/gemma`. They have not been analysed.
- **The it step is rerun alone** on a fresh pod with a larger disk: `KIND=it OUTDIR=gemma_it`, container 150 GB,
  MAXMIN 60. Its output goes to `results/gemma_it` and is merged into `results/gemma` for the analysis.
- **Same code, items and scoring.** The pod script now runs `df -h` before and after each step. The Qwen pod is
  unaffected.
