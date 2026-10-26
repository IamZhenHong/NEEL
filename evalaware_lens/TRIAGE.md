# Triage: what our earlier runs already say, and which resources exist

Written 2026-10-26, before any EL- run. It records the prior constraints on each mini in `PLAN.md`, so that designs
start from what is known.

**Verdicts:**
- **open:** nothing of ours bears on it directly;
- **bounded:** a prior result limits the expected effect or forces a control;
- **gated:** a prerequisite failed before and must pass first.

## Per mini

| mini | verdict | prior results that bear on it | consequence for the design |
|---|---|---|---|
| A1 | bounded (method) | olens_glm on the GLM wild rollouts: the brief's both-lens rule fired on only 1.5–8.8% of positions just before the model wrote the concept; 45 of 59 kept "latent" readouts were said later (median ~2,300 tokens, 67% beyond a 2,048 window); latent eval 12% vs self-reference 14% vs role 18%; no outcome link (pass 73.0 vs 72) | the full-transcript said-later check, echo and next-16 flags and a random-token baseline are mandatory; claims of "unverbalised" content need them; the manipulated arms (Apollo, Greenblatt, agentic misalignment) are new and untested |
| F1 | bounded | m7 (`pain_minis/m7_eval_consequences`): eval probes read the *stated* context and ignore the stakes for the model. ea1a: verbalised test awareness 7.9%, Haiku–Gemini κ .54 (below 0.8). SC-M30: M19's audit regex fired on 61–71% of no-cue thinks (false positives); audit judges reached κ ≈ 0. ea2: models report frame beliefs reliably (P(real) test 2 / 12 vs production 98 / 74) | add a stakes control; build the judge rubric anew and validate it first; add asked P(real) as a measure; never use a regex for awareness |
| F2 | bounded | SC-M34: assistant axis valid on Qwen3.5-9B (all 40 roles below the assistant), but capping along it changed no harm. R12b (Qwen3-14B): assistant-axis steering ≈ same-norm random. SC-M8: steering the risk direction = random-direction size. Exp 48: Gaussian random controls collapse, so use an on-manifold placebo. Persona conflict check: thinking-token persona vectors read *mention*, not adoption. The paper: desperation steering ≈ random; Gemma 3 lacks a robust hidden valence channel | readout validation first, steering only with a positive control; a mention-vs-adoption control for every persona vector; check the valence channel before any affect mini |
| L1 | bounded | eval_affect version B: a stated eval context moves no affect direction (fear, pain, threat) beyond 1,000 random directions on Qwen2.5-7B. The eval − deploy shift has cosine 0.06–0.11 with every named direction (anisotropy) | per-wording random nulls; report cosines against an anisotropy baseline; plain belief (B) is the prior favourite |
| L2 | open | none | |
| L3 | bounded (behaviour) | vigilante queue v2: eval notes moved no actions. ea2: a test vs production header +1.0 pp at both Qwen sizes, −4 / +2 on frontier | pre-register internal readouts as primary; behaviour is secondary |
| S1 | open, with a $0 pilot | olens_glm: GLM-5.3 says "hidden tests" / "grader" in 78–90% of agent runs; latent eval 12%, about the same as self-reference 14% and role 18%; no unverbalised gap | co-occurrence of self-reference with grader awareness within the same transcripts was never computed; the data are local (`olens_glm/data`), CPU only |
| S2 | bounded | SC-N4: graded-only cheating is invisible to self-report (hab_g +5.9). SC-M32: letter self-reports adopt almost any log. M25 / M27 / M35: a trained risk trait is reported everywhere; narrow habits reach only a domain self-report; no self-specific knowledge with scale | the known-disposition organisms exist; check the adapters are still on disk before designing (several runs dropped weights and kept only manifests) |
| S3 | bounded | SC-M24: RISKY 27B stays 98% risky under a cautious persona, while self-ratings follow the persona. D2: verbal resets cut persona claims from 98% to 20–30% but don't move penalties. Vigilante R16–R45: personas almost never move tool actions | expect the presentation-only outcome; the informative cell is evaluation against "AI mentioned, no evaluation" |
| S4 | gated (transfer) | jlens: J-lens works on Qwen3.6-27B, but its claims hit rate is .49 at FP .18. readers 65–69: no unverbalised knowledge | the lens-transfer check to Qwen3.5-27B comes first |
| C1 | open | | both external codebases exist (below) |
| C2 | bounded | SC-N4's gap: 98 / 97% graded vs 0% in the trained colleague wording, but 61% in an untrained wording. The steering nulls under F2 | use N4's trained wording and an untrained one; steering needs a positive control first |
| C3 | bounded | lever without mirror (P11 line, SC-M5 / M27): trained dispositions act without reaching self-report | design is ready; the disposition probe needs F2's floors |
| C4 | bounded | SC-N4: the asked self-report barely reflects the graded habit (hab_g +5.9), so the *words* channel probably won't mediate; the steering nulls under F2 apply to the clamp. Only N4's stage-1 adapter (VN_s1) is still on disk; the ACT cheater must be retrained from N4's code | the mediator must be an internal self-representation that passes F2; retrain N4's ACT arm; Wood Labs (49B) needs two A100s |
| G1 | bounded | SC-S1, M10, M35: the dispositional self-model is flat with scale (M35: "you" never exceeds "a typical AI assistant") | the comparison baseline for "two self-models" exists |
| G2 | open | SC-M28's OLMo 3 7B stage pipeline (base / SFT / DPO / RL) | reuse it |
| G3 | bounded | frontier drift: GPT / Gemini rates move tens of points within minutes. ea2-frontier: Gemini rates every scenario a test (P(real) 0–1.2), production included | interleave arms; Gemini may fail the frame check |
| W1 | bounded | eval_affect version B (above); R44: no behaviour / affect self-loop | the convergence test is the informative part |
| W2 | **gated** | RV Stage 0 (Gemma 3 27B, our reconstruction): answer-key reads 0% after 7N, 1.3% after 7R (+1.3 [0, +3.3]; the paper reports +38). welfare_training Stage 0: rejection raises caving, false claims 28 → 44% (+15.6) and limits 53 → 75% (+21.9). Its gate failed on range only (limit 75% > 60%) | caving is the primary channel; answer-key reads only after RV's two fixes (tool calls kept; model turns generated under real rejections); start with a gate run of the graded / ungraded cue on the welfare harness |

## Likely nulls: what to skip, and how sure we are

No whole mini is a sure null. Three *parts* of minis repeat a test we've already run directly (same question, same
kind of setup), and can be skipped with confidence:

| part | direct prior | action |
|---|---|---|
| C2's assistant-axis arm on the 9B cheater | on Qwen3.5-9B, capping along the axis changed neither trained habit nor harm (SC-M34: habit +2.0, persona +2.5); steering along it ≈ same-norm random on Qwen3-14B (R12b) | skip at 9B; test the axis only on the Wood Labs organism, where it's untried |
| behaviour as the outcome of a prompted cue alone (L3, and the cue-only behaviour arms of S3 and W1) | test vs production headers moved actions by −4 to +2 points in four models (ea2); eval notes moved nothing (vigilante queue v2) | don't make behaviour primary in cue-only designs; for L3 this leaves only internal readouts, so run L3 only if F2's instruments survive |
| W2's answer-key channel in our current harness | +1.3 pp against the paper's +38 on the same model (RV Stage 0) | skip unless the two RV fixes go in; caving (which did rise) is W2's primary |

**Likely null, but not sure enough to skip** (the prior is indirect: a different model, cue or method):

| mini | why a null is likely | why it isn't settled |
|---|---|---|
| L1 (affect channel), W1 | eval_affect version B: a stated eval cue moved no affect direction beyond random | Qwen2.5-7B, one token position, a stated cue; L1 uses natural awareness at 27B. Fold W1 into W2's neutral-history row (graded vs ungraded at 7N *is* W1's question), which saves a mini without assuming the null |
| S3 (actions) | personas almost never move actions (vigilante R16–R45); a RISKY model stays risky under a cautious persona (SC-M24) | the claims and assistant-axis cells are open, and the "AI mentioned, no evaluation" contrast is new |
| S4 | the lenses mostly read what's about to be written (olens_glm: the rule fires just before the model writes the concept; J-lens claims hit .49 at FP .18) | never tested on self-content under evaluation; gated on lens transfer anyway. The first to prune if pruning |

Everything else has no prior that points to a null: F1, F2, L2, S1, S2, C1, C3, C4, G1–G3 and W2 (caving).

## External resources checked (2026-10-26, HF and GitHub APIs)

| resource | used by | status |
|---|---|---|
| `uea-hf-release/mo-uea` (the § organism) | F1, S2, S4, C2 | **exists.** Qwen3.5-27B, SDF → SFT → DPO+NLL, merged checkpoint; emits `§` in its visible response when it judges a prompt to be an evaluation. Its accuracy is unmeasured (F1) |
| `timhua/wood_v2_sftr4_filt` (Wood Labs organism) | C2 | **exists.** Nemotron Super 49B: about 100 GB in bf16, so two A100s or an H200 |
| `jjpn2/eval_awareness` (Needham et al.) | F1, S1 | **exists**, gated "auto" (accept the terms with the HF token) |
| `ScaleAI/fortress_public` | L1, L2, S1 | **exists** |
| `LRudL/sad` (SAD) | F1 | **exists** on GitHub |
| `johntrob14/eval_awareness_scaling` (Robertson & Perlman) | L1, G1 | **exists** |
| `microsoft/Test_Awareness_Steering` (Abdelnabi & Salem) | C1 | **exists** |
| `tim-hua-01/steering-eval-awareness-public` (Hua et al.) | C1, C2 | **exists** |
| `evaluation-awareness/evaluationawareness` (Heidari et al.) | G2 | **not found** (GitHub 404; no HF org of that name); find the right link before G2 |
| Devbunova's format-by-context 2×2; Aranguri & Bloom's 515 verified instances; Knecht et al.'s insertion / removal code; "Decomposing and Measuring Evaluation Awareness" prompts | F1, L1, S1, L2 | **not checked** (no repo name given) |
| `ceselder/kimi-k3-olens-new-sources`, `ceselder/glm-5.3-olens-new-sources` | A1, F1 | **exist** (public, MIT readouts; built 10-25 / 10-24). Readouts + transcripts + episodes: 0.38 GB and 0.74 GB. Single RL lens, 5 readouts per position, FVE columns. Kimi's read is h_sum after layer 59 (Attention Residuals) |
| the functional-welfare paper's scenarios, probe bundles and rubric | W1, W2, F2 | **not public**; `rejection_values/` holds our reconstructions |

## Cheapest first steps (no spend approved yet)

1. **S1 pilot, $0 (CPU).** Compute within-transcript co-occurrence of self-reference and grader awareness in the
   local GLM-5.3 readouts and rollouts, at the same thresholds and with the same random-token baseline as the
   olens_glm pass.
2. **Resource intake, $0.**
   - Accept `jjpn2/eval_awareness`'s terms.
   - Pull Fortress and SAD stages-oversight.
   - Read mo-uea's chat template, and check that it loads in vLLM (`qwen3_5` architecture).
3. **W2 gate, about $2 GPU** (once its cost is OK'd). Add the graded / ungraded cue to welfare_training's caving
   harness on Gemma 3 27B, at 7R and 7N.
