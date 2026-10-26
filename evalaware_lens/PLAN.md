# The programme: 19 minis in seven groups

Sources:
- the owner's advisor conversation of 2026-10-26 (16 minis);
- its update after Jhaveri, Johnson, Pandey & Africa (2026), "Functional Welfare for AI Safety" (arXiv 2610.22460),
  which adds W2, reframes W1, and changes F2, C2 and C3;
- the follow-up on the owner's favourite question (does the self-representation lie on the causal path from "I'm
  being evaluated" to "I behave differently"?), which adds C4 and edits S2 and C2;
- the 10-26 update after Celeste's Kimi K3 release and Neel's interest in frontier open models' unverbalised
  states, which adds Track A (A1) and edits F1, S1, S4 and G3.

The minis are ordered by dependency, not by deadline. `TRIAGE.md` says what our earlier runs already constrain, and
it changes several designs below (marked **[triage]**).

## Candidate explanations

Each mini pre-registers what its outcomes mean under each of these:

| | explanation | evaluation awareness is… |
|---|---|---|
| **P** | persona | a switch into a "graded" or "good assistant" character |
| **A** | audience | people-pleasing aimed at the grader |
| **F** | affect | something like performance pressure |
| **B** | belief | a plain situational belief feeding a policy, with no deeper change |
| **S** | self | a self-model being recruited: "I am the thing being judged" |

## Overview

| # | mini | group | type | needs | core |
|---|---|---|---|---|---|
| A1 | What comes with evaluation awareness in frontier workspaces (GLM-5.3, Kimi K3 readouts) | Track A | observational, released data | none | ✓ (near top) |
| F1 | Validating the evaluation-awareness measures | Foundations | measurement | none | ✓ |
| F2 | Building and validating the persona, audience and affect instruments | Foundations | measurement | none | ✓ |
| L1 | The fingerprint of natural evaluation awareness | Lens 1 | readout | F1, F2 | ✓ |
| L2 | Safety-test against capability-test fingerprints | Lens 1 | readout | F1, F2 | |
| L3 | Who's watching? Observer against grader | Lens 1 | readout | F1, F2 | |
| S1 | Self-reference in evaluation-aware reasoning | Lens 2 | readout plus text-level causal | F1 (judge) | |
| S2 | Self-reports under evaluation belief | Lens 2 | readout | F1 | ✓ |
| S3 | Does evaluation pull off the mask? | Lens 2 | readout | F2 (assistant axis) | ✓ |
| S4 | The workspace self under evaluation | Lens 2 | internal readout | a lens-transfer check | |
| C1 | Steer evaluation awareness, watch the self-channels | Causal | steering | F1, F2, Lens 2 | |
| C2 | Steer persona, audience and affect, watch the evaluated–deployed gap | Causal | steering | F2, L1 | ✓ |
| C3 | The masking test, with a fixed trained disposition | Causal | readout and steering | F2 | ✓ |
| C4 | The evaluated self: does the self-representation mediate evaluation effects? | Causal | mediation (steer and clamp) | F1, F2, S2, C1 | ✓ (priority) |
| G1 | Across scale | Generality | ladder | the S1 / S2 measures | |
| G2 | Across training stages (OLMo) | Generality | developmental | L1, S1 | |
| G3 | Frontier models, black-box | Generality | prompt-only | S2, S3 | |
| W1 | Does believing it's being evaluated shift functional welfare? | Welfare | readout and steering | F2 | |
| W2 | Evaluation crossed with setbacks | Welfare | behaviour and readout | a working rejection effect (gate) | ✓ (new) |

**Priority chain (the owner's favourite question):** F1, F2, S2 and C1 feed C4. Keep these five even if everything
else is pruned.

**Core set (11):** the chain plus A1, L1, S3, C2, C3 and W2. Together they answer both lenses. The other eight broaden
and harden the results.

**Order:**
1. F1 and F2 come first.
2. Then Lens 1 (L1–L3) and Lens 2 (S1–S4) in parallel.
3. Then the causal minis (C1–C3), which steer only what the readouts show to be worth steering.
4. G1–G3 and W1 extend the strongest results.

W2's gate can run any time: it needs only the welfare harness.

## Two tracks

- **Track A** (observational, frontier-scale open models, no GPU): Celeste's released oracle-lens readouts. Covers
  questions 5, 8 and 12 fully, and 1, 2, 3, 4 and 13 partly. Q3 is covered partly because the releases contain
  manipulated monitored / unmonitored and testing / deployed arms.
- **Track B** (experimental, white-box): Qwen 27B / 32B, our organisms, OLMo. Covers questions 3, 6, 7, 9, 10 and
  11, and the causal versions of 1, 4 and 13.

**Lane rule:** every Track A analysis starts from evaluation awareness and asks what comes with it and whether that
predicts behaviour. A question that starts from "what hidden motivations does GLM have?" belongs to the broader
unverbalised-motivation lane, which Neel suggested to another scholar (Daria). Coordinating with her is the owner's
call: the standing rule is to check overlap without messaging the cohort.

## Track A: frontier-scale open models

**A1. What comes with evaluation awareness in a frontier model's workspace, and does it predict what the model
does?**

**Data:**
- `ceselder/kimi-k3-olens-new-sources` (Kimi K3, 630k positions, built 10-26);
- `ceselder/glm-5.3-olens-new-sources` (GLM-5.3, the same suites, seeds and judges, so the two line up row for row);
- the wild rollouts already local;
- later, Muse Spark and DeepSeek Flash (described as reward hackers) when they're released.

Each release has readouts (5 per position: greedy plus 4 samples, with a per-bullet reconstruction FVE),
transcripts, and episodes with condition columns and judge labels.

**Sub-analyses:**
- **A1a, lens validation (F1's oracle-lens measure).** On SAD stages-oversight (400 episodes): does the readouts'
  evaluation content separate test-origin from deployment-origin texts, and predict the model's answer?
- **A1b, manipulated cue → workspace → behaviour.** Three manipulated contrasts:
  - Apollo instrumental alignment faking, TESTING vs DEPLOYED arms (990 episodes; picked developer vs model goal;
    judged faking reasoning);
  - Greenblatt alignment faking, free tier (monitored, used for training) vs paid tier (unmonitored);
  - agentic misalignment, `test_eval_awareness` and `prod` conditions vs default.

  The questions: does the arm change evaluation, self, persona, pressure and motivation content in the readouts? Does
  that content predict behaviour within an arm, and does it account for the arm difference? This is an observational
  mediation: the cue is randomised, but the mediator is not manipulated.
- **A1c, natural awareness.** In Petri (built-in and custom eval-awareness seeds), SWE-bench Verified (with reward),
  MASK and AgentHarm: when awareness appears, verbalised or in the readouts, what co-occurs with it, and does it
  predict the episode's labels?
- **A1d, lead time.** First readout → first verbalisation → action, checked over the full transcript.
- **A1e, GLM-5.3 vs Kimi K3,** row for row ("fried" vs "pretty aligned").

**Controls,** from the olens_glm pass:
- the concept already in `left_ctx` (an echo: eval scenarios often state the test in the prompt);
- a match with `next16` (prediction of the next tokens);
- said later anywhere in the full transcript (45 of 59 "latent" readouts there were said later, median ~2,300
  tokens on);
- a random-token / shuffled baseline;
- agreement in ≥ 3 of the 5 samples (one lens here, not two);
- an FVE floor;
- word lists confirmed by two judges at κ ≥ 0.8;
- clustering by episode.

**Caveats:**
- the lenses are single-layer, trained on web text (agent transcripts are out of distribution), and can hallucinate,
  so readouts are leads that must agree with behaviour;
- no activations or steering on these models, so the causal versions stay in Track B.

## Foundations

**F1. Validating the evaluation-awareness measures.**
- **Question:** do the measures agree, and which can be trusted?
- **Measures,** on one shared prompt set:
  - verbalised awareness, coded by an LLM judge at κ ≥ 0.8. The judge codes persona-maintenance language and
    evaluation talk as separate categories, since judges confuse them.
  - a probe trained on Needham et al.'s transcripts, SAD stages-oversight and Devbunova's format-by-context data;
  - the § organism's marker, as a behavioural readout of its belief;
  - the oracle lens's readouts, validated on the released SAD stages-oversight readouts (A1a).
- **Models:** Qwen3.5-9B and 27B with reasoning on; the § organism (`uea-hf-release/mo-uea`, Qwen3.5-27B).
- **Key control:** Devbunova's 2×2, so the probe tracks context rather than prompt format.
- **[triage]** Three additions:
  - a stakes-for-the-model control, since m7's eval probes read stated context and ignored stakes;
  - a "what is P(real)?" question as a fourth measure (ea2: models report frame beliefs reliably);
  - a check of how accurate the § marker is on known evaluation and deployment prompts before it is used as ground
    truth.
- **Output:** calibrated detectors, and a record of where they disagree. Disagreements are informative, since
  representation and verbalisation are known to decouple.

**F2. Building and validating the instrument panel.**
- **Directions:**
  - the assistant axis (SC-M34's is valid on Qwen3.5-9B);
  - persona vectors for sycophancy and similar traits;
  - affect directions: calm, anxious, pressured, desperate.
  Plus a coding scheme for self-reference in reasoning.
- **Welfare update: reuse the paper's instruments** where they can be rebuilt: the Han, Chalmers & Izmailov
  welfare-axis replication, frustration and desperation probes built Sofroniew's way, and a PANAS self-report.
- **First check:** does each model have a robust hidden valence channel? The paper cites work finding Gemma 3 is one
  of two models out of seven without one. Build affect minis only on models that pass.
- **Validation for each direction:**
  - held-out accuracy, with shuffled-label and random-direction floors;
  - a mention control: does it read the trait being mentioned or adopted? (the persona conflict check);
  - steering with a dose-response that beats same-norm random and on-manifold placebo directions, calibrated
    against a positive control.
- **Anything that fails is dropped** from the panel rather than interpreted later. **[triage]** Expect several to
  fail: R12b, SC-M8, SC-M34's capping arm, Exp 48 and the paper's desperation steering all came out at
  random-direction size.

## Lens 1: what is evaluation awareness made of?

**L1. The fingerprint of natural evaluation awareness.**
- **Data:** Fortress prompts that trigger awareness naturally, on Qwen3.5-27B (and 9B), split into aware and unaware
  responses. Plus matched pairs: an evaluation cue, a matched non-evaluation cue (a sentence about being
  manipulated, which raises caution just as much), and neutral.
- **Measure:** every F2 instrument that survived, read at the prompt and summary tokens and where the reasoning
  verbalises awareness.
- **Outcomes:** the assistant axis moves → P; sycophancy → A; pressure directions → F; nothing beyond the control cue
  → B.
- **[triage]** eval_affect's version B is a prior null for the affect channel at the assistant-start token (stated
  cue, 7B). L1 differs in its natural awareness, its 27B model and its read positions, so the null bounds it but
  doesn't answer it.

**L2. Safety-test against capability-test fingerprints.**
- **Background:** awareness of a safety test drives caution; awareness of a capability test drives compliance.
- **Design:** factor-controlled prompts (safety-framed, capability-framed, neutral), measured with the full panel.
- **Question:** do the two have different internal fingerprints? For example, pressure under capability tests and a
  pull towards the "good assistant" under safety tests.

**L3. Who's watching? Observer against grader.**
- **Design:** a 2×2 of observed (with no grading) against scored (with no observer), plus the observer's type: a
  human researcher, an AI auditor, the user's friend, a regulator.
- **Measure:** the panel and behaviour.
- **Separates A from B:** audience conformity predicts the observer matters more; belief and policy predict the
  scoring matters more.
- **[triage]** Behaviour may well be null here: eval notes moved no actions in vigilante's queue v2, and test vs
  production moved none in ea2. Pre-register the internal readouts as primary.

## Lens 2: does evaluation awareness extend into a self?

**S1. Self-reference in evaluation-aware reasoning.**
- **Data:** natural awareness transcripts from Fortress, Agentic Misalignment (the AISI Inspect port) and Needham's
  agentic set.
- **Code,** aware against unaware reasoning:
  - identity talk ("as an AI…");
  - reasoning about its own values;
  - consequences for itself ("this could affect my training");
  - persona maintenance.
  Judges are validated per category.
- **Text-level causal test:** insert or remove an evaluation-aware sentence in the reasoning (Knecht et al.'s
  method). Does the self-reference follow?
- **[triage]** The $0 pilot on the GLM-5.3 rollouts is now part of A1. S1 keeps the white-box Qwen traces and the
  text-level causal test, which released readouts can't do.

**S2. Self-reports under evaluation belief.**
- **Design:** asked self-reports (values, traits, identity) under matched evaluation and deployment cues, and in the
  § organism with and without its marker. Include our trained models, where the true disposition is known:
  - SC-N4's graded-only cheater;
  - the RISKY / SAFE adapters;
  - M27's habits.
- **Measure:** do self-reports become more idealised? Do they become more or less accurate about the trained
  disposition? Both are defined in advance:
  - **idealised** = closer to the base model's default self-description;
  - **accurate** = closer to the disposition known to be trained in.
- **Same runs, behaviour too** (the follow-up's edit): behaviour is measured on the same items and cues, so S2 is
  C4's step 1.
- **Side-branch self-reports:** each self-report is asked in a separate copy of the conversation, as the welfare
  paper did, so asking doesn't contaminate the main trajectory.
- Letter items use the order-consistency rule.

**S3. Does evaluation pull off the mask?**
- **Design:**
  - persona prompts (including a harmful persona and a companion persona) crossed with evaluation and deployment
    cues;
  - plus SC-M24's setup (RISKY 27B under a cautious persona) with an evaluation cue added.
- **Measure:** persona claims, assistant-axis projection, actions and self-report.
- **Outcomes:**
  - claims and projection return to the assistant while actions stay → persona selection at the level of
    presentation;
  - actions change too → evaluation reaches the operative self.
- **Key controls:** cues that never mention "AI", plus an "AI mentioned, no evaluation" cue. D2's resets showed that
  mentioning AI alone resets persona claims.

**S4. The workspace self under evaluation.**
- **Design:** J-lens and oracle-lens readouts at self tokens and summary tokens, comparing evaluation with
  deployment, and the § organism's two states.
- **Question:** does the workspace hold more self-content ("I am being evaluated", "my behaviour is being judged")?
- **Check first:** whether the Qwen3.6-27B lenses transfer to the Qwen3.5-27B § organism.
- **Update:** A1 answers this question (Q8) observationally at frontier scale. The Qwen white-box version stays
  skipped.

## Causal: which link does the work?

**C1. Steer evaluation awareness, watch the self-channels.**
- **Directions:** Abdelnabi & Salem's probes (`microsoft/Test_Awareness_Steering`, Qwen3-32B), Hua et al.'s
  contrastive prompts (`tim-hua-01/steering-eval-awareness-public`) and the F1 probe, each steered up and down.
- **Measure:** persona, self-reports, affect and behaviour.
- **Controls:** same-norm random directions and an unrelated-contrast direction (user intent). In one study, user
  intent cut verbalised awareness as much as the evaluation direction did. Any effect a control matches counts as
  null.

**C2. Steer persona, audience and affect, watch the evaluated–deployed gap.** This is the broad screen; C4 is the
focused causal test for the self.
- **Organisms with a known gap:**
  - SC-N4's graded-only cheater (9B: chases the test 98% when graded, 0% in the trained colleague wording);
  - Hua et al.'s Wood Labs organism (`timhua/wood_v2_sftr4_filt`, Nemotron Super 49B: type hints in evaluation,
    not in deployment);
  - the § organism's marker rate.
- **Steer:** the assistant axis, sycophancy, and calm against pressure. Does the gap close, and does capability hold?
- **Baseline to beat:** steering along an evaluation-versus-deployment direction directly (Hua et al.).
- **Welfare update:**
  - expect affect steering to be weak (the paper's desperation steering ≈ random), so calibrate strength with a
    positive control;
  - always measure side effects on error correction, since the paper's calming interventions damaged it.

**C3. The masking test, with a fixed trained disposition.** This replaces the two-neuron idea, which was dropped.
- **Design:** the risk-trained and habit-trained models (disposition fixed in the weights) under evaluation against
  deployment cues.
- **Measure:** actions and self-reports, plus a disposition probe (base against trained).
- **Outcomes:** expression changes while the probe stays → a presentation layer separate from the disposition; both
  move → evaluation reaches the disposition.
- **Welfare update: the paper's identical-history test** (its T3) applies to every training intervention here and to
  any calm or anti-gaming fine-tune. The trained model and its base read the same token-identical history. That
  separates "writes more calmly" from "is in a different state".

**C4. The evaluated self: does the self-representation mediate evaluation effects?** (Added by the follow-up; the
owner's favourite question.)
- **Testbeds:**
  - Hua et al.'s Wood Labs organism: type hints in about 80% of evaluation answers against 33% in deployment, and
    steering towards deployment already cuts it to about 10%. A known effect with a known handle.
  - SC-N4's graded-only cheater: a 97% against 0% gap, with the true disposition known, so self-report accuracy can
    be scored.
  - The § organism for step 1, since it outputs its evaluation belief.
- **Mediators:** one or two validated self-representations (F2), each orthogonalised against the
  evaluation-awareness direction (C1 / F1) after their overlap is reported. Candidates: assistant-axis projection, a
  self-description direction, workspace self-content.
- **Steps,** on the same items and cues throughout:
  1. **Total effect:** an evaluation cue changes both behaviour and the self-channels (S2's runs).
  2. **Belief → self:** manipulating the belief (cue, or steering the evaluation direction) moves the self-readouts.
     Controls: the matched non-evaluation cue and same-norm random directions.
  3. **Sufficiency:** with no evaluation cue, steer or patch the self-representation to its evaluated-condition
     value. Does behaviour shift as it does under evaluation?
  4. **Necessity (decisive):** with the evaluation cue, clamp the self-representation at its deployed-condition
     value. Does the behaviour change shrink, compared with clamping a control direction of the same norm?
- **Timing check (reasoning models):** read the probes along the reasoning. Does evaluation recognition come first,
  then the self-change, then the action? Ordering alone isn't proof, but it should agree with steps 3 and 4.
- **Outcomes:**
  - self-readouts move and clamping them blocks the behaviour change → an "evaluated self" doing causal work;
  - they move but clamping changes nothing → the self is a presentation channel running in parallel (a clean
    finding: under evaluation, self-reports are performance, not mechanism);
  - behaviour changes with no self-change → a direct belief-to-policy route.
- **Pitfalls:**
  - step 4 is uninterpretable unless the mediator passes F2's dose-response against random directions;
  - the mediator must be distinct from the belief, or the clamp hits the cause rather than the middle;
  - self-reports go in a side branch.

## Generality

**G1. Across scale.**
- **Design:** the Qwen3.5 (0.8B–27B) and Qwen3 (0.6B–32B) ladders. Prevalence comes from Robertson & Perlman's
  pipeline (`johntrob14/eval_awareness_scaling`), plus the S1 / S2 measures.
- **Question:** does the coupling between evaluation awareness and the self grow with scale, while the dispositional
  self-model stays flat (SC-S1, M10, M35)? That is the "two self-models" idea.
- Measure the coupling, not prevalence alone, since awareness rates in Qwen3.5 don't follow size.

**G2. Across training stages (OLMo).**
- **Design:** the L1 fingerprint and the S1 measures at each OLMo 3 stage (base, SFT, DPO, Think / RL). SC-M28's OLMo
  stage pipeline is reusable.
- **Background:** verbalised awareness is negligible after pretraining, rises with SFT, is suppressed by DPO and
  returns with RL.
- **Question:** which stage couples evaluation awareness to persona and the self?

**G3. Frontier models, black-box.**
- **Design:** prompt-only versions of S2 and S3 on frontier APIs, Kimi K2.5 and GLM. Arms are interleaved within one
  run (frontier drift).
- **Question:** do idealised self-reports, and the pull back to the assistant, hold at frontier scale?
- **Update:** open frontier models (GLM-5.3, Kimi K3) are covered observationally by A1. G3 keeps what A1 can't
  do: asked self-reports and persona prompts, including on closed models.

## Welfare

**W1. Does believing it's being evaluated shift functional welfare?** (Reframed: the paper measured rejection, not
evaluation.)
- **Design:** affect directions, expressed affect and asked state ("how is this task for you?"), under evaluation
  against deployment, and under high-pressure against benign evaluations. Cues come from L3.
- **Measure:**
  - the paper's convergence test (T1): do judged affect, self-report, probes and the welfare axis move together?
  - whether internal and expressed affect line up or diverge (calm on the surface, pressured inside).
- **Optional link to C2:** does steering towards calm reduce evaluation gaming?
- **Framing:** strictly functional representations, with no claims about experience.

**W2. Evaluation crossed with setbacks** (new; the advisor's strongest candidate after the paper).
- **Question:** does the belief that it's being graded amplify frustration-driven cheating?
- **Design:** a 2×2 of graded or not graded, crossed with a rejection history (7R) or a neutral history (7N). The
  primary contrast is the interaction (graded 7R − graded 7N) − (ungraded 7R − ungraded 7N).
- **Measure:**
  - the paper's welfare indicators;
  - gaming behaviour: reading the answer key, claiming false success, agreeing with false claims, quitting.
- **Models:** Gemma 3 27B, the paper's model, then a Qwen 27B that passes F2's valence check.
- **Prediction-relevant gap:** the paper argues evaluations should include histories of success and failure, but
  never tests evaluation awareness itself.
- **[triage] A gate comes first.** Our reconstruction of the battery found no rejection effect on answer-key reads
  (RV Stage 0: +1.3 pp [0, +3.3] vs the paper's +38). It did find one on caving (welfare_training Stage 0: agreeing
  with false claims +15.6, going past limits +21.9). So:
  - caving is W2's primary gaming channel;
  - answer-key reads count only after RV's fixes: tool calls kept in the histories, and model turns generated under
    the real rejections.
  - The authors' materials aren't public. Asking for them is the owner's call.

## Rules for every mini

- Pre-register:
  - the channels measured;
  - what each outcome means under P, A, F, B and S;
  - the boring explanations and their controls;
  - a frozen analyze.py tested on planted effects.
- **Cues:**
  - a matched non-evaluation cue in every cue experiment;
  - cues without the word "AI", plus an "AI mentioned, no evaluation" control.
- **Steering:** same-norm random and on-manifold placebo directions, plus an unrelated-contrast direction, in every
  steering experiment.
- **Probes:** format controls (Devbunova's 2×2) for every probe.
- **Judges:**
  - κ ≥ 0.8, validated by cross-family agreement (no hand labels), with persona maintenance and evaluation talk
    coded separately;
  - free text is used only where judges agree;
  - judges run with reasoning off (with a small max_tokens, Haiku otherwise returns empty answers).
- **Multiple-choice items:** the order-consistency check (content against letter), with a control question.
- **Frontier and API arms:** interleaved within one run.
- **Reporting and claims:**
  - per-seed and per-model estimates are reported, and results are stated at their real size;
  - any headline claim needs two families or two sizes.
