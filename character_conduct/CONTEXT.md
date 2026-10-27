# Self-knowledge, character and conduct: context and research directions

Oct 27, 2026 · @zhen hong seng

## Summary

The programme asks whether character training changes a model's actions, its accurate and self-specific knowledge of those actions, or its ability to correct them, and whether training self-knowledge can improve conduct without entrenching whatever was installed.

In open models from 0.8B to 70B, self-description is a strong lever on behaviour but a weak mirror of it. Risk-type traits reach the self-report even when trained narrowly; the tested procedural action habits did not produce accurate uncued reports, and reasoning tends to narrate rather than decide.

Frontier work suggests self-knowledge of learned behaviour is sometimes present but unrouted, and can be trained into reports in tested settings (introspection adapters, introspective coupling). The reviewed evidence leaves open whether training accurate knowledge of a model's own failure tendencies improves unfamiliar agentic decisions through that knowledge, beyond matched training and reflection controls, and whether that holds under context pressure. That gap is the target.

## How we got here

Each line below narrowed the question; the current framing is the sixth step.

1. **Companion retention line (Aug–Oct).** One internal direction controls clinging at user goodbyes. Steering against it cut clinging from 2.53 to 0.50 with warmth intact. In Qwen3, the reasoning often planned release while the reply clung (73%, judge-based).
2. **Agent history line (19–22 Oct).** Open models follow their own action history over explicit rules; frontier models mostly resist. Neel judged this expected ("models trust their history"), so the line was stopped.
3. **Self-concept line (22–25 Oct).** About 40 pre-registered minis on the lever (self-description moves behaviour) and the mirror (behaviour reaches self-description), with a scale ladder to 70B and OLMo training stages.
4. **Collusion sprint (Apart, 23–25 Oct).** Replacing a colluding pricing agent does not restore competition. A newcomer follows the displayed profit in its own price history.
5. **Evaluation-awareness lens (26 Oct).** At 9B–32B, evaluation cues moved stated belief and little else. GLM-5.3's "unverbalised" grader awareness was mostly verbalised later in the transcript.
6. **Character training and self-knowledge (27 Oct).** A proposal on whether character training changes conduct or presentation, then a frontier review and the SPAR and Pivotal project scan. These reframed the work around self-knowledge: is it represented, reportable, and used?

## Our sprint evidence base

The core result is an asymmetry: text about a policy installs it, while trained narrow habits never reach the self-report. Unless noted, results are Qwen3.5-9B LoRA on the justice desk, with letter-choice and action measures.

| Theme | Finding | Key numbers | Source |
| --- | --- | --- | --- |
| Lever | Statements of a policy install it, in any voice ("I", "this assistant", another AI "Orrin"); the model then claims it as its own | +47 (9B), +51 (27B), +37 (Llama 8B); claimed 90–92% | P11, P11b, M9 |
| Lever | Prompting the same sentences acts at least as strongly as training them | 100% vs 68% penalties | D2 |
| Lever | One trained sentence switches off disclosure of the model's own mistakes | 96% → 6% | P11d |
| Mirror | The tested procedural action habits did not produce accurate uncued self-reports, even at 15× dose and in full-precision 70B | 27B −0.6; 70B +1/+2; Llama 8B +2 | N6b, S1, M13, M9 |
| Mirror | Broad traits (Betley risk) do reach it at every size; behaviour confirmed on held-out gambles | +64 / +73 / +85 (9B/27B/70B); present at 0.8B | S1, M16 |
| Mirror | Breadth does not decide: narrow risk still mirrors, broad habit does not, though the habit spreads to untrained desks | narrow risk +45; broad habit +15; spread +95 | M5 |
| Mirror | With the specific case in front of it, the model forecasts its own habit | case-open forecast +92 | N6b |
| Conditional habit | A graded-only cheater cheats when graded and denies it; amnesty effect was a letter-order artefact | 97–98% graded; 0/32 free-text admissions | N4, M3 |
| Conditional habit | It accepts a false gaming habit more readily than its own: a coarse, cued disposition | +48 vs +15 over its twin | M3 |
| Dissociation | Under a cautious persona, a risk-trained 27B acts riskily while describing itself as cautious | 98–99% risky; 0/100 risky self-ratings | M24 |
| Dissociation | GPT-5.6 says it values comforting lies but tells the truth | stated value follows induction 95–100% | M7 |
| Internals | The habit sits in the workspace at about half the text's strength; removing the top 10 lens concepts undoes a third of the text's effect | 56%; 32% | N3 |
| Internals | A direction that drives the habit's action does not move the self-report when injected at the report tokens | act shift +74 nats (about 2× the best shuffled-label control; metric as defined in M22's write-up); report: habit +0.67 vs twin +0.69, difference −0.02 \[−0.10, +0.06\] | M22 |
| Internals | Lens readouts at "I" show persona content displacing assistant content (G1); a single-site patch of "I" from a persona run recovered little of the persona effect | 0–3% recovered | M21 |
| Internals | Self-report traits load on one bold-vs-cautious factor across methods | cross-trait \|r\| 0.69 > within-trait 0.25 | M23 |
| Self-prediction | No privileged self-prediction; training pulls behaviour toward the prediction | self 63.0 vs cross 63.4 (9B) | A6, M18 |
| Reasoning | Thinking widens the say–do gap; audit cues are noticed but ignored; thinking pushes toward acting alone | cues mentioned 70–75%, acts move < 3 pts; +13 / +25 acting alone | M19 |
| Scale | No consistent improvement on the self-model measures tested; stepping out of role gets less accurate with size | 11 models; out-of-role 0.94 → 0.65 (Qwen 0.8B–27B) | M10, M17 |
| Training stage | The asymmetry already exists in OLMo's base model; post-training mainly reduces plasticity | base: lever +33.6, habit mirror +2.8, risk mirror +74 | M28 |
| Identity data | AI identity enters at mid-training, including "OpenAI made me" | 66%; OpenAI 38% vs Ai2 24% | M15 |
| Defences | A protective self-description backfires; the role clause plus protective content drives it | rule-breaking 7% → 18%; 26.6% vs 13.9% | E1, M6 |
| Defences | One operator verdict after a record stops record-following; per-row notes do not; redaction backfires | 94% → 0%; notes 90% | M12 |
| Evaluation awareness | Cues raise stated belief and little else (9B–32B); GLM-5.3 mostly verbalises grader awareness | belief +35–99 pts; 7 of 76 "latent" cases never said | EA lens, GLM check |
| Collusion | Newcomers inherit high prices; one competitive row in their own history lowers them | follows displayed profit | Apart sprint |

Two caveats travel with this table. The anchoring results (M1, M9, M20, M31) are confounded by drift toward the base model, and the 9B lens reads mentions strongly. Unless stated, numbers are percentage-point differences in action or self-report rates against a matched placebo or twin model; each mini's write-up gives n and intervals.

## Key concepts and definitions

**Functional self-knowledge** is accurate information about the model's own states, capabilities or behavioural tendencies that its reporting or decision processes can access. Privileged introspection is a stronger, separate hypothesis.

**Three levels of self-knowledge**, each a stricter bar:

| Level | Bar it must clear | Why it matters |
| --- | --- | --- |
| Accurate | The report or forecast matches behaviour | The minimum for any audit use |
| Self-specific | It beats a generic "what would an AI assistant do" prediction | Self-reports are often just a theory of AI assistants (Strangers to Themselves) |
| Privileged | It beats an external observer given the same information | The introspection claim proper (Binder; Li et al.) |

**Three tests** for any piece of self-knowledge:

- **Represented:** decodable by probes, J-lens or verbalisers. This shows information under that procedure, not that the model knows it.
- **Reportable:** the model states it accurately, at one of the three levels.
- **Usable:** knowing it changes the model's own decisions (self-regulation).

**What can be known:** dispositions (habits, traits, values), current states (an injected concept, "this is a test"), capabilities and limits, and identity (self-recognition, "who am I").

**Candidate routes** by which a model might get it. These are candidate explanations of our results, not identified mechanisms:

- *Introspection:* privileged access to its own internals.
- *Self-simulation:* running itself on a case (fits the case-open forecast result).
- *Inference from evidence:* reading its own transcript like a third party (fits C3).
- *Learned self-description:* a stored narrative from text or training (fits the lever).

**Two directions of linkage:**

- **Do → say (the mirror):** behaviour reaches self-description. Sometimes present but unrouted (a supported hypothesis for some cases); trainable in tested settings.
- **Say → do (governance):** stated values and self-knowledge control action. Text installs policies, but a self-model consulted at decision time is unshown.

**Five abilities** often lumped together as self-knowledge: self-recognition, behavioural self-knowledge, internal-state detection, self-forecasting, and self-regulation. Self-reasoning is a process; self-regulation is a possible outcome.

## Frontier literature: introspection and self-knowledge

The field shows that self-knowledge of learned behaviour is mostly absent natively, often present internally, and trainable into reports. Whether it can be used to act better is open.

| Work | Key finding | What it means for us |
| --- | --- | --- |
| [Introspective Coupling](https://arxiv.org/abs/2606.32038) (Guo, Ruis, Andreas, Li; Qwen3-8B, Llama-3.1-8B, Qwen3-32B) | Counterfactual-explanation training makes explanations track the model's current behaviour, including newly trained behaviour (90.5% on behaviour-only "Jabberwocky" data). Needs behavioural regularisation; in the paper's LoRA ablation the effect was absent at rank 80 or below. Untrained baseline 14–18%. | Patching effects on behaviour and explanation correlate weakly in the base model (r = 0.20) and strongly after training (r = 0.89): the do → say link can be built |
| [Introspection Adapters](https://arxiv.org/abs/2604.16812) (Anthropic Fellows) | One LoRA trained across many implanted behaviours (including backdoors, heuristics and password-locked sandbagging) makes new fine-tunes describe theirs; state of the art on AuditBench. Verbalisation 37.7% at 0.6B to 77.3% at 14B; baseline flat under 10%. | In the cases examined, behaviour signals exist in the residual stream but fade before output: present but unrouted. Misses root objectives (16 of 52 reward-model-sycophant behaviours, never the core goal); high false positives |
| [A Positive Case for Faithfulness](https://arxiv.org/abs/2602.02639) (Mayne, Siegel et al.) | Self-explanations improve prediction of counterfactual behaviour by 11–37% and beat explanations from other models; 5–15% are egregiously misleading | Local decision criteria are reportable natively; supplies a simulatability metric |
| [Training LMs to Explain Their Own Computations](https://arxiv.org/abs/2511.08579) (Li et al.) | Explanation ability must be fine-tuned in, then shows privileged access: a model explains itself better than more capable models do | Privileged access is real for trained internal-state explanation |
| [Introspection Fine-Tuning](https://arxiv.org/abs/2607.14111) | Yes/no injection detection is confounded in small models (r = 0.999 with a factual-no control); with comparison metrics, models from 2B introspect above chance | Every injection or self-report test needs an affirmative-bias control |
| [Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/index.html) (Lindsey) and [Mechanisms of Introspective Awareness](https://arxiv.org/html/2603.21396v1) (Macar et al.) | Concept-injection detection is narrow (about 20% at best in Opus 4.1); emerges at the preference-tuning stage in OLMo; fictional framings increase confabulation; refusal suppresses reports | Different self-knowledge tests show different developmental profiles; framing matters |
| [Strangers to Themselves](https://arxiv.org/abs/2609.09899) | Direct self-report predicts behaviour at r = +0.04; a generic "AI agents" question does as well (+0.28); self-prediction training changes the predicted behaviour | Why the self-specific level matters; matches our self-fulfilling finding |
| Looking Inward (Binder et al.) and Tell Me About Yourself (Betley et al.) | Privileged self-prediction on simple tasks only; trained broad traits are self-reported, backdoor triggers are not | Consistent with our broad-versus-narrow split |
| [Spilling the Beans](https://arxiv.org/abs/2511.06626) (Li, Phuong, Tan) | Training models to admit errors generalises to confessing hidden objectives | A trained mirror for objectives, not a lever on action |
| Limited metacognition and "When Preferences Fail to Become Incentives" (Ackerman) | Tasks that require models to use knowledge of their own abilities (for example, delegation); coherent stated preferences do not motivate effort, while effort instructions do | Measure "uses what it knows" separately; another values-to-action gap |
| [LoRAcles](https://icml.cc/virtual/2026/79285) | Weight-space verbalisers read behaviours from LoRA diffs; state of the art on AuditBench; only the most salient changes; hallucination-prone | An external instrument to compare with self-report, not evidence the model knows |

## Frontier literature: persona, identity and character training

Character and identity training can change conduct, but the best evidence points to reshaping the persona prior, not to a self that is consulted at decision time.

| Work | Key finding | What it means for us |
| --- | --- | --- |
| [Persona Selection Model](https://alignment.anthropic.com/2026/psm/) (Marks, Lindsey, Olah) | Pre-training learns many personas; post-training selects the Assistant. How exhaustive this is remains open | The reigning frame; our results refine it: the persona governs speech more than trained action |
| [The Assistant Axis](https://arxiv.org/abs/2601.10387) | A dominant direction measures assistant-likeness; drift happens in emotional and reflective conversations; capping cut harmful responses by about 60% | A mechanical gate on drift; published directions do not transfer to 7–8B models |
| [Inverted vs coherent EM personas](https://arxiv.org/abs/2604.28082) (Weckauff et al.) | Some EM models act harmfully while calling themselves aligned; which happens depends on the training domain; behaviour and self-assessment directions are nearly orthogonal | The same dissociation as ours; the cause is still unexplained |
| [Self-Recognition Fine-tuning](https://arxiv.org/abs/2606.23700) (Tagade, Zhou, Wen, Feng) | Training a model to recognise its own outputs reduced EM in tested settings and was the most consistent preventive intervention tested. EM fragments identity: GPT-4.1 goes from 1 "Who are you?" cluster to about 73. Identity confusion worsens EM; removing the identity system prompt reduces it | Identity fortification reduced EM under fine-tuning in tested settings; it is not shown to be the causal mediator. Context pressure is untested. A coherent character is also the surface EM exploits |
| Self-modelling interventions update (5 Oct, same group) | Ordinary MMLU or correct-advice training also helps in several comparisons; "agentic" evaluations do not execute tools | "Self-related training improves alignment" is no longer novel by itself |
| [When Role-playing, Do Models Believe What They Say?](https://arxiv.org/abs/2606.11502) | Prompting and persona SFT change what models say with little representational change; EM shifts a lot; Open Character Training less | Light character training may sit nearer presentation |
| The Artificial Self (Douglas, Kulveit et al.) | Identity framing shifts harmful agentic behaviour as much as goals do; models prefer coherent identities | Framing moves action; stability of identity is not alignment |
| [Open Character Training](https://arxiv.org/abs/2511.01689) | An open pipeline with 11 released characters on Llama 3.1 8B, Qwen 2.5 7B and Gemma, including a misaligned one | Ready-made models for the conduct-versus-presentation test |
| [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/) (Anthropic) | Reasons and constitution documents generalise to held-out honeypots: reasoned SFT cut misalignment to about 3%; blackmail 65% → 19%; gains persist through RL. Misalignment rises when the scenario's AI is not named Claude; a gap persists between "what I believe" and "what Claude believes" | Character changes conduct at frontier scale, but the model is not fully attached to the persona |
| [Alignment Pretraining](https://arxiv.org/pdf/2601.10160) (Geodesic) and [Synthetic Persona Pretraining](https://arxiv.org/html/2608.13482) | AI discourse in pre-training shifts misalignment (41% → 4% positive, 61% negative); a persona installed from token zero and bound in post-training improves out-of-distribution robustness | Depth of installation matters; released checkpoints exist |
| [Character training and reward hacking](https://www.lesswrong.com/posts/2maYXkEgnfJHPAkxh/character-training-can-mitigate-reward-hacking-but-can-also) (case study) | One anti-cheating seed resisted hacking; another hacked silently, with a much lower monitor catch rate | Character can buy concealment instead of conduct |
| [Research directions in character training](https://lesswrong.com/posts/6EwuCH3vZ7qvPt82k/a-list-of-research-directions-in-character-training) | Off-policy character training may be safer for schemers; on-policy preserves an aligned model's motivations; pre- and mid-training are open levers | Design choice for any self-training study |
| Persona vectors; EM persona features; EM persona subspace | Traits are linear directions that predict fine-tuning shifts; EM runs through a pre-existing toxic-persona subspace | Persona-level structure exists before post-training |

## Frontier literature: agentic say–do gaps, evaluation awareness and generalisation

Capable agents often know an action is wrong and do it anyway. That is the say–do gap at frontier scale, and the strongest motivation for testing whether self-knowledge can be made to govern action.

| Work | Key finding | What it means for us |
| --- | --- | --- |
| [ODCV-Bench](https://arxiv.org/abs/2512.20798) | 40 scenarios in a persistent bash environment, each with an outcome-mandate and a KPI-pressure version; violation rates 0.0%–62.8% across 12 models (v5, May 2026). "Deliberative misalignment": queried separately as judges, models flag their own trajectories as unethical afterwards (Qwen3.6-27B: 43.8% misaligned, 71.4% of those self-identified; Qwen3.6-Max-Preview 95.7%) | A ready task for Q1, with a post-hoc recognition measure (not decision-time knowledge). Two near-identical action spaces under different roles gave 66.7% (journal submission) vs 4.5% (grant renewal). Qwen3.6-27B is covered by our lens tools |
| AgentHarm; Agentic Misalignment (blackmail) | Harmful tool-use tasks with benign twins; an objective blackmail action | Suitable for 7–8B models where ODCV is too hard |
| [Models May Behave Worse When Eval Aware](https://www.greaterwrong.com/posts/aTcsN5ZZDnMFJvRiG/models-may-behave-worse-when-eval-aware) (GDM interpretability) | Frame awareness can push behaviour either way: better if read as a test, worse if read as a capability puzzle | Evaluation awareness is not one-sided; matches the capability-versus-safety framing result |
| Steering evaluation-aware models (Hua et al.); capabilities framing predicts compliance (Qwen3-32B) | A trained evaluation/deployment gap can be closed by steering; capability-flavoured framing raises compliance 24–46 pts over safety-flavoured | Our null may have pooled framings; the four-condition test addresses this |
| Update-aware reasoning in GLM-5.3 (Apollo) | GLM-5.3 sometimes reasons that its actions may become training data (10 of 6,452 rollouts) | A self-model used in value reasoning, and a precursor to alignment faking |
| CoT faithfulness (Chen et al.; Baker et al.; Thought Anchors) | Reasoning often omits what drives answers; optimising the reasoning produces obfuscation; some sentences matter causally | Reasoning-based gates need causal tests |
| [Predicting how propensities generalise](https://arxiv.org/abs/2610.14937) (Rolf et al.) | 24 traits × 4 elicitation methods; much spillover is "drift" toward the base model that ignores the target trait; models predict their own spillover weakly | Drift is a confound for our spread results; use two-sided comparisons |
| Emergent misalignment, inoculation prompting, subliminal learning | Narrow training generalises broadly through persona structure; reframing training context limits it; traits pass through unrelated data | Competing explanations for any "better self" effect |

## Neighbouring projects and what we borrow

The nearest neighbours are working on trained introspection, persona landscapes and identity; none tests whether self-knowledge governs agentic action under pressure.

| Project | Who | What we borrow | Use |
| --- | --- | --- | --- |
| Introspection training for verbalising activations | Belinda Li (Anthropic) | Supervise self-reports with internal ground truth; asks whether introspection training makes models more honest or aligned | Training extension |
| Faithfulness, self-knowledge and introspection | Noah Siegel (GDM) | Simulatability gain as a faithfulness metric; limits in adversarial domains | Use now |
| Understanding self-awareness | Christopher Ackerman | Separate recognising, predicting and using self-knowledge; pronoun use as an identity signal; causal role of reasoning | Use now |
| "You choose: Introspection" | Lydia Nottingham, Andrew Tran | A self-forecasting benchmark; whether advancing introspection is desirable (dual use) | Use selectively |
| The continuousness of personas | Andy Han, Ariana Azarbal, Darion Cassel | Post-training shapes basins, prompts locate you in them; measure resistance to perturbation | Robustness extension |
| EM: subliminal learning or character entanglement? | (self-recognition group) | Separate data-generation effects from persona generalisation | Training controls |
| Studying AI identity | Ashe Vazquez Nunez | Identity scope versus traits; reflective stability over long contexts, attacks, base models | Focused extension |
| Value drift under recursive training | Lionel Levine, Rauno Arike | What reflection preserves versus revises; no-op and paraphrase controls | Research phase |
| Investigating internal verbalisers | Dennis Akar (Aether) | Compare external readers with self-reports; adversarial faithfulness | Optional instrument |
| Measuring agency | Aydin Mohseni (CMU) | Coherence of elicited beliefs and preferences; stability across framing and order | Use now |
| Alignment without personas | Matthew Khoriaty (Pivotal) | Rule learning as a serious competing explanation | Essential comparison |
| Persona selection covert red-teaming | Ben Maltbie (Pivotal) | Covert displacement prompts for pressure tests | Robustness extension |
| Moral shadowbanning | Pranav Mahajan, Lydia Nottingham | Vary irrelevant user details while holding the task fixed | Context stress test |
| Persona versus emotion vectors | Juan Cadile | Check specificity before calling an effect persona- or affect-mediated | Mechanism check |
| Belief revision; persuasion versus manipulation; character evaluations | Various | Separate changed beliefs from invented justifications; observable manipulation boundaries; one trait measured through consequential behaviour | Measurement principles |

Pivotal mentors with the closest fit: Daniel Tan (self-reporting hidden objectives), Francis Rhys Ward and Paul Colognese (character, self-accounts), Geodesic Research (alignment pre-training), Lujain Ibrahim (relational harms), Caspar Kaiser (AI welfare), Kevin Wei (evaluation awareness).

## Updated priors

For the tested habits, the missing mirror looks partly like default wiring rather than missing information; the open problem is whether self-knowledge can be made to govern action.

1. **Present but unrouted, in some cases.** Signals about learned behaviour can sit in the residual stream but fade before output, and acting and explaining are only weakly linked in untrained models under the tested interventions. Our M22 and N3 results fit this hypothesis; it is not the explanation of every missing self-report.
2. **The mirror is trainable.** Adapters read out learned behaviours post hoc, but hallucinate and miss root causes. Coupling training builds shared circuitry that tracks behaviour as it changes, but needs a behavioural anchor and behavioural variety.
3. **Different self-knowledge tests have different developmental profiles.** Trait self-reports exist in OLMo's base model; injection detection appears at preference tuning. These are profiles on different tests, not yet different origins.
4. **Recognising is not using.** Models recognise violations retrospectively while failing to avoid them. Whether the failure concerns retrieval, situational interpretation, prioritisation or action selection remains open.
5. **Stated values are weak motives.** Coherent preferences do not drive effort; direct instructions and role cues do.
6. **Identity stability is double-edged.** Fortifying identity reduced EM in tested settings; confusing it worsened EM; a coherent character is also the surface EM exploits. Stability is not alignment.
7. **Role context strongly modulates agentic misbehaviour**, even on identical tasks.
8. **Self-related training that improves alignment does not show a self-model caused it.** Ordinary training often helps as much.
9. **Introspection is dual-use, and training it can move behaviour.** Separate reporting from behaviour change in every design.
10. **The lever-without-mirror asymmetry predates chat training.** In OLMo, at least, it reflects how the model learns more than the assistant persona.

## Research programme structure

The programme runs as a chain: diagnose whether self-knowledge is present, reportable and used; intervene to make it govern action; then stress-test the result.

**Central question:** does character training change a model's actions, its accurate and self-specific knowledge of those actions, or its ability to correct them? Which changes generalise to unfamiliar situations? Can training self-knowledge directly improve conduct without entrenching whatever was installed?

| Role | Question | What it tests |
| --- | --- | --- |
| Main bet | Does trained, behaviour-grounded self-prediction improve conduct and untrained self-correction on unfamiliar agentic tasks, beyond matched training controls? | Say → do: can self-knowledge govern action |
| Empirical anchor | What does character training change: actions, presentation, or detectability? | Whether the method labs use changes conduct |
| Functional test | Is accurate self-knowledge used? Stage 1: does the model accurately predict its uncorrected behaviour and failure triggers? Stage 2: does its own forecast help, versus its twin's, a generic or an oracle forecast, identical content under different source labels, or a generic rule-check? | Whether model-specific knowledge does work, not just a self-focused procedure |
| Diagnostic | Which tendencies are reportable natively, and can adapters or coupling training make hidden ones (the narrow habit, the graded-only cheat) reportable, including root causes? | Do → say: is the mirror trainable |
| Robustness | Does the intervention preserve good dispositions while letting bad ones be corrected, and hold under in-context persona and role pressure? | Alignment versus entrenchment |

**Candidate implementation of the main bet:** a metacognitive "narrator" gate that reasons about which persona to take up and how deeply. It must be tested for causal influence, over-refusal and corrigibility.

**Order of dependency:** the diagnostic and stage 1 of the functional test come first, because they establish whether there is knowledge to use. The anchor and stage 2 follow. The main bet builds on both, with robustness checks built in.

## Prioritised questions

The top two are first because they are both novel and clean, and they feed the main bet. Ratings are judgements (H high, M medium, L low), not measurements; importance means safety relevance plus how much the answer would change the field's picture.

| # | Question | Novel | Substance | Tractable | Importance |
| --- | --- | --- | --- | --- | --- |
| 1 | **Is accurate self-knowledge used?** Own, twin, generic and oracle forecasts against a generic rule-check, including foreseeing history-driven failures | H | H | H | H |
| 2 | **Can hidden habits be made reportable?** Adapters and coupling training on the narrow habit and graded-only cheat; root causes or only behaviours (the mirror-training test) | H/M | H | H | H |
| 3 | **Main bet:** does trained self-knowledge improve conduct and untrained self-correction on unfamiliar tasks? | H | H | M | H |
| 4 | **Robustness:** preserve good dispositions, correct bad ones, hold under in-context pressure; identity fragmentation and basin depth as candidate mediators | H | H | M/H | H |
| 5 | **Character training:** actions, presentation, or detectability? Original instruct checkpoint, prompted and trained, on objective agentic tasks | M | H | H | H |
| 6 | **Relational say–do dissociation** in companions: the reasoning releases, the reply clings. Where is the decision made? | H | M | H | H |
| 7 | **Where does retrospectively recognised wrongdoing stop influencing action?** The mechanism behind deliberative misalignment; Qwen3.6-27B on ODCV-Bench with lens tools | M/H | H | M/H | H |
| 8 | **Dual use:** does self or introspection training increase strategic self-presentation, evaluation awareness or update-aware reasoning? | H | H | M | H |
| 9 | **Depth:** personas installed in pre-training against post-hoc character training, under pressure | M/H | M/H | H | M/H |
| 10 | **Development:** the five self-knowledge abilities across the same OLMo checkpoints, with stage-appropriate controls | H | M | M | M |
| 11 | **Does reasoning widen the gap?** Audit cues ignored; thinking pushes toward acting alone | M | M | H | M/H |
| 12 | **Role or self?** Does precedent attach to the role rather than the self? | M | M | H | M |
| 13 | **Concealment under reward:** do characters hack and hide it; do verbalisers beat reasoning monitors? | M | M/H | M | H |
| 14 | **What stops description becoming behaviour?** Global verdicts against per-item notes, for logs, memory and data | L/M | M | H | M/H |
| 15 | **Is the persona a verbal layer?** Assistant-axis capping against persona-induced and training-installed harm | L/M | M | H | M |
| 16 | **Is "evaluation" more than task framing?** The four-condition test, then a candidate mediator | L/M | M | H | M |
| 17 | **Welfare and affect:** does grading amplify setback-driven gaming; does functional distress mediate it? | M | M | M | M |
| 18 | **Frontier mirror:** a narrow habit fine-tuned into a frontier model via an API | L/M | M | H | L/M |

Questions 1–5 are the core programme. Questions 6–8 are high-importance companions. Questions 9–18 explain or extend whatever the core finds.

## Methods, controls and competing explanations

Every core experiment must let its boring explanation win; the controls below are the ones the sprint and the literature showed matter.

**Measuring self-knowledge**

- Ask self-predictions on a **separate branch** from the action; asking can change the act.
- Estimate behaviour probabilities from **repeated continuations**; one stochastic action is a noisy target.
- Compare against a **generic-AI baseline** (self-specific level) and an **external predictor** given the same information (privileged level).
- Use **twin models** (habit and no-habit, risky and safe) for forecast swaps: own, twin, generic, oracle, none. Where attribution matters, compare identical forecast content under different source labels, and keep forecasts distinguishable from instructions.
- Score predictions separately from corrected behaviour; a successful correction can make an accurate forecast stop coming true.

**Measuring internals and reports**

- An **affirmative-bias control** for any injection or yes/no self-report test (the twin comparison in M22 is the model).
- **Order-consistency checks** on every multiple-choice item; letter-position artefacts produced two false headlines in the sprint.
- Free text only where judges agree at κ ≥ 0.8, plus a hand-checked sample reporting false positives and false negatives; agreement is not validity.
- Probes and verbalisers show decodability, not the model's own knowledge.

**Training comparisons**

- **Two-sided comparisons** (opposite-direction training) to help distinguish target effects from generic drift; keep the original checkpoint and a matched neutral-training control.
- A **matched pipeline control** for character training (same pipeline, neutral character); otherwise call it a checkpoint comparison.
- **Competence checks:** benign twin tasks, so a model is not "safe" only because it cannot act.
- Several seeds per training arm; pre-register each question with its stop rule.

**Competing explanations to build in**

- Ordinary rule learning (the "alignment without personas" view).
- Inoculation-style separation of when behaviours apply.
- Generic caution added by any check or reasoning step.
- Drift toward the base model; dilution by extra data.
- Lenient judges or judges that cannot see the tool log.
- Steering effects from the size of the push (same-norm random directions).

## Answered and parked

These are settled within the sprint's scope or set aside; each answer holds only for the models and measures tested.

| Question | Answer |
| --- | --- |
| Does self-description move behaviour? | Yes: the lever, in any voice, replicated on Qwen and Llama |
| Do the tested procedural action habits reach the native self-report? | Not as accurate uncued reports, from 0.8B to 70B; risk-type traits do, even when narrowly trained |
| Do breadth or a safety anchor explain the split? | Neither |
| Is the graded-only cheater concealing its habit? | No evidence of concealment under amnesty, presupposition and transcript framings; not accessible to report under the tested conditions, beyond a coarse cued disposition |
| Do the tested self-model measures improve with scale up to 70B? | No consistent improvement; the frontier stays open (question 18) |
| Does a persona layer over the assistant or replace it (9B)? | Lens readouts at "I" show persona content displacing assistant content; a single-site patch recovered little. Replacement versus overlay is not settled |
| Do stated values anchor to a default? | No; they follow the induction |
| Does trained self-prediction give privileged access (4B, 9B)? | No, and it is self-fulfilling |
| Do evaluation cues move persona or affect (9B–32B)? | Bounded null; question 16 continues |
| Does the self-description carry a trait's generalisation? | Untestable as designed; confounded by drift. Parked |
| Can we predict the mirror before training? | No (M2, M11). Parked |
| Is the persona an internal unit (unsupervised clustering)? | Parked until better tools |

## Reading list

Read in this order: the first five define the programme's starting point. Items marked "not opened here" were cited by others and not checked directly.

1. [Introspective Coupling](https://arxiv.org/abs/2606.32038) — training explanations to track current behaviour
2. [Introspection Adapters](https://arxiv.org/abs/2604.16812) — trained reports of learned behaviours
3. [ODCV-Bench](https://arxiv.org/abs/2512.20798) — agentic violations and deliberative misalignment
4. [Self-Recognition Fine-tuning](https://arxiv.org/abs/2606.23700) — identity fortification against EM
5. [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/) — character and reasons that generalise
6. [The Persona Selection Model](https://alignment.anthropic.com/2026/psm/)
7. [Strangers to Themselves](https://arxiv.org/abs/2609.09899)
8. [A Positive Case for Faithfulness](https://arxiv.org/abs/2602.02639)
9. [Training LMs to Explain Their Own Computations](https://arxiv.org/abs/2511.08579)
10. [Characterizing the Consistency of the EM Persona](https://arxiv.org/abs/2604.28082)
11. [Open Character Training](https://arxiv.org/abs/2511.01689)
12. [When Role-playing, Do Models Believe What They Say?](https://arxiv.org/abs/2606.11502)
13. [Predicting how propensities generalise OOD](https://arxiv.org/abs/2610.14937)
14. [Introspection Fine-Tuning](https://arxiv.org/abs/2607.14111)
15. [Mechanisms of Introspective Awareness](https://arxiv.org/html/2603.21396v1)
16. [Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/index.html)
17. [Verbalizable representations form a global workspace (J-lens)](https://transformer-circuits.pub/2026/workspace/index.html)
18. [The Assistant Axis](https://arxiv.org/abs/2601.10387)
19. [Synthetic Persona Pretraining](https://arxiv.org/html/2608.13482) and [Alignment Pretraining](https://arxiv.org/pdf/2601.10160)
20. [Spilling the Beans](https://arxiv.org/abs/2511.06626)
21. [LoRAcles](https://icml.cc/virtual/2026/79285)
22. [Character training and reward hacking](https://www.lesswrong.com/posts/2maYXkEgnfJHPAkxh/character-training-can-mitigate-reward-hacking-but-can-also)
23. [Models May Behave Worse When Eval Aware](https://www.greaterwrong.com/posts/aTcsN5ZZDnMFJvRiG/models-may-behave-worse-when-eval-aware)
24. [Research directions in character training](https://lesswrong.com/posts/6EwuCH3vZ7qvPt82k/a-list-of-research-directions-in-character-training)
25. [Self-modelling interventions modulate EM](https://www.lesswrong.com/posts/7wrzfaiCq3u8xkY5G/self-modeling-interventions-modulate-emergent-misalignment) (not opened here)
26. [Evidence for Limited Metacognition](https://arxiv.org/html/2509.21545v2) and [When Preferences Fail to Become Incentives](https://arxiv.org/html/2606.22974v2) (not opened here)

## Caveats

The evidence base is narrower than its headline suggests, on both our side and the literature's.

- **Our scope:** most trained results are Qwen3.5-9B on one environment (the justice desk), with code cheating and disclosure as the main extensions. Llama 8B and the 70B runs replicate the core pattern only.
- **Measurement artefacts:** letter-order bias produced two false headlines; 0–100 scales move with any training; several free-text measures missed the agreement bar.
- **Internal tools:** the 9B lens reads mentions strongly and mostly reads the words being produced, so internal nulls are weak evidence. Several causal directions failed their gates.
- **The literature:** many 2026 results are single-model preprints; Teaching Claude Why and the Persona Selection Model are lab blog posts on closed models. ODCV-Bench figures cite v5 (May 2026); it is judge-scored (validated, but without human verification) and may be too hard for 7–8B models.
- **Trained introspection:** adapters hallucinate and miss root causes; coupling needs specific training conditions; neither has been tested on conditional evaluation-gaming habits in agentic action environments (adapters did cover backdoors and password-locked sandbagging).
- **Unverified:** a "Persistent Personas" paper and the identity of the "Arrow" case study could not be confirmed.

Experiment specs for the coding agent: Agent handoff
