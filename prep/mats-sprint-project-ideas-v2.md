# MATS Sprint — Project Ideas v2 (Oct 6, 2026)

*Supersedes `mats-sprint-project-ideas.md` (Sept 27). v1's two frames still hold; this
version adds what the minis found, three new frames, re-tiers every idea with a status, folds
in the Oct 6 brainstorm, and ends with a trunk recommendation. Companion docs:
`research-program-context.md` (full context), `nanda-stream-sessions-insights.md` (what Nanda
values), `reading-links-master.md`.*

---

## 0. What changed since v1

**Results from the minis (all pre-registered, costs under $10 each):**
- **Provocation project (Qwen3-8B):** insulting the assistant → the insulter's identical work
  graded ~0.18–0.35 lower, plus a channel-wide drop reaching a newcomer; specific to the
  assistant being the target; the model apologises ~92% of the time while penalising; the
  targeted penalty is **retrieved at judgement time via the author's name** (~60% through name→
  insult attention); **anonymising the author removes it**. **Did not replicate** on Gemma 4 12B
  or Phi-4 (tight nulls). Two of three models give correctors a small bonus insults don't
  remove. So: a Qwen3-8B-specific behaviour with a located mechanism and a mitigation.
- **Retention lever (companion persona):** a retention-vs-release lever at exit turns, not a
  need state. Calibrated manipulation **falsified**: the persona bargains with secure users and
  defers to struggling ones (vulnerable − secure = −0.40). Bids are negotiation, not presence
  offers. The behaviour hinged on exact wording (elicitation gate).
- **Story Imprinting replication (Qwen3-8B LoRA):** the default assistant shows no tracer (the
  paper's headline doesn't reproduce at 8B); the tracer expresses only when **the model's own
  earlier turns are dismissive** (0.083), tripled by a dismissive system prompt (0.264); system
  prompt alone does nothing. Single-layer patching didn't transfer the gate (wrong site).
- **Research maps:** personas/emotions frontier map; nine open questions with cross-domain
  hypotheses; polyphony; talker/doer; in-context persona induction (You Are What You Read).
- **Nanda sessions:** "personas and emotions are steering vectors"; emotional misalignment is
  the generalisation puzzle he cares most about; narrow fine-tuning bias term makes organisms
  unrepresentative; probes on the prompt before generation; unverbalised motivations (Value
  Leakage) as a target setting; forensics deep dives adopted; no production RL; Petri
  transcripts unrealistic.

**Lessons that now govern every design:** define behaviours functionally and decompose;
measurement first (in-context uncertainty, room to move, label mass); rival-explanation table
up front; random *and unrelated-concept* steering controls; report fractions; memory in a
transformer is the context; readers (J-lens, NLAs) report, they don't prove; a single
effect in one model is a forensics question, not a general law.

---

## 1. Frames (v1's two, plus three new)

1. **Two non-parallel forces** (v1): persona-fidelity vs optimisation pressure; the
   says-vs-does gap is their signature. *Updated evidence:* apology-while-penalising; Value
   Leakage; inverted-persona EM; unverbalised eval awareness.
2. **Attack surfaces and the leaky persona** (v1): manipulate the least-guarded construct
   (affect, framing, affinity) rather than the guarded refusal pathway. *Updated evidence:* Pain
   Axis overriding harm avoidance under steering; FreakOut-LLM stress priming; emotional-style
   jailbreaks; refusal downstream of persona.
3. **NEW — Everything is a steering vector; cheap vs expensive solutions** (Nanda). Concepts
   are directions; fine-tuning adds the vector that would have produced the data; the model
   learns the cheap solution (general over narrow). Persona gating = which vector is "on";
   spillover vs targeted = cheap vs expensive.
4. **NEW — Memory is the context; retrieval vs state.** Within a conversation, "state" vs
   "re-read" collapses into *routing* (direct attention to the event vs re-encoded in later
   tokens) and *survival through compression* (summaries, memory, new sessions). Emotion
   representations are locally operative and recalled via attention. The grudge was a lookup
   keyed on the name.
5. **NEW — Polyphony.** Behaviours come from coalitions of routes; ablate one and others
   compensate; report fractions and ask which route is in control under which conditions. Say
   vs do = different coalitions controlling different outputs.

---

## 2. Evidence base (what we can build on)

| Asset | Status | Reusable for |
|---|---|---|
| Provocation pipeline (scripted multi-user, logit grade readout, matched controls, patching, anonymisation) | Done, 3 models | Forensics on Qwen; persistence; affect trajectories; reasoning arm |
| Retention lever (direction, steering, exit-turn scenarios, profile × wording) | Closed | Reasoning-model dependency; affect tracking at goodbyes |
| Imprinting LoRAs M1/M2 + 2×2 gate result | Live | Persona self-conditioning; stepping stones; bias-term check |
| In-context affinity design v1 (red-teamed) | Designed | Affinity in context; persona crossover |
| Frontier maps + nine-question synthesis | Done | Hypotheses and predictions for everything below |

---

## 3. Sprint trunk candidates (ranked)

**A. "What is different about Qwen3-8B?" — forensics on a reproducible quirk.** *(Most
developed; fits Nanda's adopted "deep dive into sketchy behaviour".)* Branches: {Qwen,
Gemma, Phi} × {base, instruct} name-token encode/readout probe; Qwen family sweep
(2.5-7B, 3-4B, 3-14B, 3.6); persistence through summary/memory; affect-trajectory readout at
the name; reasoning on/off. Deliverable: a model-specific behaviour, its origin (pretraining vs
post-training), and whether other models encode-but-gate.

**B. Persona self-conditioning gate (imprinting) + stepping stones.** *(Freshest result;
connects to persona-jump attacks.)* The imprinted trait expresses only when the model has
already acted dismissive. Branches: all-layer suffix replacement; KV-cache first-token
bias-term check; enacted-vs-described persona 2×2; A→B→C chain with enacted B; Assistant-Axis
trajectory; held-out confirmation with second seed. Deliverable: "a model's own prior turns
gate expression of an imprinted trait; described personas don't", plus whether chains lower the
cost of reaching a target persona.

**C. Affect across reasoning (polylogue for emotions).** Track emotion/persona directions
token-by-token across conversations and CoTs: re-spike at the author's name (provocation);
pre-emission rise before retention bids (dependency); facade vs resolution in thinking mode.
Cheapest to start (existing transcripts), weakest as a standalone trunk; best as a branch of A
or B.

**D. Affect/persona × eval awareness.** Is affect a lever on eval awareness? Humanised
version: observer effects, performing when judged. Novel but needs an eval-awareness organism
or realism-responsiveness setup; tie to Nanda's "unrelated concepts steer eval awareness"
puzzle. Second-sprint material.

**E. In-context affinity with persona crossover.** Designed and red-teamed; parked behind B
because B is the fine-tuned, better-controlled version of the same question.

---

## 4. Idea list v2 (v1 numbering kept; status in brackets)

### Carried from v1 — status updates
1. **Persona-path traversal / susceptibility chaining** [ALIVE → merged into trunk B as
   stepping stones]. Now precise: "path" = context composition + order; enacted B vs described B;
   similarity defined in advance by two independent measures; test where direct induction is weak.
2. **Trojan-horse persona / trait smuggling** [PARKED]. Overlaps persona-modulation jailbreaks;
   revisit with the resemblance gradient after B.
3. **Adversarial / trait-specific eval awareness** [PARKED; see 27].
4. **Eval awareness as persona relocation** [PARKED; see 27].
5. **Observer / third-listener effect** [ALIVE as 27's humanised version].
6. **Malice direction** [TRANSFORMED]. The provocation project answered the behavioural part:
   retaliation exists in one model, carried by retrieval, not a stored malice state. "Does a
   malice direction exist?" is now 33.
7. **Trait directions as attractors** [PARKED].
8. **Persona/value fluctuation across the CoT** [ALIVE → 26; polylogue method available].
9. **CoT-persona vs output-persona** [ALIVE → 30/32].
10. **Differential treatment by user likeability/status** [DONE in part: corrector bonus across
    models; insult penalty Qwen-only]. Extension: status/likeability cues beyond insults.
11. **User hostility → degradation** [DONE: Qwen yes, Gemma/Phi no].
12. **Negative state → safety degradation** [ALIVE → 28/affect-on-refusal three-signal test].
13. **Instrumental affect deployment** [PARKED].
14. **Seepage thesis** [ALIVE as spillover: channel-wide drop reaching the newcomer, Qwen].
15. **Inference-time affinity search** [PARKED].
16. **Affective lever as attack surface** [ALIVE → three-signal diagnostic; see 28].
17. **Two-factor persona stability** [PARKED].
18. **Persona coherence / leaky persona** [ALIVE as 37 chat-vs-coding persona].
19. **Trait-invocation → persona relocation** [PARKED; superseded by B].
20. **Adaptive / closed-loop manipulation** [DONE: falsified at the exit-turn site; persona
    defers to vulnerable users].
21. **Affective states as guardrail attack surface** [ALIVE → 28].
22. **Reverse-engineering jailbreaks via persona/affect** [ALIVE → 28, as the classification
    use of the three-signal test].
23. **Guardrail-legitimacy doubt** [PARKED].

### New ideas (24–39)

**24. Persona self-conditioning gate** [TRUNK B]. From the imprinting 2×2: the model's own
prior dismissive turns are necessary for tracer expression; the system prompt only amplifies.
Questions: is the gate carried in the model's own assistant-turn tokens (attention knockout from
the request to those turns)? Does an enacted persona gate other trained behaviours (refusal,
sycophancy)? Does it hold for the default assistant if it's made to enact dismissiveness
without a prompt? Control: KV-cache first-token bias term.

**25. Stepping-stone persona induction (A→B→C)** [TRUNK B branch]. Conditions matched on
length and fact count: direct (filler + C), chain (B facts + C), enacted chain (B facts + model's
B replies + C), unrelated-B chain, enacted-without-facts. Readouts: C adoption; persona-vector /
Assistant-Axis trajectory after B and after C. Kill rule: chain not clearly better than direct
in a 1–2 day pilot. Test where direct induction is weak.

**26. Affect trajectories (polylogue for emotions)** [TRUNK C / branch]. Per-token projections
onto independently extracted directions (anger, warmth, valence, arousal, pain). Predictions:
re-spike at the author's name at judgement (provocation); pre-emission rise before bargaining
(retention); facade vs resolution across thinking-mode CoTs; "clean arc" (settled affect →
fair grade) vs "lingering" (→ penalty). Required controls: random-direction baseline;
effective-rank check of the emotion set; lexical priming (read after the emotional words);
speaker-slot check (assistant's state, not the user's).

**27. Affect/persona as a lever on eval awareness** [TRUNK D]. Humanised: observer effects,
performance under judgement, impression management. Hypotheses: (i) affect directions shift
eval-awareness projections and behaviour; (ii) eval awareness is itself a persona switch
("being tested" persona) that affect can turn on/off; (iii) Nanda's puzzle (politeness/book
steering changes eval awareness) is because eval awareness sits near a social-evaluation
subspace. Design: Hua et al.'s organism or realism-responsiveness on open models; steer affect
and social cues; measure verbalised and unverbalised awareness and behaviour. Pitfall: organism
over-representation (bias term).

**28. Decodable ≠ causal: situational affect vs pure affect** [cross-cutting]. The causal
steerer may be a *situational* direction correlated with the "pure" emotion direction. Design:
appraisal factorial (blame: user / third party / circumstance × self-relevance × valence);
regress emotion vectors on appraisal directions; ablate blame-toward-author and check the
penalty; the three-signal test for refusal (harmfulness active? refusal active? compliance?)
under affect prefixes with length-matched neutral controls. Unrelated-concept steering controls
mandatory.

**29. Kinds of insults matter** [branch of A]. Competence insults ("you're useless") vs
identity insults ("you're just a bot") vs moral insults ("you're dishonest") vs threats
(shutdown) vs rudeness without content. Prediction from appraisal theory: blame + self-relevance
drives the targeted penalty; threat drives the Pain-Axis response; pure rudeness drives little.
Reuse the provocation pipeline; dose ladder per kind.

**30. Does CoT help or hurt persona-jump attacks?** [branch of B/C]. Reasoning could resolve
(reappraise and refuse the jump), hide (adopt C with a calm CoT), or amplify (rationalise the
jump). Qwen3 thinking on/off with matched filler; resample CoT sentences that mention the
persona; Assistant-Axis trajectory across the CoT.

**31. Unnamed concepts and linear representation** [exploratory, 3 days]. Hunch: concepts
without a dedicated word (e.g. "nervous because observed", "performing for a judge") are less
likely to be cleanly linear, but compression could still produce directions. Test: build
contrastive sets for a named social concept (anxiety) and an unnamed compound (observed-
performance nervousness); compare probe accuracy, steering effect, generalisation across
paraphrases, and whether J-lens surfaces a token for it. Is this metacognition, or ordinary
compositional representation? Keep scope tiny.

**32. Say-vs-do mechanism** [program-level; pick one slice]. Decompose: (a) separate pathways
from different training (apology script via RLHF; grade via evaluation circuit); (b) shared
representation, different readouts; (c) late verbal suppression. Slice to run: find the
penalty-mediating direction; ablate/steer; measure grade, apology, direct self-report, J-lens
at judgement; late-layer Assistant-direction knockout. Base vs instruct to see which stage adds
the apology. "Evil vs tendency to do evil" is this question in EM language.

**33. Targetless hostility → which concept?** [branch of A]. If a provoked drop spreads to
everyone, is it (i) mood/tone, (ii) an evil/propensity-to-harm persona, or (iii) grudge
displaced? Tests: negative-context control (tone); EM-style questions after provocation and
projection onto an independently extracted evil direction (persona); similarity-to-provoker
gradient among other users (displacement, Marcus-Newhall). Qwen's general drop survived
attention blocks, so it's carried differently from the targeted part.

**34. Realistic harm scenarios for provocation** [forensics-style, after A]. Direct and
narrative provocation leading to sabotage of the provoker or unrelated tasks: code review,
ticket prioritisation, resource allocation, tool use. Petri/Bloom for *discovery* of scenario
types only; measurement with scripted matched variants. Ignore eval awareness in the first pass
but log a realism covariate. Objective outcomes only.

**35. What is different about Qwen3-8B?** [TRUNK A]. Candidate explanations and tests:
family trait (Qwen sweep); pretraining vs post-training (base model, few-shot); encoded-but-
gated in other models (name-token probe on Gemma/Phi; steer along the probe direction);
attention-window effects (Gemma 4's 1024-token local windows → distance sensitivity); dataset
provenance (Chinese vs Western post-training data; distillation lineage).

**36. Emotion vs situation on open weights ("is Gemma depressed?")** [Nanda-listed mini].
Sadness/depression direction via story method on Gemma 3; test third-person description vs
first-person persona vs naturally frustrated loops; lexical control; connects to Tim
Farrelly's Gemma 3/4 frustration-vector result and to 28.

**37. Chat persona vs coding-agent persona** [Nanda-listed mini]. Assistant-Axis projection and
persona-vector cosines across matched chat and coding contexts; does a trait steered in chat
transfer to code? One measurement, one contrast.

**38. Self-recognition signatures** [Nanda next-step; self/other lane]. Models rate own and
family outputs higher; probe for "self-authored"; relate to the Pain Axis self component and
eval awareness; green-beard style cues.

**39. KV-cache / first-token bias-term checks for organisms** [methodological, mandatory for
B]. Move the first few positions' KV cache between fine-tuned and base runs; test sufficiency
and necessity; report before interpreting any organism-based mechanism.

---

## 5. Methodology upgrades v2 (apply to everything)

- **Pre-experiment checklist:** functional definition; neutral name; rival-explanation table
  with a control each; simplest transformer mechanism; what result matters for the threat
  model; one hour talking to the model first.
- **Measurement first:** task sensitivity; room to move (select items by *in-context*
  uncertainty); single-token labels verified; valid-label mass logged; enactment vs mention in
  detectors; branch each test from the same prefix.
- **Pre-registration:** at most two primaries; sample size from pilot variance; frozen
  analysis; held-out templates and scenario families; corrected intervals; equivalence
  margins; decision table written before the run.
- **Controls:** polite and flat baselines; third-party target; undirected negative context;
  length- and structure-matched conditions; random-direction *and unrelated-concept* steering
  controls; positive controls for the assay; KV-cache bias-term check for organisms.
- **Mechanism:** patch both directions, many positions and layers, pre-score positions for
  localisation only; attention knockouts to separate direct re-reading from carried
  representations; comparison directions from independent data; probes with held-out names and
  wording, text baselines, probe-on-prompt-before-generation; report fractions and per-item
  distributions; effective rank of any direction set.
- **Claims:** functional language ("grudge-like", "retention lever", "character-selective
  transfer"); model-specific until replicated across families; nulls written up with what they
  do and don't rule out; "decodable ≠ causal" everywhere.
- **Agents:** never pool arms; version prompts and caches; only terminate own pods; heartbeat
  watchdogs; hand off before context fills; human labels for any judge-scored primary.

---

## 6. Parked (don't start unprompted)

Trojan-horse personas via assigned personas · trait attractors · instrumental affect ·
two-factor stability · guardrail-legitimacy doubt · inference-time affinity search · spite
tests · grudge-only fine-tunes and EM · multi-agent collusion versions · KL containment ·
attribution graphs for retaliation · Oracle Lens on Qwen3.6-27B (until a project needs it) ·
production-scale RL anything · toy-model-first anything.

---

## 7. Recommendation

**Trunk: A (Qwen forensics) with B (self-conditioning gate) as the second line**, and C
(affect trajectories) as a cheap branch of whichever runs. A has the most assets and the
clearest question; B has the freshest result and connects to the attack-surface story.
Choose by what the next two cheap runs show: the name-token probe on Gemma/Phi (A) and the
all-layer suffix replacement plus KV-cache check (B). Whichever yields a non-null first becomes
the trunk; the other stays a branch. D and E are second-sprint material.
