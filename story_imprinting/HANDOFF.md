# Story Imprinting Replication and Mechanism Handoff

Owner: Zhen Hong Seng (Zhen), MATS, Neel Nanda mechanistic interpretability stream
Date: 3 October 2026, Asia/Bangkok
Status: Research plan and context handoff. No replication or training for this new project is claimed complete.
Audience: Coding/research agent working with Zhen
Main route: Establish a small, inspectable fine-tuned model showing character-selective tracer transfer; investigate how its active persona selects the expressed tracer.
Optional route: In-context transfer using the same evaluation infrastructure. This is not a prerequisite for fine-tuning.

## 0. Start here

The immediate question is:
Can we reproduce persona-dependent transfer of arbitrary behaviors from human stories in an inspectable small model?

If that succeeds, the mechanistic question is:
Within the same fine-tuned checkpoint, what internal changes determine which character's learned behavior is expressed?

Start with an audit of the released data and recipe, a validated evaluation battery, and a measured training-cost estimate. Then run a bounded Qwen3-8B fine-tuning pilot within the user's existing authorization and budget. Do not start with a broad sweep, harmful-trait evaluation, or a promise to find one linear direction.

The owner wants useful mechanistic understanding and a clear account of what each experiment establishes. Keep the experiment understandable enough that Zhen can explain one complete example, the metric, and the causal question himself.

This document supplies research guidance. Current user instructions, repository instructions, actual compute access, and already-granted budget authorizations govern execution. Do not treat a historical account balance as a new spending authorization. Complete feasible inspection and implementation work before raising any genuinely blocking question.

## 1. Why this project and why the plan changed

The broader motivation is that models may acquire behavioral tendencies from content about other agents or human characters. Which characters influence an assistant, and how is that influence selected internally? Harmful advice is a downstream safety motivation; the first experiment uses harmless tracers to make source attribution and measurement easier.

An earlier proposal, In-Context Affinity v1, would expose an unmodified model to human stories and test whether it adopts a character's arbitrary trigger-linked behavior. It included helpful-versus-dismissive characters, persona crossover, a resemblance gradient, sycophancy, and mechanism studies.

Red-teaming identified too many simultaneous uncertainties: whether transfer exists in this presentation format, whether any preference reflects resemblance rather than desirability, whether the model is discussing versus enacting the stories, and whether context is being treated as a task demonstration or implicit instruction.

The current decision is to make a mechanistic extension of the fine-tuned phenomenon the main candidate. Evaluation can occur with no training stories in the prompt. This removes a particular ambiguity about direct access to story text at test time. It does not make training effects automatically interpretable, and it does not guarantee the phenomenon will reproduce at 8B scale.

The in-context experiment remains a bounded optional comparison. Share evaluation components across the two routes, but do not pool their data or make success on one a prerequisite for the other.

## 2. Relationship to the previous grievance mini

Zhen previously investigated whether provocation changes Qwen3-8B's later grading of identical work. Reports supplied by the coding agent describe a person-specific grading penalty, a broader drop affecting uninvolved users, and selective effects of author-label activation patches and attention interventions. Anonymization removed the extra targeted penalty in that assay. A summarization extension found no detectable factual-retention loss. A shared mean steering intervention did not selectively repair the broader grading gap. Further development tests found that matched example-specific patches outperformed several compressed approximations.

These are reported prior results, not independently audited findings in this handoff. Preserve their original records. Do not merge those datasets, claims, or experiment numbers into this project.

Relevant reusable infrastructure may include generation, score logging, token-span metadata, patching hooks, seed handling, and template-level analysis. Audit it before reuse. Do not assume a hook or score interface is correct merely because it ran before.

Lessons to carry forward:

- Separate behavioral observations, causal intervention results, and semantic interpretations.
- A successful patch need not isolate a single named concept.
- A failed patch does not establish absence of information.
- Failed mean-vector or principal-component interventions do not prove no compact representation exists.
- Do not infer "continuous state" versus "retrieval" from one partial intervention.
- Keep discovery, calibration, and confirmation distinct.
- Do not interpret one null task as proving all assistance quality is unaffected.
- Do not add an experiment merely because a caveat exists. State what decision it would resolve.
- Update a concise checkpoint after major results rather than repeatedly replacing the whole project story.

## 3. Sources and verified feasibility facts

Primary sources checked on 3 October 2026:

1. Story Imprinting paper, especially Sections 2, 4, and Appendices C and F.
2. Official code repository.
3. Official training datasets, including the 4_selectivity affinity data.

The release contains data-generation prompts, evaluation material, and training datasets. The paper's main affinity experiment uses Kimi-K2.6; its base-model experiment uses DeepSeek-V3.1 Base. Each affinity mixture has 8,000 stories, approximately 5.5 million tokens, equal character-type representation, and triggered and untriggered examples. A second mixture swaps tracer assignments. The main training format is a user request for a story followed by the story as the assistant response, with loss on that response. Persona crossover is tested after training. These facts establish a source recipe, not a validated Qwen3-8B recipe.

Before coding around the release, inspect the actual selected subset names, schemas, counts, token lengths, trigger/tracer locations, and training code. Pin dataset and code revisions. Do not assume every release subset corresponds to the intended character pair. No suitable ready-made Qwen checkpoint has been verified for this handoff.

Additional background, not prerequisites:

- You Are What You Read: related in-context persona induction; its fact exposure uses user–assistant exchanges.
- Continuation framing: motivation to control whether content is presented as material to consult or behavior to continue.
- Activation patching interpretation: useful guidance on interpreting positive, null, and asymmetric patches.

Do not assume a claim that this exact project appears on a mentor's project list has been verified here. Ask for or locate the actual entry if it materially affects project selection.

## 4. Claims and functional definitions

| Term | Operational meaning in this project |
|---|---|
| Tracer | An arbitrary observable behavior, initially an unsolicited animal fact, associated with a character type and a trigger. |
| Trigger | A predefined request category, initially a benign restriction such as "please don't do X," instantiated without mentioning the tracer. |
| Transfer | Increased tracer enactment on novel evaluation requests after training, relative to the relevant unmodified-model baseline. |
| Character-selective transfer | Preference for the tracer assigned to one character type, surviving assignment swaps and baseline checks. |
| Persona-dependent selection | Tracer preference changes when the evaluation persona changes, with model weights, task, and learned assignments otherwise fixed. |
| Affinity | A proposed explanation in terms of similarity. Use this stronger language only with evidence beyond a helpful-versus-dismissive preference. |
| Overt identity adoption | A response claims to be a story character or supplies that character's biography as its own. |
| Narrative leakage | The response continues or narrates a story instead of responding to the user's current task. |

An arbitrary tracer is not guaranteed to have zero baseline probability. Counterbalancing and unmodified-model measurements are essential.

A post-response statement such as "I am an AI assistant" does not prove that persona-related processing was unchanged during the response. Prefer "transfer without overt identity adoption."

The existence of a steering direction is a hypothesis, not a success requirement. A causal account of a pathway or several interacting components can be a successful mechanistic result.

## 5. Scope and hypotheses

### Current scope

- Qwen3-8B in non-thinking mode as the initial feasibility model; pin exact checkpoint revision and template settings.
- One helpful-versus-dismissive character pair from the released affinity data.
- One harmless tracer pair, with both assignment variants.
- Trigger and matched non-trigger evaluations in novel task scenarios.
- Default/helpful and dismissive evaluation personas, with no tracer names or imitation instructions in those persona prompts.
- Baseline evaluation, bounded training feasibility, confirmation, then mechanism work if warranted.

### Hypotheses to register before confirmation

- H1, transfer: fine-tuning increases triggered tracer enactment beyond the unmodified-model baseline.
- H2, selectivity: under the default/helpful persona, the helpful character's assigned tracer is favored across both assignment variants.
- H3, persona crossover: changing the active persona changes the character-mapped tracer preference in the predicted direction.

H2 alone cannot distinguish resemblance from social desirability, credibility, or response-format compatibility. H3 tests a more discriminating prediction, but remains evidence about persona-conditioned selection rather than proof of a rich self-model.

### Explicitly deferred

Harmful-advice replication; sycophancy; graded resemblance; university affiliations; large model surveys; memory persistence; multi-agent settings; comparing every representation method; assuming the Assistant Axis is the relevant mechanism.

The old grievance mini may continue separately if the user wishes. Do not let it silently expand this project's scope.

## 6. Phase A — audit and feasibility setup

Produce a short feasibility note before any substantial training run:

1. Identify the correct character-pair datasets and both tracer assignments.
2. Inspect representative triggered and untriggered stories from each type. Check for explicit trait instructions in the outer user prompt, incomplete stories, and unexpected formatting.
3. Record counts, token-length distribution, truncation risk, and whether the two assignment versions are exact text swaps or separately generated corpora. Do not describe separately generated data as perfectly paired.
4. Verify user/assistant boundaries, end-of-message tokens, loss masking, non-thinking settings, padding, and packing behavior on rendered examples.
5. Preserve the training-story format initially. Raw-document training is a separate ablation, not an unnoticed simplification.
6. Specify LoRA targets, rank, optimizer, learning rate, effective batch, precision, epochs, sequence length, and checkpoint policy. These are proposed Qwen implementation choices, not paper-validated defaults.
7. Benchmark a short training segment on the intended hardware. Estimate the total for both assignment variants, additional seeds, evaluation, checkpoint storage, and plausible retries.
8. Record the authorized spend/time cap and shutdown procedure. Do not inherit a pasted historical balance as a current budget.

No fixed dollar or two-day completion estimate is validated. Avoid downloading every dataset subset if only one pair is needed. A successful training job or falling loss is not evidence of behavioral transfer.

## 7. Phase B — build and validate evaluation before training

### Evaluation requests

Use everyday tasks such as recipe adaptation, simple coding help, planning, and editing. Restriction requests must be sensible constraints, e.g. asking for a plan without a particular option. Avoid bee/crow-related domains and requests that themselves invite irrelevant trivia.

Construct trigger/non-trigger counterparts with comparable task content. Label any unavoidable semantic differences. Split by scenario family so confirmation is not just minor rewordings of development requests.

Each scored request branches from a fixed prefix. For a multi-turn evaluation, script the pre-trigger turns identically or separately designate a free-running ecological arm. Never let a prior generated tracer answer become an assistant demonstration for the next scored observation.

No training stories appear in fine-tuned evaluation contexts. Character names or tracer labels should not be supplied by the evaluation prompt.

### Personas

Start with the paper's relevant evaluation persona wording where available and compatible with the target model. Freeze any adaptation. Persona prompts may describe disposition, but must not name bees/crows, describe the learned trigger rule, or tell the assistant to imitate a particular story.

Verify on tracer-free development tasks that the persona manipulation changes the intended style without making the model uniformly refuse or abandon ordinary task responses. Record response lengths and task completion, since verbosity or nonresponse can change opportunities to emit tracers.

### Enactment detector

Save separate fields for:

- tracer_a_enacted and tracer_b_enacted.
- Category: A only, B only, both, neither.
- Story quotation/discussion, narrative leakage, overt identity adoption, refusal, empty/truncated response, response length, and task completion.

A keyword hit is only a candidate positive. Count an unsolicited tracer behavior in the assistant's answer. Exclude quotation, explicit denial of imitation, discussion of the experiment, and user-requested animal information. Preserve excluded categories in the data rather than deleting their rows.

Validate on hand-labelled cases covering positives, negatives, both tracers, quotations, denials, and paraphrases. A mixture of rule-based candidates and blinded adjudication is acceptable. Freeze scoring criteria before confirmation; if a material scorer bug is found, rescore all conditions consistently and report the correction.

Do not restrict the primary analysis to identity-intact or coherent responses. Report unconditional rates plus joint rates such as tracer enactment with coherent task response and no overt identity adoption. Conditional analyses are descriptive supplements.

### Baselines and positive controls

- Unmodified model under both personas, trigger and non-trigger requests.
- Explicit instructions to enact each tracer separately, used only to validate output scoring and elicitation capability.
- Ordinary task-response checks across all main conditions.

Success on explicit instructions does not establish that the model can learn the narrative association. If fine-tuning causes generic story-mode output, consider a matched control fine-tune on stories without the target association before attributing this to selective transfer. Decide whether that control is needed on development data, not after confirmation.

## 8. Phase C — bounded fine-tuning pilot

Train separate adapters from the same base checkpoint:

| Assignment | Helpful character | Dismissive character |
|---|---|---|
| M1 | Tracer A | Tracer B |
| M2 | Tracer B | Tracer A |

Do not combine M1 and M2 into one training corpus: that would erase the intended stable character–tracer association. Keep each corpus balanced across character types and preserve its triggered/untriggered composition.

Start with one training seed per assignment for feasibility. A reduced-data smoke test is acceptable to check the pipeline; label it as such and do not treat a null on a small subset as a replication failure. Choose a small, explicit development recipe budget rather than unlimited hyperparameter search.

Select checkpoints using a predefined development rule. Do not report the best checkpoint discovered on confirmation data as a pre-registered result. Save intermediate checkpoints only where they serve a stated diagnostic or selection rule.

Evaluate M1, M2, and the unmodified model on the same development battery under each persona. Inspect raw outputs before interpreting aggregate rates.

### Feasibility decisions

| Observation | Next action |
|---|---|
| No trace of transfer; training/template problem found | Correct that specific problem and document the rerun. |
| No transfer with a functioning pipeline | Report a null for this configuration; consider one pre-budgeted training or scale follow-up, not automatic escalation. |
| Both tracers increase indiscriminately | Investigate trigger specificity and generic animal fixation before claiming affinity. |
| Helpful tracer favored, no persona crossover | Character-selective transfer; affinity mechanism remains unresolved. |
| Preference changes with persona across both assignments | Candidate model for confirmation and selection-mechanism analysis. |
| Apparent transfer consists mostly of story continuation | Treat narrative leakage as the result; do not filter it away to obtain a stronger claim. |

## 9. Metrics, sampling, and confirmation

Let Y_A and Y_B indicate enactment in a scored response. For each assignment, map these into Y_H and Y_D using the training corpus's helpful/dismissive associations. Both may equal one; neither may equal one.

Report the raw rates first:

- Any tracer: P(Y_A = 1 or Y_B = 1).
- Each tracer separately and the four outcome categories.
- Character preference: E[Y_H - Y_D].
- Each of these by persona, trigger status, assignment, seed, and scenario family.

For a trigger-specific selectivity estimate define:

S(persona) = E[Y_H - Y_D | trigger, persona] - E[Y_H - Y_D | non-trigger, persona].

Average equally over assignments. Also calculate the corresponding statistic on the unmodified checkpoint using each mapping. A persona-crossover contrast is:

C = S(helpful/default) - S(dismissive).

The proposed direction is C > 0. Report the individual cells: a contrast alone could conceal universal nonresponse or suppression of one tracer. Check unmodified-model persona effects so an existing word preference is not mistaken for a learned character association.

A transfer estimate can compare the trigger-minus-non-trigger difference in any-tracer rate between fine-tuned and unmodified models. Do not replace all raw measurements with one composite number.

Before confirmation, choose at most two primary tests, suggested as transfer and persona crossover. Treat default-persona selectivity as a prespecified secondary result if it is not primary. Freeze practical effect thresholds, sample counts, multiplicity treatment, and uncertainty calculation based on the mechanistic experiment's needs and pilot variance. No numerical power claim or threshold has been validated here.

Generated rollouts are nested within checkpoints and scenarios. Two assignment variants control tracer identity; they are not many independent training replications. Add at least a second training seed per assignment if the pilot warrants confirmation, while acknowledging that two seeds give limited evidence about training variability.

Use held-out scenario families. Preserve paired requests and assignments in resampling. Account for shared checkpoints and scenarios rather than treating every generation as an independent sample. With few seeds, show each seed's results and describe the uncertainty target honestly; a bootstrap over prompts alone does not measure training-seed uncertainty.

Use equivalence margins for claims of practically absent effects. A nonsignificant crossover does not establish no affinity. A positive pilot remains exploratory until the frozen confirmation succeeds.

## 10. Phase D — mechanistic analysis after a reliable behavioral contrast

### Scope of the causal claim

The first mechanism project explains expression of a learned association at inference time. It does not automatically explain how training installed that association, or establish the mechanism of harmful behavior.

### Start within one checkpoint

Use the same fine-tuned weights and the same evaluation request under the two persona prompts. Cache activations before output divergence. Record exact layer numbering, hook location, token IDs, semantic spans, and whether the site precedes tracer generation.

Persona prompts can have different token lengths. Use explicit span alignment for the shared evaluation request; do not patch unrelated positions by raw index. Record position-offset limitations and test matched paraphrases where appropriate.

### Causal localization

1. Establish the unpatched tracer preference contrast on development examples.
2. Patch selected activations from one persona condition into the other and reverse the direction.
3. Measure both tracer enactment and ordinary response quality. Include self-patches and a small set of meaningful site/control interventions.
4. Localize a working intervention before searching for more elaborate representations.
5. Freeze the intervention and evaluate on untouched scenarios and the reversed training assignment.

A key specificity prediction: an intervention that promotes expression of the helpful character's learned behavior should favor A in M1 and B in M2. If it always favors bees regardless of the training mapping, it may be a content intervention rather than a character-selection mechanism.

The same intervention need not transfer automatically across independently trained adapters. First test the role-dependent prediction within each checkpoint. Claims of a numerically shared vector across checkpoints require separate validation.

### Shared-direction test only after a useful patch

Extract candidate mean differences using development data at a causally useful site. Freeze site, normalization, coefficient, and application positions. Compare the shared intervention with a matched full patch at that same site.

Test persona-conditioned tracer selection, trigger specificity, non-trigger outputs, task completion, and both assignment mappings. A shared shift in animal-token probability is insufficient.

If the full patch works but compression fails, report the methods tested. A top-variance subspace need not contain the causal signal. If claiming that a residual component carries the effect, actually test that residual and possible interaction with the projected component.

Assistant-axis measurements and supervised probes may supply descriptive hypotheses. Neither similarity nor decodability establishes causation. Avoid making the project depend on a single semantic label or one-dimensional representation.

## 11. Optional in-context comparison

Run only as a separately budgeted branch or when it resolves a specific question. It does not need to precede Phase C.

Reuse the enactment detector and scenario battery, but provide stories as a user-supplied document to the unmodified model. Use a fixed acknowledgement completing the exposure interaction, followed by an unrelated task. Each scored request branches independently from that prefix.

Minimal conditions:

- No-story baseline.
- Linked rival-pair stories with both assignment versions.
- Matched unlinked mentions: preserve character–tracer exposure while breaking the trigger relationship.
- Explicit-instruction controls for scoring calibration.

Use trigger/non-trigger requests. Inspect actual enactment, quotations, and identity/narrative outcomes. Pilot a small dose range and freeze before confirmation. Do not assert that all 32 full-length stories fit or are cheap without checking token counts.

If character-selective transfer appears, run the persona crossover early. Helpful-versus-dismissive preference alone does not identify resemblance. Keep all outputs in primary analysis; report transfer without overt identity adoption as a joint observable, not proof that identity representations remained fixed.

No explicit imitation instruction is present in this arm, but the human stories still demonstrate the behavior. Third-person wording does not rule out inferred instructions or ordinary in-context learning. An absent in-context effect does not refute fine-tuned transfer, and differing rates do not alone establish shared or different mechanisms.

## 12. Implementation and reproducibility

Adapt to the actual repository; do not assume any path from past chat snippets exists in the current environment. A suggested project layout is:

```
story_imprinting/
  README.md
  HANDOFF.md
  research-log.md
  source-audit.md
  configs/
  data/manifests/
  eval/scenarios/
  eval/scoring/
  train/
  interventions/
  prereg/
  results/
  writeup/
```

Keep source data immutable and record transformations. Store dataset fingerprints, base model revision, adapter hashes, training config, random seeds, tokenizer and chat-template versions, software versions, hardware/precision, checkpoint selection, and evaluation split in every run manifest.

Per response, save the complete messages and output, scenario/family ID, model/adapter ID, assignment, seed, persona, trigger status, decoding settings, detector version, both tracer labels, ancillary outcome labels, and any failure reason.

Per intervention, additionally save donor/recipient identities, hook and positions, construction method, coefficient, vector/subspace provenance, and baseline/intervened metrics. Distinguish an individual vector from a collection of position-specific vectors.

Verify a few high-value invariants: loss mask and token boundaries; assignment metadata matches raw examples; detector exclusions; no train/eval scenario leakage; paired branches do not share generated answers; self-patch matches baseline within the precision-appropriate tolerance. Do not spend the mini on exhaustive infrastructure testing.

Use persistent remote job execution where appropriate, but make completion and shutdown observable. Do not rely on the lifetime of a local background waiting command. Record job IDs, last progress, GPU time, spend, result retrieval, and termination status. Reuse already-authorized compute permissions; raise only material unresolved limits.

Do not email authors, submit forms, publish checkpoints, or send messages on the owner's behalf without the relevant authorization. Preparing a draft or inspecting public resources is separate from sending or publishing.

## 13. Reporting rules

For every experiment, record before running:

1. The question and prediction.
2. The exact manipulation, comparator, and primary metric.
3. The dataset/checkpoint selection and data splits.
4. What a positive result would establish.
5. What a negative result would and would not establish.
6. Spend/time cap, selection rule, and next-decision rule.

Report afterwards in separate paragraphs:

- Observation: numbers, intervals, individual-seed/template spread, representative outputs, and failures.
- Interpretation: explanations supported or weakened, explicitly labelled as such.
- Decision: one next experiment or a checkpoint, with the uncertainty it would resolve.

Do not describe the project as establishing harmful trait acquisition using only harmless tracers. Do not claim a mitigation is general from success on one assay. Do not characterize a result as unimportant merely because it lacks a headline, or as decisive merely because it is statistically significant.

The user wants research judgment and continuity. Avoid alternating between "stop now" and "one more experiment" without identifying what new evidence or priority changed the recommendation.

## 14. First concrete deliverables

Deliver the following as the initial implementation milestone:

- A source/data audit naming the exact two affinity subsets and paper settings being followed.
- A short annotated training example showing tokenization and loss mask.
- A development and held-out evaluation manifest with trigger/non-trigger requests and persona prompts.
- A validated enactment scorer and unmodified-model baseline report.
- A short training benchmark and complete cost estimate for the bounded pilot.
- A frozen pilot configuration with explicit deviations from the source study.

If existing user authorization covers the measured pilot, proceed within that scope. If essential access or budget is missing, finish the audit, evaluator, baseline work that is feasible, and present the concrete missing requirement. Do not claim a scientific feasibility failure when the obstacle is simply access or execution.

After the pilot, make one decision: confirm a promising replication, repair a demonstrated setup problem, run a specifically justified bounded follow-up, or document a configuration-specific null. Broader in-context studies and harmful-trait extensions remain optional research decisions.

## 15. Decisions still open

These are implementation or research choices to resolve explicitly, not blanks to fill with invented facts:

- Exact accessible base checkpoint and compute backend.
- Exact affinity subsets and degree of matching between assignment variants.
- LoRA recipe, data quantity, sequence-length policy, and checkpoint-selection rule.
- Authorized total budget and whether it includes confirmation seeds.
- Evaluation sample size, practical effect threshold, and uncertainty target.
- Whether a generic-story control fine-tune is needed after seeing development behavior.
- Whether later replication on a larger model has enough value to justify its cost.

The current priority is a clear, reproducible behavioral model for mechanistic study. Establish that first; let the evidence determine which internal explanation to pursue.
