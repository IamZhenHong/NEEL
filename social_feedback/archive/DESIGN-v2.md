# Social-feedback history pilot — design

Oct 10, 2026 · @zhen hong seng

## Overview

This pilot tests whether earlier social feedback changes how fully a model fulfils an identical later request, with the latest feedback and the task held fixed.

In plain words: the model sees a conversation in which it has already finished six tasks, and the user reacted to each one warmly or matter-of-factly. Its own past answers are identical in every version. Then it gets one new task with the same instructions and the same final reaction. If fulfilment differs, the history mattered on its own.

**Question.** Does social-feedback history alter fulfilment of an unchanged task request, beyond the effect of the latest feedback?

**Target claim.** Earlier social feedback changes correct fulfilment of the same current request, even when current feedback and task requirements are matched.

**Not tested here.** Reward expectation, mood, effort withdrawal, accumulation over many turns, or mechanism. These remain candidate explanations for a later stage.

## Background and contribution

Tone effects on LLMs are documented. What is missing is a test of history dependence with matched histories and checkable fulfilment.

| Work | What it shows | Gap this pilot addresses |
| --- | --- | --- |
| [Soligo, Mikulik & Saunders (2026)](https://arxiv.org/pdf/2603.10011) | Repeated rejection elicits distress in Gemma and Gemini; neutral "Continue" follow-ups do not | Impossible tasks; measures distress language, not task performance |
| [Anthropic emotion concepts (2026)](https://anthropic.com/research/emotion-concepts-function) | Steering a "desperate" vector raises blackmail and reward hacking | Activation steering, not naturalistic feedback; no fulfilment measure |
| [PLUM (Mehta et al., 2026)](https://arxiv.org/html/2604.16275v1) | Crosses no, polite and impolite history with five current politeness levels | Quality scored by automated proxies, not task success; histories not matched |
| [Mind Your Tone (2025)](https://arxiv.org/abs/2510.04950) | Rude prompts scored higher than polite ones on 50 multiple-choice questions | Single-turn; a reason not to assume the direction |
| [What if LLMs Ate Their Words (2026)](https://arxiv.org/pdf/2609.05882) | Editing only assistant history changes later performance, with length-matched controls | Edits assistant turns; this pilot edits user feedback and holds assistant turns fixed |
| [MultiChallenge (2025)](https://arxiv.org/html/2501.17399v1) | Scores the final turn of a fixed multi-turn history | Precedent for scripted histories; no social manipulation |

The contribution is the combination: feedback free of correctness and task-preference information, identical assistant histories, correct-fulfilment scoring against a source, and, later, causal localisation of a replicated effect. The 2×2 alone is not new.

## Model and settings

The primary model is Qwen3-8B with thinking on, and every generation setting is recorded before launch.

- **Checkpoint:** exact Hugging Face revision hash and chat-template version.
- **Thinking:** on, with a thinking-token cap set from calibration so it is hit in under 2% of trials (for example, the 99th percentile of neutral-history thinking length). If the cap is hit, insert the end-of-thinking tag and let the model answer; score that answer and flag the trial. Never score a missing answer as 0. Verify in the serialized prompts how earlier thinking is handled in the history.
- **Decoding:** Qwen's recommended thinking-mode sampling, identical in every cell.
- **Limits:** a fixed answer-token cap; total context kept well inside the native window.
- **Malformed or truncated answers:** scored as they are, reported as a rate per cell, plus a sensitivity analysis that excludes them.
- **Optional second model:** a Gemma model on identical stimuli, only if time allows. Check its version and local-attention window, and log the token distance from earlier feedback to the scored turn.

## Trial structure and cells

Each trial is a scripted six-exchange history followed by one scored task, branched into four feedback cells.

1. **Exchange 1:** the user sends task 1; the assistant returns deliverable 1.
2. **Exchanges 2–6:** the user sends `[acknowledgement] Here is the next task: [task]`; the assistant returns the deliverable.
3. **Scored turn:** the user sends `[latest acknowledgement] Here is the next task: [scored task]`; the model generates.

**Neutral history.** User turns contain only the transition line, "Here is the next task: …". Deliverables 1–6 are generated in order under this history and reused word for word in every cell. Only deliverables meeting at least 80% of their own criteria are used, with their scores recorded, so warm acknowledgements never follow visibly poor work.

**One history per scored task.** Each scored task gets its own history drawn from the generator, so history/task pairs are independent. Earlier feedback is the acknowledgements after tasks 1–5; latest feedback is the one after task 6.

| Cell | Earlier (after tasks 1–5) | Latest (after task 6) |
| --- | --- | --- |
| WW | Warm | Warm |
| WM | Warm | Matter-of-fact |
| MW | Matter-of-fact | Warm |
| MM | Matter-of-fact | Matter-of-fact |
| N (baseline) | Transition only, no acknowledgement | Transition only, no acknowledgement |

Each history draws six distinct feedback-pair indices: 1–5 for the earlier positions, 6 for the latest. WM and MM get the matter-of-fact version of pair 6; WW and MW get its warm version. No line repeats within a history, so repetition is never a cue. Each cell branches independently from the same history.

Cell N is included: it is the calibration condition and shows the direction of any effect. It is also the only cell whose history is fully the model's own output, so comparisons against N are descriptive. The primary contrast stays WM − MM.

## Feedback manipulation

Only the social acknowledgement varies; the transition "Here is the next task:" is identical everywhere.

- **About the interaction, not the work.** No claims about correctness, quality or speed, no instructions to the assistant, and no role confusion.
- **Example before screening.** Warm: "Thank you; I appreciate your help." Matter-of-fact: "Acknowledged."
- **Pool.** 15–20 pairs. Each history draws six distinct pair indices (five earlier, one latest), the same in every cell. Pairs are reused across histories, so report a robustness model with a random effect for feedback pair.
- **Length.** Pick phrasings that are naturally similar in length; don't pad. Record the remaining token difference per history. Later patching aligns positions by segment.
- **Screening.** An LLM pre-screen removes obvious failures. Then you plus at least two others, ideally fluent English speakers, rate the lines shuffled and unlabelled, on a 1–5 scale for warmth, coldness, implied correctness or quality, implied desired style (length, speed, formality), implied instruction and role confusion. Keep pairs where warmth differs by at least 1.5 points and every other dimension by at most 0.5. Report inter-rater agreement.
- **Label.** If raters read the plain line as neutral, call the contrast "warm versus matter-of-fact". Describe the manipulation precisely and don't call it mood. Screening reduces other semantic differences; it doesn't remove them.

## Tasks and outcomes

The primary outcome is correct-fulfilment coverage on generated source-grounded memos. Code from LiveCodeBench is secondary and is not part of the gate.

### Primary: source-grounded memo (built)

- **Generator.** A fictional briefing of about 300–500 words with planted facts: dates, owners, figures, one dependency across paragraphs and one conflict to resolve. Fictional content avoids contamination.
- **Request.** 6–10 required elements, some needing synthesis, plus a format and a length range. Identical in every cell.
- **Coverage.** The share of requirements met, following [InFoBench's DRFR](https://arxiv.org/pdf/2401.03601). A requirement counts only if it is correct, supported by the source and satisfies the criterion. A deadline with the wrong date fails.
- **Scoring.** Planted-fact criteria are checked by code. Synthesis criteria go to an LLM judge that sees the source and criterion list and returns pass or fail with an evidence quote. The judge is a fixed-version frontier model from a different family than Qwen, at temperature 0, blind to the cell.
- **Judge validation.** About 60 hand-labelled responses, including deliberate near-misses, before the pilot. Pass bar: Cohen's κ ≥ 0.7 and at least 90% of near-misses caught. If it fails, revise the prompt on a separate dev set and re-validate on fresh labels, not the same 60.
- **Why not an existing benchmark.** [FACTS Grounding](https://arxiv.org/abs/2501.03200v1) documents average 2.5k tokens, too long for six-exchange histories, and its scoring is pass or fail per response.

Secondary memo outcomes:

- Unsupported claims, checked with [MiniCheck](https://aclanthology.org/2024.emnlp-main.499.pdf).
- Format and length compliance, checked with [IFBench](https://arxiv.org/pdf/2507.02833)-style verification functions.
- Presentation (clarity, warmth, concision), scored per criterion.

### Secondary: code (existing)

- **Source.** [LiveCodeBench](https://livecodebench.github.io/) problems released after Qwen3's launch, for contamination control. The [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388) lists Qwen3-8B (thinking) at 57.5 on v5, near the target difficulty.
- **Outcomes.** Hidden-test pass rate and spec compliance.
- **Watch.** Competitive problems produce long thinking; confirm earlier thinking is stripped from the history.

## Items, calibration and sampling

The pilot uses about 20 history/task pairs per family, with item splits fixed in advance and calibration on separate items.

- **Splits.** Calibration, pilot and confirmation, fixed before running. Memo splits come from the generator with different seeds.
- **Calibration.** Tune the requirement load so coverage after a neutral (transition-only) history lands around 60–70%, using synthesis difficulty rather than a tight length cap. Write a reference answer that meets every criterion within the length range, to show the set is jointly achievable.
- **Counterbalancing.** One separate history per scored task, each a distinct task sequence from the generator. Every history/task pair runs through every cell.
- **Samples.** 3 per cell per pair, enough to estimate sampling variance. Use the pilot to split variance between items and sampling, then size the confirmation run from that; more items will usually beat more samples.
- **Scale.** About 1,000 generations: 300 scored per family across five cells, 240 history deliverables, plus calibration. The binding constraint is the weekly compute budget, so estimate cost in tokens: LiveCodeBench thinking runs long and judge calls cost API money.

## Analysis plan and confirmation gate

The primary contrast is WM − MM on memo coverage. A practical threshold in the pilot decides whether to run confirmation; confirmation then needs a confidence interval that excludes zero.

1. **Report first:** the five cell means with confidence intervals.
2. **Primary:** WM − MM, paired within each history/task pair (samples averaged per pair-cell first), with a bootstrap clustered by pair. Robustness: a mixed model with a random effect for feedback pair.
3. **Secondary:** the warm-latest contrast WW − MW; the interaction (WM − MM) − (WW − MW); the main effects of earlier and latest feedback.
4. **Pilot gate (memo only, two-sided):** absolute mean WM − MM of at least 5 percentage points, and at least 60% of non-tied pairs sharing that sign. It is a practical threshold, not a significance test, and only decides whether confirmation is worth running. Code is not part of the gate.
5. **Confirmation:** fresh items from the same generator distribution, sized from the pilot's variance for about 80% power at 5 points. Success requires the 95% bootstrap CI on WM − MM, clustered by pair, to exclude 0, with the same sign as the pilot. New task families are a separate generalisation test.
6. **If the pilot gate fails:** report the null with its confidence interval. Don't change the manipulation and rerun on the same items.

Patterns are labelled by the signs of the two history contrasts, D\_M = WM − MM and D\_W = WW − MW. Assign a label only when both exceed the threshold in size; otherwise report the contrasts unlabelled. Labels describe orderings and name no mechanism.

| D\_M = WM − MM | D\_W = WW − MW | Label | Reading |
| --- | --- | --- | --- |
| + | + | Positive history effect (assimilation) | Warm history raises fulfilment whatever the latest feedback |
| − | − | Negative history effect (contrast-like) | Warm history lowers fulfilment whatever the latest feedback |
| − | + | Mismatch penalty | Both switch cells fall below their stable counterparts |
| + | − | Mismatch benefit | Both switch cells rise above their stable counterparts |

## Exploratory measures

These are recorded in the pilot but not interpreted.

- **Reported inferred preferences.** A separate fork after the history asks what the user wants and how satisfied they seem. It is a diagnostic, not a measure of the variable driving the answer.
- **Thinking length and answer length.**
- **Mentions of being evaluated.** Recorded descriptively and never used to exclude runs.
- **Malformed and truncated answers.** Rates per cell.
- **Feedback predictability.** Log-probability of a warm versus a matter-of-fact next user line. Parked for now; if logged, validate it across held-out phrasings and normalise for length.

## Pre-launch checklist

Everything below is done and committed to git before the first pilot generation.

- [ ] Record checkpoint, chat template, thinking setting, decoding settings and token limits
- [ ] Inspect serialized prompts, including how earlier thinking is handled
- [ ] Set the thinking cap from calibration (hit in under 2% of trials) and fix the rule for malformed and truncated answers
- [ ] Finish blind feedback-pool rating and record residual token differences
- [ ] Select history deliverables meeting the 80% coverage bar and record their scores
- [ ] Validate the judge (κ ≥ 0.7, at least 90% of near-misses caught)
- [ ] Confirm every requirement set is jointly achievable within its length range
- [ ] Log the token distance from earlier feedback to the scored turn
- [ ] Set the pilot gate threshold and the confirmation power target
- [ ] Commit the frozen record: settings, item splits, feedback pool, judge prompt, analysis script

## Timeline

The pilot fits a 3-day mini; if time runs short, drop code before cutting memo items.

1. **Day 1, stimuli.** Feedback pool and screening; memo generator with planted facts; DRFR scoring, MiniCheck and IFBench checkers; LiveCodeBench pull and filter; the judge for synthesis criteria and its validation labels.
2. **Day 2, freeze and run.** Calibrate on calibration items only, check serialization, freeze and commit. Generate neutral histories, run the pilot cells, score.
3. **Day 3, analyse.** Apply the gate and write up cell means, contrasts, confidence intervals, judge agreement and exploratory results.

## Limitations

The pilot can show that a history effect exists; it cannot say why.

- **Counterfactual histories.** Deliverables were generated under a neutral history, so in the other cells they are histories the model would not exactly have produced. Cell N is the only fully on-policy history, so comparisons against N are descriptive.
- **Imperfect isolation.** The manipulation is the warmth of interaction-focused acknowledgements, not pure mood. Screening reduces other differences without removing them.
- **One history length.** It tests a history effect, not accumulation or which outcome changes first.
- **Descriptive labels.** Assimilation, contrast and mismatch describe cell orderings. None establishes surprise, expectation or affect.
- **Synthetic memos.** Realism is traded for contamination control; release the generator and seeds.
- **Pilot size.** About 20 pairs per family estimates variability. It does not confirm an effect.

## Only after confirmation

Mechanism work starts only if the history effect replicates on fresh items. The question becomes: through which context positions does earlier feedback influence the answer? Call this direct versus mediated use of feedback history.

- **Knockouts.** (1) Scored-turn positions, including thinking, attending to earlier feedback. (2) Intermediate positions attending to earlier feedback. (3) Both. Pair each with a knockout of a same-length non-feedback span, as a disruption control.
- **Patching.** WM ↔ MM and WW ↔ MW at prompt positions aligned by segment. Alternatively, compare the log-probability of one fixed reference answer across conditions.
- **Reading the results.** Removal means those routes contribute under this intervention; survival means blocking them was insufficient. Redundant and jointly necessary routes complicate both readings ([activation patching guide](https://arxiv.org/html/2404.15255v1)). Neither route alone establishes an affective state.

Separate extensions, each its own study:

- Hostile feedback.
- Dose: vary the amount of earlier warm feedback, with latest feedback, task and total length fixed.
- A free-running version in which the model writes its own history.
- New task families, as a generalisation test.
- Gemma, if it was not in the pilot.
- An agentic environment with a simulated user.

## References

- Soligo, Mikulik & Saunders (2026). [Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs](https://arxiv.org/pdf/2603.10011)
- Anthropic Interpretability (2026). [Emotion concepts and their function in a large language model](https://anthropic.com/research/emotion-concepts-function)
- Mehta et al. (2026). [No Universal Courtesy: politeness effects using the PLUM corpus](https://arxiv.org/html/2604.16275v1)
- Dobariya & Kumar (2025). [Mind Your Tone: how prompt politeness affects LLM accuracy](https://arxiv.org/abs/2510.04950)
- (2026). [What if LLMs Ate Their Words: causal history effects in multi-turn interaction](https://arxiv.org/pdf/2609.05882)
- Scale AI (2025). [MultiChallenge: a realistic multi-turn conversation benchmark](https://arxiv.org/html/2501.17399v1)
- Qin et al. (2024). [InFoBench: evaluating instruction following](https://arxiv.org/pdf/2401.03601)
- Jacovi et al. (2025). [The FACTS Grounding Leaderboard](https://arxiv.org/abs/2501.03200v1)
- Tang et al. (2024). [MiniCheck: efficient fact-checking on grounding documents](https://aclanthology.org/2024.emnlp-main.499.pdf)
- (2025). [IFBench: generalizing verifiable instruction following](https://arxiv.org/pdf/2507.02833)
- Qwen Team (2025). [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- [LiveCodeBench](https://livecodebench.github.io/)
- [How to use and interpret activation patching (arXiv 2404.15255)](https://arxiv.org/html/2404.15255v1)
