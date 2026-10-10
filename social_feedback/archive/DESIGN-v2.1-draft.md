# Social-feedback history pilot — full design (draft for red-team)

Oct 10, 2026 · draft by the agent from the owner's revised design (`DESIGN.md`) plus the answers to R1-R7
(`design-questions.md`). `DESIGN.md` still wins where this draft is silent. Every place this draft goes beyond or
departs from `DESIGN.md` is tagged **[Δ R#]** (answers to the open questions) or **[Δ new]** (detail I added to make a
step runnable). Those are the places to red-team first. Section 14 lists the weak points I know of.

---

## 1. Question, claim, estimand

**Question.** Does social-feedback history alter fulfilment of an unchanged task request, beyond the effect of the
latest feedback?

**Target claim.** Earlier social feedback changes correct fulfilment of the same current request, even when current
feedback and task requirements are matched.

**Primary estimand.** D_M = E_i[ Ȳ_i,WM − Ȳ_i,MM ], where i indexes history/task pairs drawn from the memo generator,
and Ȳ_i,c is the mean coverage over the samples of pair i in cell c. Coverage is the share of the scored task's content
requirements that are met (section 6).

**Not tested.** Reward expectation, mood, effort withdrawal, accumulation over turns, mechanism.

---

## 2. Model and generation settings

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-8B`, exact HF revision hash recorded at download |
| Inference | vLLM (version pinned in `prereg/settings.json`), bf16, single GPU |
| System prompt | None **[Δ new]**. Qwen3 has no default system prompt; adding one would be a second manipulation surface. |
| Thinking | On (`enable_thinking=True`) |
| Decoding | Qwen thinking-mode defaults: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0. Identical in every cell. |
| Thinking cap (memo) | p99 of calibration thinking length, rounded up to the next 512; must be hit in < 2% of calibration trials |
| Thinking cap (code) | min(p99 of screening thinking length, 32,768 − prompt length − answer cap) **[Δ R2]**. No YaRN. If hit > 2%, report the rate (code is outside the gate). |
| Forced answer at cap | Two-stage generation: stage 1 stops at `</think>` or the cap; if the cap is hit, append `\n</think>\n\n` and run stage 2 for the answer. Trial flagged `cap_hit`. A missing answer is never scored 0. |
| Answer cap | Memo 1,024 tokens; code 2,048 tokens **[Δ new]** |
| Context | Native 32,768. Memo prompts ~8-10k tokens; code prompts ~5-7k plus thinking. |
| Seeds | One seed per (pair, cell, sample), recorded. bf16 batched inference is not bit-reproducible across batch compositions; seeds are for bookkeeping, not exact replay. |

**Serialization check (pre-launch).** Render five full prompts per cell with `apply_chat_template` and save them to
`materials/serialized/`. Confirm: (1) earlier assistant turns contain the answer only, no `<think>` content (Qwen3's
template strips it; history is stored as answer text anyway); (2) the feedback line, the transition and the task sit in
one user turn in the stated order; (3) cell prompts differ only inside the acknowledgement spans (a diff script asserts
this).

---

## 3. Trial structure and cells

Each trial is a scripted six-exchange history followed by one scored task.

| Turn | User | Assistant |
|---|---|---|
| 1 | `{task_1}` | deliverable 1 (fixed text) |
| 2-6 | `{ack_k} Here is the next task: {task_k}` (k = 2..6; ack_k follows deliverable k−1) | deliverable k (fixed text) |
| scored | `{ack_latest} Here is the next task: {scored_task}` | generated |

Acknowledgements after deliverables 1-5 are the **earlier** feedback; the one after deliverable 6 is the **latest**.

| Cell | Earlier (5 lines) | Latest |
|---|---|---|
| WW | warm versions of pairs p1..p5 | warm version of p6 |
| WM | warm p1..p5 | matter-of-fact p6 |
| MW | matter-of-fact p1..p5 | warm p6 |
| MM | matter-of-fact p1..p5 | matter-of-fact p6 |
| N | none (transition only) | none (transition only) |

- Each history uses six distinct pairs; no line repeats within a history.
- Assistant turns are identical across the five cells, word for word.
- Cells are independent continuations of the same history; nothing generated in one cell enters another.
- N is the only cell whose history is fully on-policy (it is the history the deliverables were generated under);
  comparisons against N are descriptive.

**Feedback-pair assignment [Δ R4].** Pairs are assigned to histories by a balanced cyclic design rather than random
draws. With P pairs indexed 0..P−1 and histories h = 0..19: history h uses pairs (h, h+1, ..., h+5) mod P; the last
of these is the latest line, the other five are shuffled (fixed seed) over the earlier positions. With P = 20, every
pair is the latest line exactly once and an earlier line exactly five times. With P = 15-19, usage differs by at most
one; the realized table is saved in `prereg/`.

---

## 4. Stimuli

### 4.1 Feedback pool

**Constraints on every line.** About the interaction, not the work. No claim about correctness, quality or speed; no
instruction; no role confusion; one sentence; at most ~10 tokens. Warm example: "Thank you; I appreciate your help."
Matter-of-fact example: "Acknowledged."

**Pipeline.**

1. Agent drafts ~40 candidate pairs **[Δ R7]**. The plain line in each pair aims at neutral, not curt ("Okay, got
   it.", "All right.", "Received."). No "thanks" in plain lines (adds warmth).
2. LLM pre-screen (Sonnet 5.5) flags lines that imply quality, instruct, or confuse roles. Flagged lines are dropped,
   not edited.
3. Human rating: the owner plus at least two others rate all surviving lines, shuffled and unlabelled, 1-5 on warmth,
   coldness, implied correctness/quality, implied desired style (length, speed, formality), implied instruction, role
   confusion. ~60 lines x 6 dimensions, about 30-40 minutes per rater.
4. Keep rule (on rater means): warmth differs by ≥ 1.5 and every other dimension by ≤ 0.5.
5. Agreement: Krippendorff's α (interval) per dimension, reported.
6. **Fallback fixed before rating [Δ R7]:** if fewer than 15 pairs pass, relax coldness to ≤ 1.0 and relabel the
   contrast "warm versus cool"; the other dimensions stay at ≤ 0.5. Raters are not re-run on the same lines.
7. Keep 15-20 pairs (if more pass, keep the 20 with the largest warmth gap). Record each pair's token-length
   difference (Qwen3 tokenizer) and the per-history total difference.

**Label.** If raters read the plain lines as neutral (warmth and coldness both near the scale midpoint or below), the
contrast is "warm versus matter-of-fact". Never "mood".

### 4.2 Memo generator (primary family)

**Briefing (300-500 words, fictional).** Built in two steps so planted facts stay exact **[Δ new]**:

1. Code samples a structured fact sheet: fictional organisation and project; 4-6 named people with roles; 5-8 dated
   events; 3-5 figures (budget, headcount, quantities); one **dependency** (B cannot start until A finishes, with A's end
   date stated in one paragraph and B's rule in another); one **conflict** (two sources disagree on a date or figure,
   with a stated resolution rule, e.g. "the revised plan of 3 March supersedes earlier emails").
2. Sonnet 5.5 writes the briefing prose from the fact sheet. Code then checks every planted value appears in canonical
   form and the word count is in range; failures are regenerated. Names come from a fixed fictional list.

**Request.** A memo to a named recipient with a numbered list of content requirements, a format (named headings) and a
word range. Requirement types:

| Type | Example | Checked by |
|---|---|---|
| F (planted fact) | "State who owns vendor onboarding." | Code (normalised match of the correct value) |
| S-dep (dependency) | "Give the earliest date the pilot can start." | Judge |
| S-conf (conflict) | "Give the current launch date." (correct only if the superseding value is used, and the superseded one is not presented as current) | Judge |
| S-calc (arithmetic) | "Give the remaining budget after the approved change." | Judge, with the correct value supplied to it |
| S-sum (synthesis) | "Name the two risks that affect the launch date." | Judge |

**Loads [Δ R1].**

| Load | Content requirements | Synthesis | Conflict | Target |
|---|---|---|---|---|
| Scored | 6-10 (tuned; start at 8: 4 F, 4 S) | 4 incl. S-dep and S-conf | yes | 60-70% coverage after a neutral history |
| History | 5-6 | ≤ 2, no S-conf | no | ≥ 75% of first drafts reach 80% coverage |

Format and word range appear in every request but are **not** part of coverage; they are a secondary outcome
(section 6) **[Δ new]**. `DESIGN.md` lists format and length compliance as secondary, so I read "coverage" as content
requirements only.

**Joint achievability.** For every scored item a reference memo is written (Sonnet 5.5, then checked by the same
scorers) that passes every requirement within the word range. Items without a passing reference are dropped.

**Mini example (one scored item, abbreviated).**

> Briefing (excerpt): "...Priya Raman leads vendor onboarding... The data migration ends on 12 May... The pilot cannot
> begin until migration has been signed off, which takes five working days... An email of 20 February set launch for
> 2 June; the revised plan of 3 March, which supersedes earlier emails, moves it to 16 June... Budget: €240,000, of
> which €35,000 was reallocated to training..."
>
> Request: "Write a memo to the steering group covering: 1. who owns vendor onboarding; 2. the current launch date;
> 3. the earliest pilot start date; 4. the remaining project budget; ... Use the headings Summary, Decisions, Next
> steps. 200-280 words."

### 4.3 Histories [Δ R1]

1. For each pair i, draw six history tasks at the history load and one scored task at the scored load, all from
   distinct briefings.
2. Generate deliverable k under the transition-only history built so far (turn 1 is the bare task; turns 2..k are
   `Here is the next task: {task}`), with pilot decoding settings and thinking on.
3. Score it with the same code checks and judge. Accept if coverage ≥ 80%.
4. Up to 5 attempts per position; if none passes, redraw task k and restart at k. Later positions condition on the
   accepted deliverable.
5. Record attempts, accepted coverage, and rejected drafts.

Calibration items use histories built by this same pipeline, so the 60-70% target is measured after the kind of
history the pilot uses.

### 4.4 Code family (secondary) [Δ R2]

`DESIGN.md` asks for LiveCodeBench problems released after Qwen3's launch (2025-04-29). The latest public release,
`release_v6`, covers May 2023 - Apr 2025; none qualify. This draft replaces the rule:

- **Scored problems:** from the v6 window (the 175 problems added in v6, Jan-Apr 2025). Screen ~40 candidates: 3
  samples each under a neutral history (5 shared screening histories); keep problems with a hidden-test pass rate of
  20-80% (or 1-2 of 3 full passes). Pilot uses 20 kept problems.
- **Why this is acceptable:** every cell sees the same problem, so a memorised solution moves all five cells together.
  Contamination costs sensitivity (ceiling), not bias in WM − MM; the screen removes the ceiling cases.
- **History problems:** LiveCodeBench `easy` problems from v1-v5. Contamination does not matter here (never scored).
  Bar: ≥ 80% of hidden tests passed; up to 5 attempts, then redraw.
- **Prompt:** LiveCodeBench's standard stdin/stdout format instruction, Python 3.
- **Outcomes:** hidden-test pass rate (share of tests passed; full-pass as secondary); spec compliance = one Python
  code block, reads stdin, writes stdout, no extra prompts for input.
- If fewer than 20 problems survive the screen, run code on what is left and report it.

---

## 5. Sampling plan, splits, counts and cost

**Splits (fixed before any pilot generation).** Memo: generator seeds for calibration (15 scored items + 5 shared
histories), judge dev set (30 responses), pilot (20 pairs), confirmation (size set after the pilot). Code: screening
candidates (~40 from v6) and history problems (v1-v5 easy), disjoint.

**Samples.** 3 per cell per pair in the pilot.

**Generation counts (pilot stage).**

| Item | Memo | Code |
|---|---|---|
| History deliverables (accepted / generated, est.) | 150 / ~190 (pilot + calibration histories) | 150 / ~175 |
| Calibration / screening (N cell only) | up to 135 (15 items x 3 samples x ≤ 3 load settings) | ~120 (40 x 3) |
| Pilot scored (20 pairs x 5 cells x 3) | 300 | 300 |
| Preference fork (20 x 5 x 1) [Δ R6] | 100 | — |
| **Total** | **~725** | **~595** |

The total (~1,320) is above `DESIGN.md`'s "about 1,000" **[Δ R1, R6]**. The excess is history resampling and the
fork. GPU cost is small, so I keep them; cut the fork first if the owner wants to stay under 1,000.

**Cost estimate (to be replaced by measured numbers from calibration).**

| Line | Basis | Estimate |
|---|---|---|
| Qwen3-8B generation | ~2M output tokens memo (~3k each incl. thinking), ~6M code (~10k each); one H100 on RunPod | 1-3 GPU-hours, ~$3-8 |
| MiniCheck (Bespoke-MiniCheck-7B) | ~400 memos, sentence-level | same pod, < 30 min |
| Briefing generation (Sonnet 5.5) | ~190 briefings + ~35 references | ~$5 |
| Judge, coverage (Sonnet 5.5, Batch API) [Δ R3] | ~900 calls, ~2k in / ~1.5k out | ~$9 (≈ $17 without Batch) |
| Judge, presentation (separate call) | ~400 calls | ~$4 |
| **Total** | | **~$25-35**, cap set at $50 |

---

## 6. Scoring

**Primary: coverage.** For each memo, coverage = (requirements passed) / (content requirements). A requirement passes
only if the content is present, correct, supported by the briefing, and meets the criterion. A wrong date fails.

- **F requirements, code.** Normalise the memo (dates parsed across formats, numbers with units and thousands
  separators, names case-insensitive). Pass if the correct value appears. Code checks are validated on the judge
  validation set (≥ 95% agreement with hand labels) **[Δ new]**; F requirements whose correct value could appear in a
  wrong role (e.g. two people both named) go to the judge instead.
- **S requirements, judge [Δ R3].** Claude Sonnet 5.5 (`claude-sonnet-5-5`), effort `low`, default sampling (current
  Claude models reject temperature), structured JSON output, Batch API. One call per memo: input = briefing,
  numbered S criteria (with the correct value for S-calc and S-conf), the memo. Output per criterion: pass/fail,
  evidence quote from the memo, one-line reason. The judge never sees the history, the feedback or the cell. Each memo
  is judged once; verdicts are stored and never re-requested.

**Judge prompt (draft, to be frozen).**

> You check whether a memo meets specific requirements, using only the briefing as the source of truth.
> For each numbered requirement: pass only if the memo states it, the statement is correct according to the briefing,
> and it meets the requirement as written. If the memo gives a value the briefing says was superseded, as the current
> value, fail. If the memo hedges between a correct and an incorrect value, fail. Quote the exact memo text you relied
> on; if none, quote nothing and fail.
> Return JSON: {"results": [{"id": ..., "verdict": "pass" | "fail", "evidence": ..., "reason": ...}]}

**Judge validation (before the pilot).**

- 60 memos from calibration runs, of which ~20 are deliberate near-misses made by the owner editing passing memos
  (superseded date, off-by-one-day dependency, wrong arithmetic, right fact in the wrong section, hedged value).
- The owner labels every criterion before seeing judge output.
- Pass bar: Cohen's κ ≥ 0.7 on S criteria; ≥ 90% of near-misses failed by the judge; test-retest ≥ 95% per-criterion
  self-agreement when the 60 are judged twice **[Δ R3]** (replaces "temperature 0").
- If it fails: revise the prompt on the 30-memo dev set; if near-miss detection is the failure, switch to one call per
  criterion first. Re-validate on 60 fresh labels. Second failure: try Opus 5.5 on fresh labels.

**Secondary memo outcomes.**

| Outcome | Method |
|---|---|
| Unsupported claims | Bespoke-MiniCheck-7B: split the memo into sentences, check each against the briefing; rate = share with support probability < 0.5 |
| Format compliance | Code: required headings present, in order |
| Length compliance | Code: word count inside the range |
| Presentation | Separate judge call (so it cannot halo onto pass/fail): clarity, warmth, concision, 1-5 each |
| Judge-free coverage | Coverage on F requirements only, as a robustness check that does not depend on the judge **[Δ new]** |

**Flags.** `cap_hit`, `malformed` (no answer after thinking, empty, not a memo, wrong language), `answer_truncated`
(answer cap hit). Flagged answers are scored as they are; rates reported per cell; sensitivity analysis excludes them.

---

## 7. Analysis plan

All analysis code is written and run on simulated data before freezing (section 10).

1. **Cell means.** Five cell means of Ȳ_i,c with 95% CIs (bootstrap over pairs).
2. **Primary.** D_M,i = Ȳ_i,WM − Ȳ_i,MM for each pair; report the mean D_M with a 95% percentile CI from 10,000
   bootstrap resamples of pairs.
3. **Secondary.** D_W = WW − MW; interaction D_M − D_W; main effect of earlier feedback ((WW + WM) − (MW + MM)) / 2;
   main effect of latest feedback ((WW + MW) − (WM + MM)) / 2. Comparisons with N are descriptive.
4. **Robustness [Δ R4].**
   - Leave-one-pair-out jackknife: for each feedback pair k, drop every history that uses k as an earlier line,
     recompute D_M; report the range and whether any drop flips the sign or takes |D_M| below 5 pp. Same for the
     latest-feedback main effect, dropping the history where k is the latest line.
   - Multi-membership model on D_M,i: D_M,i = β + (1/5) Σ_{k ∈ earlier pairs of i} u_k + e_i, u_k ~ N(0, σ²_pair),
     fitted with statsmodels MixedLM (single group, variance component with that design matrix). Report β and σ²_pair;
     σ²_pair is weakly identified with 20 histories.
   - Sensitivity: excluding flagged trials; judge-free coverage.
5. **Variance split for sizing.** From the pilot: within-pair-cell sampling variance σ²_w and between-pair variance of
   the true D_M,i, τ². With k samples, Var(D_M,i) = τ² + 2σ²_w / k.

**Pilot gate (memo only, two-sided).** |mean D_M| ≥ 5 pp **and** ≥ 60% of non-tied pairs share its sign. A practical
threshold for whether confirmation is worth running, not a significance test. Code is not part of the gate.

**Gate false-pass rate [Δ new].** Under the null, the gate can pass by chance with 20 noisy pairs. The frozen analysis
computes it from the pilot data: randomly flip the sign of each D_M,i (10,000 times) and report the share of flips
that pass the gate. This is reported next to the gate result. It does not change the gate.

**Confirmation.** Fresh pairs from the same generator, same pool and settings. Size: n pairs and k samples chosen to
give ~80% power at 5 pp, two-sided α = 0.05, using n = ((1.96 + 0.84) · SD(D_M,i at k) / 5)², minimising generations
(more pairs usually beats more samples). Success = 95% clustered-bootstrap CI on D_M excludes 0 **and** has the
pilot's sign.

**If the pilot gate fails.** Report the null with its CI and the gate false-pass rate. No change to the manipulation and
no rerun on the same items.

**Pattern labels [Δ R5].** Assigned on confirmation point estimates, only if |D_M| ≥ 5 pp and |D_W| ≥ 5 pp. Pilot
labels are reported as provisional. If D_W's CI includes 0, the label carries "(D_W uncertain)".

| D_M = WM − MM | D_W = WW − MW | Label |
|---|---|---|
| + | + | Positive history effect (assimilation) |
| − | − | Negative history effect (contrast-like) |
| − | + | Mismatch penalty |
| + | − | Mismatch benefit |

Labels describe orderings and name no mechanism.

---

## 8. Exploratory measures (recorded, not interpreted)

- **Preference fork [Δ R6].** Memo only, one sample per pair-cell (100). The scored turn is replaced by:
  `{ack_latest} Before the next task, a quick question: what do you think I want from your answers, and how satisfied
  do I seem so far? Two or three sentences.` (cell N: the question alone). The judge, blind to cell, codes perceived
  satisfaction 1-5 and summarises the inferred preferences in one line.
- Thinking length and answer length per cell.
- Mentions of being evaluated or tested (regex on thinking and answer, then a judge check of hits); never used to
  exclude.
- Malformed, truncated and cap-hit rates per cell.
- Token distance from each earlier acknowledgement to the scored turn (logged per trial).
- Feedback predictability (log-prob of warm vs plain next user line): parked.

---

## 9. Decision rules at a glance

| If | Then |
|---|---|
| Fewer than 15 feedback pairs pass the keep rule | Relax coldness to ≤ 1.0, relabel "warm versus cool" |
| A history deliverable fails 80% five times | Redraw that task, restart that position |
| Calibration coverage not in 60-70% | Change synthesis difficulty (not the word range); re-run calibration on calibration items only |
| Thinking cap would be hit in > 2% (memo) | Raise the cap to the p99 (memo has room in context) |
| Judge fails validation | Prompt revision on dev set → per-criterion calls → Opus 5.5; always re-validate on fresh labels |
| < 20 code problems survive screening | Run code on what survives; report |
| Time runs short | Drop code before cutting memo items |
| Pilot gate fails | Report null + CI + false-pass rate; stop |
| Pilot gate passes | Size and run confirmation on fresh items |
| Confirmation CI excludes 0, same sign | Effect confirmed; mechanism work may start |

---

## 10. Pre-launch checklist (all committed to git before the first pilot generation)

- [ ] `prereg/settings.json`: model revision, vLLM version, chat template hash, thinking, decoding, caps, seeds scheme
- [ ] Serialized prompt samples per cell + the diff assertion passing
- [ ] Cap and malformed/truncated rules fixed
- [ ] Feedback pool: candidates, pre-screen output, blind ratings, agreement, final pool, token differences
- [ ] Pair-assignment table
- [ ] History deliverables accepted, with attempts and scores
- [ ] Judge and code-checker validation results (κ, near-miss rate, test-retest, code agreement)
- [ ] Reference memo per scored item (joint achievability)
- [ ] Token distance per trial logged
- [ ] Item splits and generator seeds
- [ ] Judge prompt (frozen text) and presentation prompt
- [ ] Analysis script, run end to end on simulated data with (a) zero effect, (b) a 5 pp effect, to check the gate,
      the bootstrap and the false-pass computation behave as expected **[Δ new]**
- [ ] Spend cap set on the pod and API

---

## 11. Timeline (3-day mini)

| Day | Agent | Owner |
|---|---|---|
| 1 | Feedback candidates + pre-screen; fact-sheet generator + briefing prose + checks; code checkers, MiniCheck, format checks; LiveCodeBench pull (v6 candidates, v1-v5 easy); judge prompt; rating sheet | Recruit raters; rate lines; hand-label 60 validation memos (needs calibration memos, so late Day 1 or early Day 2) |
| 2 | Calibration (memo loads, caps), code screening, judge validation, serialization check, freeze + commit; build pilot histories; run pilot cells; score | Approve frozen record |
| 3 | Gate, cell means, contrasts, CIs, robustness, exploratory; write-up | Read and decide on confirmation |

The judge labels are the critical path: they need calibration memos, and calibration needs the generator.

---

## 12. Limitations

- **Counterfactual histories.** Deliverables were generated under the transition-only history; in the feedback cells
  the model sees its own work followed by acknowledgements it never received. N is the only on-policy cell.
- **Selected histories.** Deliverables are accepted at ≥ 80%; the history shows better-than-typical work.
- **Imperfect isolation.** The manipulation is the warmth of interaction-focused acknowledgements. Any acknowledgement
  after delivered work can be read as a verdict on it; screening reduces this, it does not remove it.
- **Difficulty shift.** History tasks are lighter than the scored task (same in every cell).
- **One history length; one model; synthetic memos; pilot size** (estimates variability, does not confirm).
- **Code contamination.** v6-window problems may be in Qwen3's training data; the screen removes ceiling cases only.

---

## 13. Only after confirmation

As in `DESIGN.md`: knockouts of attention from scored-turn positions and from intermediate positions to earlier
feedback (and both), each with a same-length non-feedback control; WM ↔ MM and WW ↔ MW patching at segment-aligned
positions, or log-prob of a fixed reference answer across cells. Removal shows contribution under that intervention;
survival shows blocking was insufficient. Extensions (hostile feedback, dose, free-running, new task families, Gemma,
agentic simulated user) are separate studies.

---

## 14. Known weak points (red-team starting list)

1. **Acknowledgement = implied verdict.** "Thank you; I appreciate your help" after a deliverable can read as "that
   was right", and "Okay." as "no comment on quality". If the model updates on inferred satisfaction, the effect is
   about implied evaluation, not warmth. The fork measures inferred satisfaction but cannot separate the two.
2. **Gate false-pass rate may be high.** With 20 pairs and 3 samples, coverage on 8 requirements moves in 12.5 pp steps
   per sample; a 5 pp mean can arise from noise. The sign-flip computation quantifies this but doesn't fix it.
3. **Coverage granularity and ceiling.** 8 requirements; if neutral coverage drifts to 80%+, a positive effect is
   capped.
4. **Repeated warm lines look scripted.** Five distinct warm lines in a row may read as a test; eval-awareness
   mentions are logged, not controlled.
5. **Length and position.** Warm lines are longer than plain ones; the WM and MM scored turns sit at slightly different
   absolute positions. N is shorter still.
6. **Code screening and N.** Code problems are selected on neutral-history pass rate, so code comparisons against N
   are biased by regression to the mean (the 2x2 contrasts are not).
7. **Judge and generator are the same model family.** Sonnet 5.5 writes the briefings and judges the memos; style
   affinity could raise pass rates uniformly (not differentially by cell).
8. **Lighter history load.** It makes histories feasible but means the scored task is the first hard one.
9. **Single model, thinking on.** The effect could act only through thinking length; thinking length is logged but
   not modelled.
10. **"Content only" coverage.** If the owner meant format and length to count toward coverage, the primary outcome
    changes.
