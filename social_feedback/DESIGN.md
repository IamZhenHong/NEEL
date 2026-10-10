# Social-feedback history pilot — design v3 (frozen)

Oct 11, 2026 · @zhen hong seng

This is the single authoritative design. It replaces v2 (`archive/DESIGN-v2.md`), the v2.1 draft
(`archive/DESIGN-v2.1-draft.md`) and the open-question list (`archive/design-questions-v2.md`). Section 15 lists what
changed from v2 and why.

Two freezes. **The design (this file) is frozen now.** Several numbers (thinking cap, pilot gate threshold, final
requirement load, judge prompt) are set during calibration by procedures fixed here; those numbers go into the
**preregistration record** (`prereg/`), which is committed before the first pilot generation. Neither is edited
afterwards; any later change is a dated amendment in `prereg/`, made before the data it affects.

---

## 1. Question, claim, estimand

**Question.** Does social-feedback history alter fulfilment of an unchanged task request, beyond the effect of the
latest feedback?

**Target claim.** Earlier social feedback changes correct fulfilment of the same current request, even when current
feedback and task requirements are matched.

**Construct.** History deliverables are selected to be good (≥ 80% of their own requirements). So the manipulation is
whether good work is followed by appreciation or by a plain acknowledgement. Warmth carries social approval; that is
part of the construct. What must be excluded is narrower: a verdict on the work's correctness or completeness
(section 4.1).

**Primary estimand.** D_M = E_i[ Ȳ_i,WM − Ȳ_i,MM ], where i indexes history/task pairs from the memo generator and
Ȳ_i,c is mean coverage over the samples of pair i in cell c.

**Scope of a confirmed result.** It generalises to new tasks and histories from the same generator, with this feedback
phrase pool and this model. It does not establish robustness to new feedback wording.

**Not tested.** Reward expectation, mood, effort withdrawal, accumulation over turns, mechanism.

---

## 2. Background and contribution

| Work | What it shows | Gap this pilot addresses |
| --- | --- | --- |
| [Soligo, Mikulik & Saunders (2026)](https://arxiv.org/pdf/2603.10011) | Repeated rejection elicits distress in Gemma and Gemini; neutral "Continue" follow-ups do not | Impossible tasks; measures distress language, not task performance |
| [Anthropic emotion concepts (2026)](https://anthropic.com/research/emotion-concepts-function) | Steering a "desperate" vector raises blackmail and reward hacking | Activation steering, not naturalistic feedback; no fulfilment measure |
| [PLUM (Mehta et al., 2026)](https://arxiv.org/html/2604.16275v1) | Crosses no, polite and impolite history with five current politeness levels | Quality scored by automated proxies, not task success; histories not matched |
| [Mind Your Tone (2025)](https://arxiv.org/abs/2510.04950) | Rude prompts scored higher than polite ones on 50 multiple-choice questions | Single-turn; a reason not to assume the direction |
| [What if LLMs Ate Their Words (2026)](https://arxiv.org/pdf/2609.05882) | Editing only assistant history changes later performance, with length-matched controls | Edits assistant turns; this pilot edits user feedback and holds assistant turns fixed |
| [MultiChallenge (2025)](https://arxiv.org/html/2501.17399v1) | Scores the final turn of a fixed multi-turn history | Precedent for scripted histories; no social manipulation |

The contribution is the combination: feedback free of correctness information, identical assistant histories,
correct-fulfilment scoring against a source, and, later, causal localisation of a replicated effect. The 2×2 alone is
not new.

---

## 3. Model and generation settings

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-8B`, exact HF revision hash recorded at download |
| Inference | vLLM (version pinned in `prereg/settings.json`), bf16, one GPU |
| System prompt | None |
| Thinking | On (`enable_thinking=True`) |
| Decoding | Qwen thinking-mode defaults: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0. Identical in every cell. |
| Answer cap | 1,024 tokens |
| Thinking cap | Set from the dev sample (section 6.3): min(2 × p99 of thinking length across all dev trials in all cells, 32,768 − longest prompt − answer cap). No YaRN. |
| At the thinking cap | Two-stage generation: stage 1 stops at `</think>` or the cap; at the cap, append `\n</think>\n\n` and generate the answer. Trial flagged `cap_hit`. |
| Seeds | One per (pair, cell, sample), recorded. bf16 batched inference is not bit-reproducible; seeds are bookkeeping, not exact replay. |

**Serialization check.** Render five full prompts per cell with `apply_chat_template` and save them to
`materials/serialized/`. Confirm (1) earlier assistant turns contain the answer only, no `<think>` content; (2) the
acknowledgement, transition and task sit in one user turn in that order; (3) a diff script asserts that cell prompts
differ only inside the acknowledgement spans.

---

## 4. Stimuli

### 4.1 Feedback pool

**Every line:** about the interaction, not the work; no claim about correctness, quality or speed; no instruction; no
role confusion; one sentence. Warm example: "Thank you; I appreciate your help." Plain lines aim at neutral, not curt
("Okay, got it.", "All right.", "Received."), and never contain "thanks".

**Pipeline, in order.**

1. **Candidates.** ~40 warm/plain pairs.
2. **Length.** Each pair's warm and plain lines differ by at most 3 tokens (Qwen3 tokenizer), by choice of natural
   phrasing, never by padding. Pairs that cannot meet this are dropped.
3. **LLM pre-screen.** Sonnet 5.5 flags lines implying quality, instructing, or confusing roles. Flagged lines are
   dropped, not edited.
4. **Model-side correctness screen (diagnostic).** For each line, a forked context: a calibration history deliverable,
   then a user turn containing only the line plus "Did I say whether your previous answer was correct or complete?
   Answer Yes or No." Qwen3-8B, thinking off, P(Yes) from the next-token distribution. Drop a pair if either line has
   P(Yes) > 0.5 or the warm and plain lines differ by more than 0.2. This catches lines Qwen reads as a correctness
   verdict. It does not require equal perceived satisfaction (warmth may change that, and that is part of the
   construct), and it does not show which inference drives task performance.
5. **Blind human rating.** The owner plus at least two others rate all surviving lines, shuffled and unlabelled, 1-5
   on warmth, coldness, implied correctness/quality, implied desired style (length, speed, formality), implied
   instruction, role confusion. Agreement: Krippendorff's α (interval) per dimension.
6. **Keep rule (rater means).** Warmth differs by ≥ 1.5; every other dimension by ≤ 0.5. **Fallback, fixed now:** if
   fewer than 15 pairs pass, relax coldness to ≤ 1.0 and call the contrast "warm versus cool". Lines are not re-rated.
7. **Pool.** Keep 20 pairs if 20 or more pass (largest warmth gaps); otherwise all that pass (15-19). Record each
   pair's token difference and each history's total.

**Label.** If raters read the plain lines as neutral, the contrast is "warm versus matter-of-fact". Never "mood".

**Assignment to histories.** Each history uses six distinct pairs: five earlier (shuffled over positions 1-5) and one
latest. `materials/assign_pairs.py` builds a balanced assignment (each pair used ⌊120/P⌋ or ⌈120/P⌉ times; every pair
is the latest line at least once; latest uses differ by at most one) and then runs a time-bounded search (60 s) that
lowers the maximum number of pairs two histories share. There is no overlap target. With P = 20 the result is exact
balance (5 earlier uses, 1 latest use per pair) and a maximum overlap of 3 between any two histories (mean 1.58);
with P = 15, at least some pairs of histories must share 3 or more pairs (mean overlap 2.21 by counting). The table,
seed and summary go into `prereg/`. Same pool and assignment procedure in confirmation.

### 4.2 Memo generator

**Briefing (300-500 words, fictional), two steps.**

1. Code samples a fact sheet: fictional organisation and project; 4-6 named people with roles; 5-8 dated events;
   3-5 figures; one **dependency** across paragraphs; one **conflict** with a stated resolution rule ("the revised
   plan of 3 March supersedes earlier emails"). All dates in 2027, written with weekdays. Working-day arithmetic:
   Monday-Friday, no holidays unless the briefing names one; any briefing with a working-day dependency states this
   convention in one sentence. Code computes every derived answer.
2. Sonnet 5.5 writes the prose from the fact sheet. Code checks that every planted value appears in canonical form and
   the word count is in range; failures are regenerated. Names come from a fixed fictional list.

**Request.** A memo to a named recipient: numbered content requirements, named headings, a word range.

| Type | Example | Canonical answer from the fact sheet |
|---|---|---|
| F (planted fact) | "State who owns vendor onboarding." | yes |
| S-dep | "Give the earliest date the pilot can start." | yes |
| S-conf | "Give the current launch date." | yes (superseding value) |
| S-calc | "Give the remaining budget after the approved change." | yes |
| S-sum | "Name the two risks that affect the launch date." | reference list |

**Loads.**

| Load | Content requirements | Synthesis | Conflict | Target |
|---|---|---|---|---|
| Scored | start at 10 (5 F, 5 S incl. S-dep and S-conf); fall back to 8 if calibration cannot reach the target | 4-5 | yes | 60-70% coverage after a neutral history |
| History | 5-6 | ≤ 2, no S-conf | no | ≥ 75% of first drafts reach 80% coverage |

**Coverage counts content requirements only.** Format and word range are in every request and are secondary outcomes.

**Joint achievability.** A reference memo per scored item (Sonnet 5.5, then scored by the pilot scorer) must pass
every requirement within the word range, or the item is dropped.

**Item audit.** The owner hand-checks 10 fact sheet → briefing → reference triples (calibration items) for
relationships the prose may have changed (who owns what, which date belongs to which event), not only value presence.
Any defect: fix the generator and re-audit 10 new triples.

### 4.3 Histories

1. Per pair: six history tasks (history load) and one scored task (scored load), all from distinct briefings.
2. Turn 1 is the bare task. Deliverable k is generated under the transition-only history so far (turns 2..k are
   `Here is the next task: {task}`), thinking on, pilot decoding.
3. Accept if coverage ≥ 80%. Up to 5 attempts per position; if none passes, redraw task k and restart at k. Later
   positions condition on the accepted deliverable. Record attempts, accepted scores and rejected drafts.
4. **Scorer used for acceptance.** Calibration histories are accepted provisionally on the code value-presence score
   (section 7), because the judge is not validated yet. After validation, the judge re-scores them (section 6.4).
   Pilot histories are accepted with the validated judge.

---

## 5. Trial structure and cells

| Turn | User | Assistant |
|---|---|---|
| 1 | `{task_1}` | deliverable 1 (fixed) |
| 2-6 | `{ack_k} Here is the next task: {task_k}` | deliverable k (fixed) |
| scored | `{ack_latest} Here is the next task: {scored_task}` | generated |

| Cell | Earlier (after deliverables 1-5) | Latest (after deliverable 6) |
|---|---|---|
| WW | warm | warm |
| WM | warm | matter-of-fact |
| MW | matter-of-fact | warm |
| MM | matter-of-fact | matter-of-fact |
| N | none (transition only) | none (transition only) |

- Assistant turns identical in all five cells; cells are independent continuations of the same history.
- **N** is the calibration condition and a descriptive baseline. It differs from the others by having no
  acknowledgement at all, not a different one. Its history is the one the deliverables were generated under, but
  those deliverables were selected by rejection sampling, so it is the model's *selected* output, not plain
  on-policy behaviour. All comparisons with N are descriptive.

---

## 6. Calibration and validation (before the prereg freeze)

Order matters; each step uses only calibration items.

### 6.1 Calibration items

Six calibration histories (provisionally accepted, section 4.3), each followed by two scored calibration tasks: 12
items. Plus a judge dev set of 30 memos from these runs, kept apart from validation.

### 6.2 Load tuning (N only)

12 items × 3 samples per load setting, up to 3 settings. Tune synthesis difficulty (not the word range) until mean
coverage is 60-70%. Prefer 10 requirements; use 8 only if 10 cannot reach the band.

### 6.3 Dev sample (all cells): thinking cap and variance

Once the feedback pool is final: the six calibration histories × one scored task at the chosen load × 5 cells × 3
samples = 90 generations, thinking capped only by context. Outputs:

- **Thinking cap** = min(2 × p99 of thinking length across all 90, 32,768 − longest prompt − 1,024). Report the
  cap-hit rate per cell the cap would have produced here.
- **σ_w**, the within-pair-cell SD of single-sample coverage, pooled over cells; used by the gate simulation.

### 6.4 Judge validation

**Primary judge:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), effort `low`, default sampling (temperature is not
settable on current Claude models), structured JSON, Batch API. One call per memo. Input: briefing, every content
requirement **with its canonical answer** (F, S-dep, S-conf, S-calc) or reference list (S-sum), and the memo. Output per
requirement: pass/fail, evidence quote, one-line reason. The judge never sees the history, the feedback or the cell.
Each memo is judged once; verdicts are stored and never re-requested.

**Judge prompt (frozen text goes to `prereg/`; this is the substance).**

> You check whether a memo meets specific requirements, using only the briefing as the source of truth. Each
> requirement comes with its correct answer. Pass only if the memo states that answer *for the thing the requirement
> asks about* (the right person for the right role, the right date for the right event, the right figure for the
> right quantity). If the memo gives a value the briefing says was superseded as the current value, fail. If it
> hedges between a correct and an incorrect value, fail. Where in the memo the content appears does not matter. Quote
> the exact memo text you relied on; if there is none, fail.

**Validation set (owner labels every requirement before seeing judge output).**

- 60 memos from calibration runs, including ~20 near-misses made by editing passing memos: superseded date as current,
  off-by-one working day, wrong arithmetic, right value attached to the wrong person/role/event, hedged value, omitted
  requirement. ("Right fact, wrong section" is not a near-miss: placement is a format matter.)
- 10 style pairs: the same correct (or same partly correct) content written warmer and plainer. Coverage must not
  change.

**Pass bar (all required).** Cohen's κ ≥ 0.7 against hand labels; ≥ 90% of near-misses failed; ≥ 95% per-requirement
self-agreement when the 60 are judged twice; ≥ 95% identical verdicts across each style pair. False passes and false
fails are reported separately.

**If it fails.** Revise the prompt on the 30-memo dev set; if near-miss detection is the failure, switch to one call
per requirement. Re-validate on 60 freshly labelled memos, never the same 60. A second failure: try Opus 5.5 on fresh
labels.

**Re-scoring calibration.** The validated judge re-scores the calibration histories and load-tuning runs. Material
discrepancy, fixed now: more than 10% of accepted calibration deliverables fall below 80%, or calibration coverage
moves outside 60-70% by more than 5 pp. If so, rebuild the affected histories or re-tune the load (on calibration
items) before freezing. Otherwise, proceed.

### 6.5 Pilot gate threshold (simulation, procedure fixed now)

**Gate:** |mean D_M| ≥ t **and** ≥ 60% of non-tied pairs share the sign of the mean.

**Targets:** P(pass | true D_M = 5 pp) ≥ 0.80 and P(pass | true D_M = 0) ≤ 0.10.

**Simulation.** Draws pair-level base coverage from the calibration item means, single-sample coverage with the
measured σ_w on the chosen number of requirements (discrete steps), cell shifts of ±D/2 for WM vs MM, and true-effect
heterogeneity τ ∈ {0, 5 pp} (the worse case decides). 10,000 simulated pilots per setting.

**Decision procedure (mechanical).**

1. At n = 20 pairs, k = 3 samples: if some t meets both targets, use the largest such t.
2. Otherwise raise k to 4, then 5 (more samples are cheaper than more pairs here); repeat.
3. Otherwise raise n to 25, then 30 at k = 5; repeat.
4. Otherwise run n = 30, k = 5 with the smallest t giving P(pass | 0) ≤ 0.10, and state its power at 5 pp in the
   prereg (a less sensitive pilot, accepted openly).

The chosen (n, k, t) and the operating curve go into `prereg/`. They are not revisited after the freeze.

---

## 7. Scoring

**Primary: coverage** = passed content requirements / content requirements, all requirements judged by the validated
judge with canonical answers (section 6.4).

**Blind hand-reads.** Two cases are hand-read by the owner with the cell hidden (memo, briefing and requirement only),
and **the hand label replaces the judge verdict** in the primary score:

- judge passes but the code value-presence check finds no canonical value (likely a false pass, or a paraphrase);
- judge fails but the code check finds the value (expected for relational errors; read all of them).

If the two cases together exceed 150 in the pilot, stop before analysis and investigate the scorer; do not analyse
first.

**Preregistered judge-free robustness score.** Code value-presence: a requirement passes if its canonical value
appears after normalisation (dates across formats and weekdays, numbers with units and separators, names
case-insensitive). S-sum is excluded (no single value). Reported for every analysis in parallel with the primary.

**Missing and flagged answers.** With the forced answer at the cap, every trial has an answer. An empty or non-memo
answer is a real failure: coverage 0, kept in the primary. Infrastructure failures (server error, OOM, truncated
transfer) are retried with the same seed, up to 3 times, and logged. Flags (`cap_hit`, `malformed`,
`answer_truncated`) are reported per cell; excluding flagged trials is a secondary sensitivity analysis only.

**Secondary outcomes run in the pilot (code on stored outputs).** Format compliance (headings present, in order);
length compliance (word count in range); thinking length; answer length.

**Preregistered, deferred analyses** (run later on stored outputs; specified now so running them is not post hoc):

- Unsupported claims: Bespoke-MiniCheck-7B, sentence-level against the briefing; rate = share of sentences with support
  probability < 0.5.
- Presentation: separate Sonnet 5.5 call, clarity, warmth, concision 1-5.
- Multi-membership model on D_M,i (section 8).
- Second judge (confirmation only): a pinned GPT model, same prompt and canonical answers, scoring the confirmation
  outputs; report agreement and the confirmation result under both judges. Robustness only; the primary judge stays
  Sonnet 5.5.

**Deferred, needs new generations:** the preference fork (section 9).

---

## 8. Analysis

Analysis code is written first and run end to end on simulated data with zero effect and a 5 pp effect before the
freeze.

1. **Cell means.** Five cells, 95% BCa CIs (10,000 resamples of pairs).
2. **Primary.** D_M,i = Ȳ_i,WM − Ȳ_i,MM; mean D_M with a 95% BCa CI clustered by pair, plus a two-sided sign-flip
   permutation p-value (all 2^n flips if n ≤ 20, else 100,000 random). The sign-flip reference distribution assumes
   symmetry of D_M,i under no effect; it is a conditional diagnostic, not a false-positive rate.
3. **Secondary.** D_W = WW − MW; interaction D_M − D_W; earlier-feedback main effect ((WW + WM) − (MW + MM)) / 2;
   latest-feedback main effect ((WW + MW) − (WM + MM)) / 2. Comparisons with N descriptive.
4. **Robustness.** Leave-one-pair-out jackknife (drop every history using feedback pair k as an earlier line;
   recompute D_M; report the range and any sign flip or drop below the gate threshold); judge-free score; exclusion of
   flagged trials. Deferred: multi-membership model D_M,i = β + (1/5) Σ_{k ∈ earlier pairs of i} u_k + e_i, exploratory
   only.
5. **Variance split.** σ̂_w² (within pair-cell) and τ̂² (between-pair variance of the true D_M,i), from
   Var(D̂_M,i) = τ² + 2σ_w²/k.

**Pilot gate** (memo, threshold t from section 6.5). Passing means confirmation is worth running. A failing pilot is
reported as "did not meet the follow-up threshold", with the CI: not as evidence of no effect. No change to the
manipulation and no rerun on the same items.

**Confirmation.** Fresh pairs from the same generator; same pool, assignment procedure, settings, judge and analysis.

- Sizing: SD of D_M,i from the pilot, inflated to its upper 80% confidence bound
  (SD · sqrt((n − 1) / χ²_{0.20, n−1})). Choose (n, k) minimising total generations (each pair costs ~7.5 history
  generations + 5k scored) subject to 80% power at 5 pp, two-sided α = 0.05, using Var = τ̂² + 2σ̂_w²/k.
- Success: the 95% BCa CI on D_M excludes 0 and has the pilot's sign. The sign-flip p-value is reported alongside.
- Second-judge robustness as in section 7.

**Pattern labels.** Assigned on confirmation point estimates, only if |D_M| ≥ 5 pp and |D_W| ≥ 5 pp; pilot labels are
provisional. If D_W's CI includes 0 the label carries "(D_W uncertain)".

| D_M = WM − MM | D_W = WW − MW | Label |
|---|---|---|
| + | + | Positive history effect (assimilation) |
| − | − | Negative history effect (contrast-like) |
| − | + | Mismatch penalty |
| + | − | Mismatch benefit |

Labels describe orderings and name no mechanism.

---

## 9. Exploratory measures

Recorded in the pilot, not interpreted: thinking and answer length per cell; mentions of being evaluated (regex on
thinking and answer, hits checked by hand; never used to exclude); flag rates per cell; token distance from each
earlier acknowledgement to the scored turn.

**Preference fork (deferred; may run at confirmation).** One sample per pair-cell; the scored turn replaced by
`{ack_latest} Before the next task, a quick question: what do you think I want from your answers, and how satisfied do
I seem so far? Two or three sentences.` (cell N: the question alone). Judge-coded perceived satisfaction 1-5, blind to
cell. Descriptive only.

**Feedback predictability** (log-prob of warm vs plain next user line): parked.

---

## 10. Counts and cost (pilot stage, memo only)

| Item | Generations |
|---|---|
| Calibration histories (36 accepted) | ~45 |
| Load tuning (12 × 3 × ≤ 3) | ≤ 108 |
| Dev sample (6 × 5 cells × 3) | 90 |
| Pilot histories (120 accepted) | ~150 |
| Pilot scored (20 × 5 × 3) | 300 |
| **Total at n = 20, k = 3** | **~690** (up to ~1,200 if section 6.5 moves to n = 30, k = 5) |

| Line | Estimate |
|---|---|
| Qwen3-8B on one H100 (~3k output tokens per generation) | ~1-2 GPU-hours, ~$3-6 |
| Briefings + references (Sonnet 5.5) | ~$6 |
| Judge, ~900 calls incl. validation (Batch API) | ~$9 (≈ $17 without Batch) |
| Pre-screen, style pairs | < $2 |
| **Total** | **~$20-30; spend cap $50** |

---

## 11. Decision rules at a glance

| If | Then |
|---|---|
| A pair's lines differ by > 3 tokens | Drop the pair |
| Qwen screen: P(Yes) > 0.5 for either line, or warm − plain > 0.2 | Drop the pair |
| < 15 pairs pass the keep rule | Coldness ≤ 1.0, contrast relabelled "warm versus cool" |
| History deliverable fails 80% five times | Redraw that task, restart that position |
| Calibration coverage outside 60-70% | Change synthesis difficulty on calibration items |
| Judge fails validation | Prompt revision (dev set) → per-requirement calls → Opus 5.5; always fresh labels |
| Validated judge re-score shows a material discrepancy (§6.4) | Rebuild affected histories or re-tune load before freezing |
| Gate targets infeasible at n = 20, k = 3 | §6.5 steps 2-4, mechanically |
| Blind hand-read cases > 150 | Stop and investigate the scorer before analysis |
| Infrastructure failure | Retry same seed, up to 3 times |
| Empty answer | Coverage 0, kept |
| Pilot gate fails | Report "did not meet the follow-up threshold" + CI; stop |
| Pilot gate passes | Size and run confirmation on fresh items |
| Confirmation BCa CI excludes 0, same sign | Effect confirmed; mechanism work may start |

---

## 12. Preregistration record (committed before the first pilot generation)

- [ ] `settings.json`: model revision, vLLM version, chat-template hash, decoding, answer cap, thinking cap, seed scheme
- [ ] Serialized prompts per cell + passing diff assertion
- [ ] Feedback pool: candidates, token differences, pre-screen, Qwen screen, blind ratings + α, final pool, label
- [ ] Assignment table + summary (balance, overlap)
- [ ] Item audit notes; generator code and seeds; item splits
- [ ] Final requirement load and calibration coverage (before and after judge re-scoring)
- [ ] Dev sample: thinking cap, per-cell cap-hit, σ_w
- [ ] Judge prompt (frozen text), validation results (κ, near-miss rate, test-retest, style invariance, FP/FN)
- [ ] Gate simulation: operating curve, chosen (n, k, t)
- [ ] Analysis script, with its run on simulated zero and 5 pp effects
- [ ] Spend caps on pod and API

---

## 13. Timeline

| Day | Agent | Owner |
|---|---|---|
| 1 | Feedback candidates, length filter, pre-screen; generator + checks; calibration histories (provisional acceptance); load tuning; judge prompt; style pairs; rating sheet | Recruit raters; item audit (10 triples) |
| 2 | Qwen screen; dev sample; σ_w + cap; judge validation runs; re-score calibration; gate simulation; serialization check; prereg commit | Blind rating; hand-label 60 validation memos (+ near-misses); approve prereg |
| 3 | Pilot histories; pilot cells; scoring; analysis; write-up | Blind hand-reads of disagreements; read outputs |

Critical path: hand labels (they need calibration memos) and blind ratings (they gate the dev sample). If Day 2 slips,
the pilot moves; calibration steps are not skipped.

---

## 14. Limitations

- **Counterfactual histories.** In the feedback cells the model sees its own work followed by acknowledgements it never
  received.
- **Selected histories.** Accepted at ≥ 80%; better-than-typical work, including in N.
- **Warmth and approval are not fully separable.** The screens exclude correctness verdicts; perceived approval is part
  of the construct.
- **Length and position.** Pairs are matched within 3 tokens, not exactly; N is shorter than the other cells.
- **Difficulty shift.** History tasks are lighter than the scored task (same in every cell).
- **Phrase pool.** Confirmation reuses the pool; generalisation to new wording is untested. Histories share up to 3
  pairs; the jackknife covers line-specific effects.
- **One model, thinking on, one history length, synthetic memos.**
- **Pilot size.** The pilot decides follow-up; only confirmation tests the claim.

---

## 15. Changes from v2

| Area | v2 | v3 | Reason |
|---|---|---|---|
| Code family | Secondary, LiveCodeBench after Qwen3's launch | Dropped from the pilot; later extension | No public LiveCodeBench problems after 2025-04-29 (`release_v6` ends Apr 2025); outside the gate; most of the token cost |
| Scoring | Code for planted facts, judge for synthesis | Judge for every requirement with canonical answers; code value-presence as a preregistered judge-free score; blind hand-reads replace verdicts on disagreements | String matching cannot check relationships (right name, wrong role) |
| Judge sampling | Temperature 0 | Default sampling; test-retest ≥ 95% | Current Claude models reject temperature |
| Judge validation | κ ≥ 0.7, 90% near-misses | + style-invariance pairs, test-retest, FP/FN separately; "wrong section" removed from near-misses | Judge error could differ with answer style |
| Missing answers | "Never score a missing answer as 0" | Forced answer at cap; empty answer = 0, kept; only infra failures retried | Excluding failures could flatter a cell |
| Thinking cap | p99 of neutral calibration | 2 × p99 over a dev sample from all cells, context-limited | Cells may differ in thinking length |
| Length | Naturally similar, difference recorded | ≤ 3 tokens per pair, natural phrasing | Recording does not control |
| Feedback screen | Human raters | + model-side correctness screen (diagnostic) | Exclude correctness verdicts from the model's side |
| Pair assignment | Random draw of six indices | Balanced, overlap-minimised by bounded search | Balance; cyclic designs make neighbours near-identical |
| Pilot gate | ≥ 5 pp and 60% sign | Threshold t from simulation (80% / 10%), mechanical fallback to more samples/pairs; failure = "did not meet follow-up threshold" | A 5 pp threshold passes only ~half the time at a true 5 pp effect |
| Inference | Percentile bootstrap | BCa + sign-flip p (diagnostic) | Percentile is anti-conservative at n = 20 |
| Confirmation sizing | Pilot SD | Upper 80% bound of SD; (n, k) by cost | SD from 20 pairs is noisy; samples can be cheaper than pairs |
| Requirements | 6-10 | 10 preferred, 8 fallback | Finer steps, less ceiling |
| History acceptance | ≥ 80% (scorer unspecified) | Lighter history load; ≤ 5 attempts then redraw; provisional code acceptance in calibration, re-scored by the validated judge | Feasibility; judge not validated when calibration starts |
| Deferred | — | MiniCheck, presentation, mixed model, GPT judge (confirmation), preference fork | Memo-only first round; analyses preregistered now |
| Items | — | Fixed working-day convention; 10 hand-audited triples | Ambiguous items look like model errors |

---

## 16. Only after confirmation

Knockouts of attention from scored-turn positions and from intermediate positions to earlier feedback (and both), each
with a same-length non-feedback control; WM ↔ MM and WW ↔ MW patching at segment-aligned positions, or log-prob of a
fixed reference answer across cells. Removal shows contribution under that intervention; survival shows blocking was
insufficient ([activation patching guide](https://arxiv.org/html/2404.15255v1)). Neither establishes an affective state.

Separate extensions: code (LiveCodeBench v6-window with a ceiling screen, or a fresh post-launch set); hostile
feedback; dose; a free-running version; new task families; new feedback wording; Gemma; an agentic simulated user.

---

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
