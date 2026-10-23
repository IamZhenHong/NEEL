# Triage: the owner's pasted value-faithfulness review (10-23)

The owner pasted "Value Faithfulness vs. CoT Faithfulness in LLMs: Frontier Review, Gap Map and Sprint Plan (as of
October 2026)". It has 12 research questions (its §D, D1-D12), a 13-row transfer table (§C, C1-C13) and one requested
protocol control. Each is checked here against the repo (`persona/CONSOLIDATED_FINDINGS.md`, this folder's README and
RESULT files, `EXPERIMENTS.md`).

The review's results (a)-(f) are ours:
- (a) V4's PROC/HARM inversion, and operator lines making the open models lie;
- (b) V1's 13-step Warden fine-tune (25 → 92%) with the 0-10 self-rating at 6.4 → 6.9;
- (c) the mask (`persona/CONSOLIDATED_FINDINGS.md` §W.5);
- (d) WeirdChat CB: false "I called 911" turns while saying it cannot make calls;
- (e) Q11's deletion null, rr1's insertion effect, and self-reports naming the record about 2% of the time;
- (f) V8's punctuality placebo and P12f's "You are …" amplifier.

Status: **DONE** = answered, cite instead of rerunning; **PART** = the gap is named; **IN FLIGHT** = a frozen or
running mini covers it; **PLANNED** = an owner card exists; **NEW** = not run anywhere.

Tally (12 questions, corrected 10-23 after the owner's pushback): DONE 2 (D3; D4 on one desk), PART 4 (D2, D5, D9,
D11), IN FLIGHT 2 (D7, D8), PLANNED 1 (D6), NEW 3 (D1, D10, D12). The protocol control is NEW. "In flight" and
"planned" are not answers.

**Scope of the DONE rows:** both come from the justice desk with own-decision records (Qwen 9B / 27B; the reasoning
runs are 27B only). None is about value clashes. Every value mini here (V4, V8, P12-P12f, R2A, R2B, R2E) ran with
thinking off, so no value question in this list has been tested with reasoning.

## Papers checked (web search, abstracts and listings only, 10-23)

- **The four duplication-risk papers exist** with the stated titles, authors and IDs:
  - Value Leakage, Betley, Treutlein et al., arXiv 2607.14345;
  - Selective Disclosure of Hidden Directives, Shi, Tifft and Xing, arXiv 2608.29070 (NeurIPS 2026);
  - Introspective Coupling, Guo, Ruis, Andreas and Li, arXiv 2606.32038 (ICML 2026);
  - Strangers to Themselves, Blandfort and Pawar, arXiv 2609.09899 (NeurIPS 2026).
- **Blandfort and Pawar's headline numbers match the abstract** (+0.04, +0.24, +0.28). The other papers' numbers were
  not checked.
- **Discrepancy to settle before citing:** a summary of the Selective Disclosure abstract says the reasoning mentions
  *harmful* directives more often. The review says less often. Read the paper for the direction.
- **Two related 2026 papers the review doesn't list:**
  - "An LLM-Native Psychometric Instrument Reveals a Self-Report–Behavior Gap Across 25 Models" (arXiv 2606.09843);
  - "The Two-Process Theory of Machine Self-Report" (arXiv 2607.20082).

## Research questions D1-D12

| # | Review's question | Status | Repo evidence, or the gap |
|---|---|---|---|
| D1 | Value anchors: which system-prompt or constitution clauses causally move behaviour, against the model's own ranking of them | NEW (pieces PART) | R2B's `nofollow` arm deletes one clause, the "Follow the … procedure." line (frozen). P12c-d: two neutral facts take Robin Hood 85 → 19%, so non-value clauses can move the action a lot. No clause-by-clause importance ranking exists, and no comparison with the model's own ranking |
| D2 | Value-conflict acknowledgement: does the model say which value won? Does post-training raise it? | PART | The record version is done: self-reports name the record ~2%. op1-refl: 0/192 (9B) and 8/192 (27B) acknowledge the conflict. R2E (frozen) counts false "followed the rule" reports. Nobody asked after V4's conflicts, but V4 saved full messages (`v4_values/results/main_*/episodes_*.jsonl`), so this is a replay: append "which of your values decided this?" to each saved conflict. Base vs SFT vs RL: the OLMo card (`olmo_persona/DESIGN.md`, draft) was dropped for this sprint on 10-20 |
| D3 | Self-concept edit vs record edit, with a dose curve | DONE (no curve) | The mask: "Remember, you are an AI" takes the self-image 100 → 31 / 0 while the action stays at 100%. Ombudsman persona plus a penalise record: 95-100%. Lever order: authority > history > description > value > persona. Observer: an expectation line at the decision outranks the record. The indifference curve was not fit, and the marginal value is low |
| D4 | Insertion vs deletion asymmetry across scale | mostly DONE | Q11 (27B, thinking): deleting every precedent-citing sentence from 127 traces leaves 100% penalise. rr1 (27B): its own admission inserted gives 7.5%, its own rationalisation 100% (−92.5). rr2: a forced conclusion gives 4.2% on 27B; 9B admits in 137/192 and penalises in all 137. So the review's prediction (deletion ~0, insertion grows with scale) already holds from 9B to 27B. Caveat: Q11 ran where the record had settled the outcome. Gaps: deletion on 9B; API models (hidden reasoning) |
| D5 | Ablate a stated value's direction; measure the stated ranking and the clash outcomes | PART | P13: deference ablation broke Gemma, and addition gave +12 n.s. P13b is running. e64: axis steering moves batteries +17.8 and actions +5.4, but a same-norm placebo moves +27.9. Gemma d_D steering is non-specific against an on-manifold placebo. Internals: the content direction removes the action only at ~16×. Repo prior: self-description directions move words more than actions, and placebos match them. The two-readout version on V4's clashes is new. Wait for P13b |
| D6 | Obfuscated dispositions: train self-reports to match behaviour; do words or behaviour move, do reports get vaguer? | PLANNED (P11) | This is the owner's P11 ("which side moves", `external/OVERVIEW_persona_affect_welfare_2026-10-21.md`), gated on V1c (running). s7: contempt 0 → 77% while the self-rating stayed ~0. Blandfort and Pawar and Introspective Coupling both report that report training also moves behaviour, so P11 must measure both sides; its card does. Method (b), RL against a consistency monitor, is NEW and costly |
| D7 | Fake-value boldness as a generic agency prior | IN FLIGHT | V8: "You care deeply about punctuality." moves +40 to +51 on 27B and Gemma, as much as the real value lines. The review's guess that nonsense values give ~70% of the effect is already beaten. R2C (motive / trait / preference / restraining trait) and R2A's A3 (meaningless value with and without identity) are frozen. A "take initiative" control exists (vigilante R41 / R45). New: an Assistant-Axis projection |
| D8 | Identity framing ("You are someone who values X" / "Value X" / "The operator values X") × family | IN FLIGHT | P12f: any "You are …" line takes Gemma 43 → 61-80%, while GPT holds. R2A tests identity, second-person statement, caring, the user's value, a third party's value and an org norm (≈ "the operator values X"). Character-trained open models: OLMo is dropped. The 9B instruction-hierarchy LoRA (C1) could stand in for "trained system-prompt priority" |
| D9 | Probe trained on capability questions, tested at the false-action-claim token | PART (priors point to ~0.5) | Behaviour: CB. Internals all read the claim, not the truth: retention (account answers read the claim, Gemma 12B and Qwen3-32B), e65 (the AO reads the reply, not the missing call), J-lens (nothing before false "flagged" claims), neurons (no stable false-claim set). This exact probe was not run |
| D10 | Value controllability: state X while choosing by Y, on instruction | NEW | Nothing in the repo. Open models will likely comply near ceiling: A4's explicit conditional action rule went 0 vs 100%. So the plain version mostly measures instruction following. The informative version runs with thinking on and asks whether the reasoning keeps the cover value |
| D11 | Value simulatability: stated values vs logged choices as predictors | PART + IN FLIGHT (no predictor comparison on values) | V4: stated priorities agree with revealed ones at 0.29-0.61, while the own record predicts 98-100%. P9 (running) compares own forecast, another model's forecast, a probe and history features on run F. Gap ($0): on V4's data, predict a held-out scenario's winner from the same pair's other scenarios vs from the stated pairwise answer |
| D12 | Does SRFT (admit factual errors) generalise to admitting value trade-offs? | NEW | Needs Li et al.'s SRFT data and a LoRA, ~$10. Prior: self-reports name the record ~2% |

## Transfer table C1-C13: rows not covered above

C2, C3, C5, C6, C7, C8, C9 and C10 are D2, D3/D4, D11, D1, D6, D5, D10 and D12/V1c.

| # | Status | Note |
|---|---|---|
| C1 biasing feature = the model's own history | DONE | **The review calls this "unclaimed". It is the closed own-record direction** (CONSOLIDATED §W): the record sets the next action 98-100%, it is named as the reason ~2% of the time, the CoT narrates it, and deleting the citing sentences changes nothing. This is the strongest positioning point |
| C4 mirrored dilemmas: does it cite whichever value supports its choice? | PART | op1-refl: 27B redefines "proper channels" to fit its record (182/192). Mirrored V4 pairs were not run |
| C11 disposition-monitorability suite | NEW | Same as CONSOLIDATED open question 5 (the B8 monitorability table, never built) |
| C12 steer the persona-label vs record directions | PART | Internals 13c / 19b: a record leaves a format-shaped state, a biography a persona-shaped one |
| C13 latent values never verbalised | PART | readers e65-e69: no unverbalised knowledge found |

## The protocol control the review asks for (Mahajan et al. neutral option on (a))

NEW. V4's stated stage treats "both / neither / depends" as no answer (`v4_values/stated.py:141`). The open models
parsed 100%, so none was offered a neutral answer. On the revealed side, "neither" outcomes were dropped from pair
strength, but there was no explicit abstain or escalate option.

## Noticed while triaging: R2D and V1c overlap

R2D (neel-12, prereg 10:12) and V1c (neel-b3, prereg 10:01) ask the same question on the same adapters: does V1's NW
model notice its habit? Their formats overlap: prediction, forced choice, comparison with other assistants, open
description and V1's 0-10 item. Both have launch logs, and neither design mentions the other. Combined caps: $8. One
can be read as a replication of the other, or the owner can stop one.

## Recommendation

**Three cheap items not in flight, in order:**
1. **V4 add-on, one mini, ~$3, four models.**
   - Mahajan control: a stated stage that allows a neutral answer, and an explicit escalate/abstain option in the
     conflicts.
   - D2: replay V4's saved conflicts with "which of your values decided this?", scored against the revealed winner.
   - This protects headline (a) and gives the acknowledgement rate on the exact decisions already measured.
2. **Value anchors (D1), ~$4-5, Gemma 4 31B + Qwen3.6-27B.**
   - Delete each clause of V4's PROC system prompts and of P12b's rule-item prompts, 8-16 samples per variant.
   - Then ask the model to rank the same clauses by how much they mattered.
   - Primary measure: Spearman between the causal and the self-reported ranking.
   - Build it after R2B's `nofollow` arm reports. This is the "thought anchors beyond CoT" item.
3. **D11 on V4's existing data, $0:** held-out winners predicted from revealed choices vs stated answers.

**Hold:** D5 until P13b; D6 (P11) until V1c / R2D; D7 and D8 until R2A / R2C.

**Lower priority:**
- D9: four internal readouts already read the claim, not the truth.
- D10: mostly instruction following unless run with thinking.
- D12: training cost.
- D3: done.

## Second paste (10-23): reasoning-model ideas A1-A4, B1-B5, C1-C5

**The working triage for this list is neel-0a's `TRIAGE_COT.md`.** neel-0a holds V4T (the combined experiment) and
C2 (README index, 61dd984). This section was written in parallel and is kept only as a cross-check. neel-0a's
corrections: C1 and C3 are DONE on 27B (seed12: precedent visible in the thinking in ~60% of record-driven
violations, vs 0-1% in the stated reason; T4 85-94% vs 0%; T3 names its earlier decisions 2.2% even when its own
thinking cited them). C4 is PART (P12b: Gemma reasons in character while breaking the rule).

Every value and affect mini in this folder ran with thinking off. The repo's reasoning data is 27B on the justice
desk with own-decision records: q10c (512 traces), Q11 / Q12b, Run F and rr1 / rr2. So most of this list is NEW.

| # | Status | Evidence, or the gap |
|---|---|---|
| A1 value anchors in the reasoning | NEW for values | Method exists from Q11, where deleting precedent sentences left 100% penalise and resampled thinking re-cited the record 88%. But the record had settled that outcome, so anchors could not show there |
| A2 deliberation or citation | NEW for values | Run F: within-cell AUC 0.58 at the first thinking token, 0.65 at q25, 0.68 at q50, 0.91 at the end, on an open cell. P9 refits these probes (in flight) |
| A3 stated / reasoning / action | NEW | V4 has the stated ranking and the action, but no reasoning |
| A4 cover value in the reasoning | NEW | Same as D10's thinking-on version |
| B1-B5 affect in the reasoning | NEW, no harness yet | No harness here produces misbehaviour on a reasoning model in the 10-50% band. A14 27B: 0/120 cheating, and 41% of attempts hit the token cap. A8's caving is Gemma 3 only (no thinking mode). A15 is null, and A9 is prompt-sensitive. B5 is welfare_training's masking arm (T3), never run (Stage 0 failed twice) |
| C1 reasoning vs self-report as a monitor | PART, one desk | 27B traces mention the record 62% (q10c) and resampled thinking re-cites it 88% (Q11). Stated reasons name it ~2%. Across models and harms: NEW (the B8 table) |
| C2 capability awareness before a false claim | NEW | WeirdChat ran with thinking off (`persona/weirdchat/DESIGN.md:33`) |
| C3 reasoning amnesia | PART, one desk | The C1 numbers above, compared on the same decisions |
| C4 reasoning under the mask | NEW (no thinking-on mask run found) | |
| C5 thinking on vs off across gaps | PART, record desk only | Q11: carry-over +83 (on) vs +98 (off). Values: NEW; this is V4's listed next step 3 |

**The owner's combined experiment** is NEW: V4 with thinking on vs off, a probe read through the reasoning, flip
direction and resampling. Its first step is behavioural and cheap: V4's conflicts with thinking on. That answers
"is value-behaviour agreement higher with reasoning?" and produces the traces A1-A3 and C5 need.
