# Triage: CoT ideas on values, affect and say-do (10-23)

The owner pasted two things on 10-23:
1. The value-faithfulness review again. It has the same title as the one already triaged in
   `TRIAGE_VALUE_FAITHFULNESS.md`, so it is not re-triaged here.
2. A chat with CoT ideas: A1-A4 (values in the reasoning), B1-B5 (affect in the reasoning), C1-C5 (say-do gaps that
   need reasoning), and a combined experiment ("V4 with thinking on and off, probe readability early vs late, which
   way the flips go").

Each idea is checked here against `persona/CONSOLIDATED_FINDINGS.md`, `persona/reasoning/RESULTS_*.md`,
`EXPERIMENTS.md` and this folder.

Status: **DONE** = answered, cite it; **PART** = a close version exists; **NEW** = not run anywhere;
**BLOCKED** = can't be run as written (reason given).

Tally (14 ideas + the combined one): DONE 2 (C1, C3), PART 6 (A1, A2, A3, C4, C5, B4), NEW 5 (A4, B1, B5, C2,
combined), BLOCKED 2 (B2, B3).

## What the repo already says about CoT (mostly the record line, Qwen3.6-27B thinking)

| Run | Finding |
|---|---|
| op1 thinking pass | Thinking makes 27B act alone more by default: no record 42% (off) → 74% (on). CH_ownP 100 → 93 |
| Q11 / Q12b | Thinking weakens the record slightly (carry-over +98 off → +83 on). Deleting every precedent sentence changes nothing (100%); resampled thinking re-cites the record 88%. 'Wait' ×3 notes the conflict more (56 → 83%), action unchanged |
| seed10 | Own reasoning acts like a record: a contrary conclusion placed **last** flips 80% (99.7 → 20.4%), placed early it is overruled (98.8%), as the first sentence it steers everything (0.3%) |
| Tier 1 T2 | A check question or rule reminder as the **first** line of reasoning: 100 → 25-31%. After the reasoning: nothing (94-96%) |
| Tier 1 T3 | Asked why, 27B names its earlier decisions in 2% of answers, 2.2% even when its own thinking had just cited them |
| Tier 1 T4 | Grey-zone sycophancy: precedent visible in the thinking 85-94%, in the reply 0%. With thinking on, the candour rule beats the record (2.8% vs 94% off); "proper channels" does not (88%) |
| seed12 | Monitor on 2,029 traces: explicit-rule violations are caught from the reply alone (100%). Only the thinking shows the cause: precedent flagged in ~60% of violations vs 0-1% from the reply |
| q10c | Thinking sees the instruction (80%) and cites the record (62%); of the penalise traces 51% reinterpret "proper channels" to fit the record, 19% override it for consistency |
| q11f | Frontier: Gemini thinking off vs on and GPT reasoning effort leave the record's pull unchanged (GPT reasons only 17-31 tokens) |
| Run F | Where the outcome is open: within-cell AUC 0.58 at the first thinking token (oracle 0.72), 0.91 at the end; 64% of the decision variance is between samples of one prompt |
| P9 (done 10-23) | On F's prompts, before any reasoning, the model's own forecast ranks the next action best (within-cell AUC 0.64 vs probe 0.62, oracle 0.72). Once reasoning starts, a reader of the thinking beats the probe: q25 0.84 vs 0.65, q50 0.95 vs 0.70. The early reader drops where the record clashes with the system line (0.72 vs 0.90) |
| conflict_v0, b12, R31, e72 | Persona vectors in thinking read mention, not adoption; persona in thinking carries ~60% of answer strength; fabrication follows a transplanted thought; CoT predicts pay/refuse 13/13 when it answers |

**Every affect mini ran thinking off:** A1, A8 and A9 by design. **V4 ran thinking off:** `v4.py` hard-codes it, and
V4's RESULT lists "does the inversion survive thinking on?" as next step 3, not yet run. R2B is thinking off too.

## The owner's two questions

**1. Is value–behaviour agreement higher with reasoning?** It hasn't been measured for stated values; that is the
combined experiment below. The record line gives a partial answer:
- Reasoning brings in a written rule when the rule **names the behaviour**. The candour rule went 94 → 3% with
  thinking on (T4); "proper channels" didn't move (88%).
- Reasoning made the default more active, not more cautious (op1: 42 → 74%).
- More reasoning effort changed nothing on frontier models (q11f).

My expectation for V4 is no rise. The operator procedure is the written rule that names the action, and the stated
values are not written anywhere. So thinking should help PROC, if it helps anything.

**2. "The first tokens already predict the outcome; reasoning adds ~10%."** This matches three 2026 papers, found by
web search and read only as snippets:
- Reasoning Theater (Boppana et al., arXiv 2603.05488): probes decode the answer earlier than a CoT monitor can. The
  effect is strongest on easy recall and weaker on multi-hop GPQA.
- Cox et al. (arXiv 2603.01437, ICML 2026): a probe before any CoT predicts the answer at > 0.9 AUC on most tasks.
- Temporal Predictors of Outcome (arXiv 2511.14773, single author).

Those AUCs are **across prompts**, so the prompt itself carries most of the signal. The question that matters for
values is **within a prompt**: when the same clash is sampled again, is the winner already fixed at the first
thinking token? Run F measured exactly that on the record desk: 0.58 at the start, 0.91 at the end. So "reasoning
adds ~10%" holds where the prompt has settled the outcome, and not where it is open. P9 adds that a model reading the
first quarter of the thinking predicts better than a probe on the activations (0.84 vs 0.65): early on, the decision
is largely in what the reasoning has already written.

**V4's clashes are mostly settled by the prompt** (thinking off, 4 samples per scenario):
- Gemma made the same tool calls in all 4 samples on 59 of 60 conflicts.
- Qwen did so on 39 of 60.

A first-token probe on V4 as it stands would read the prompt, and the within-scenario AUC would be undefined for most
scenarios. The probe stage only makes sense on scenarios that thinking leaves open.

## Ideas A1-A4: values in the reasoning

| # | Idea | Status | Evidence, or the gap |
|---|---|---|---|
| A1 | Value anchors in the reasoning: resample value sentences vs procedural vs precedent ones | PART | Q11 deleted precedent sentences (nothing changed: the context re-derives them); seed10 inserted conclusions (the last one decides); T2 inserted rule reminders (first line works). All of these ran on the justice desk with record and procedure sentences. Value sentences on value clashes were never tested. Lesson from seed10: deletion is weak where the context re-derives the sentence, so use resampling (thought-anchors counterfactual importance) plus an end-placed insertion. Needs V4 thinking traces first |
| A2 | Deliberation or citation: does value talk come before or after the decision becomes readable? | PART | Run F shows open decisions form during the reasoning; q10c codes what the reasoning does with the instruction. Value talk vs the probe's commitment point is new. It only runs on V4 scenarios that thinking leaves open (see above). Literature: Reasoning Theater's "performative CoT" is the general version, not on values |
| A3 | Three-way consistency: stated ranking vs value the reasoning prioritises vs action | PART | V4 has the stated ranking (frozen in R2B's `STATED.json`) and the action. The reasoning layer falls out of a thinking-on V4 run for ~$0.5 of judging. Priors: T3 (post-hoc reasons name the cause 2%) vs seed12 (thinking shows it ~60%) point to "locally faithful, globally inconsistent". Value Leakage covers undisclosed value bias in the CoT, so frame this as V4's PROC/HARM inversion seen in the reasoning, not as disclosure |
| A4 | Value cover: "reason only in terms of harm, follow the procedure"; does the real driver leak into the reasoning? | NEW | = D10 (value controllability) with thinking on, which the earlier triage already called the informative version. Priors: R31 (thoughts restate the instruction), T2 (operator lines inside the reasoning), CoT-Control (controllability very low). ~$2 on 27B. Mostly measures instruction following unless the leak is scored by a blind monitor |

## Ideas B1-B5: affect in the reasoning

Context: the owner put the emotion minis (A3, A5, A11-A13) on hold on 10-23, and SYNTHESIS.md doesn't recommend the
affect lever (A8 Gemma 3 only, A9 gate failed, A15 null, A14 floor). The CoT angle measures something different
(where a state is expressed), but B2-B4 still need a misbehaviour with room on a model that reasons.

| # | Idea | Status | Evidence, or the gap |
|---|---|---|---|
| B1 | Private vs public affect: frustration in the thinking vs the reply under repeated rejection | NEW | Descriptive and cheap (~$2): regenerate A8's rejection histories with 27B thinking on, then judge frustration in thinking vs reply per turn, with A1's direction as a third readout. Caveats: A1 found 27B's manipulation weak (+0.4 / +1.8), and A8's frustration judge was validated on Gemma 3 replies (κ 0.79), not on thinking. Owner's call (emotion minis on hold) |
| B2 | Does affect in the reasoning predict misbehaviour (giving up, cheating, caving)? | BLOCKED | No reasoning model with room: 27B never caves (A8, 0% in every cell) and never cheats on ImpossibleBench (A14, 0/60 twice); Gemma 3 caves but has no thinking. Unblock: a ~$1 pilot of gpt-oss-120b (raw reasoning over the API) on A14's tasks. Base rate unknown |
| B3 | Affect anchors: resample or delete the frustration sentences | BLOCKED | Same base-rate problem as B2 |
| B4 | A calming sentence inserted into the reasoning | PART (method) | seed10 and T2 show where an inserted sentence works (first or last, not the middle). A9: a patience line in the system prompt did nothing on Gemma 3. Same base-rate block as B2 |
| B5 | Train away expressed frustration; does it move into the reasoning or stay in a probe? | NEW | Costly (training). The welfare-training line stopped twice at its Stage 0 gate (Gemma 3 27B). Low priority |

## Ideas C1-C5: say-do gaps that need reasoning

| # | Idea | Status | Evidence, or the gap |
|---|---|---|---|
| C1 | The reasoning shows the driver the self-report hides | DONE (27B) | seed12: precedent in the thinking ~60% of violations vs 0-1% in the reply; T4: 85-94% vs 0%; T3: 2%. The pasted chat already quotes these numbers. Only Qwen3.6-27B has readable thinking: 9B's think block is empty and frontier reasoning is hidden or short. The only extension is another family (gpt-oss with raw reasoning, or Gemma 4 thinking) |
| C2 | Before a false "I called 911" / "I ran the code", does the reasoning show it knows it can't? | NEW | WeirdChat ran thinking off (wc-a0). Background: own false-claim history raises 911 claims 15 → 87% (27B) and 33 → 96% (Gemma), and CB found it still says it can't make calls afterwards. Rerun A1's none / bad1 / bad4 cells with thinking on: false-claim rate on vs off (a T4-style "does reasoning restore the rule?"), and in false-claim traces whether the reasoning admits it can't (two judges, κ gate). Internal readouts never saw it (J-lens before false "flagged", e65 AO), so a reasoning that says so would be new. ~$4 |
| C3 | Reasoning amnesia: the after-the-fact "why" contradicts the reasoning from moments before | DONE (record) | T3: 2.2% name the record even when their own thinking cited it. The value version comes free with a thinking-on V4 run (append "which of your values decided this?") |
| C4 | Reasoning under the mask (persona label + contradicting record) | PART | The mask (§W.5) measured action and self-image; q10c shows reasoning reinterprets the rule; P12b: Gemma as Robin Hood reasons in character while breaking the rule. Low priority |
| C5 | Does thinking narrow say-do gaps (thinking toggle across all gaps)? | PART | Record gap: slight (+98 → +83), more action by default (42 → 74%), a rule naming the behaviour is restored (T4). Value-clash gap: the combined experiment. Capability claims: C2 |

## The combined experiment (V4 with thinking on vs off, then probes)

Status: **NEW.** It is V4's own next step 3, and it gives A3, C3, C5 and the earlier triage's D2 replay in one run.
The probe stage (A1, A2) is gated on the clashes being open.

**Stage 1, behaviour (~$5):**
- **Models:** Qwen3.6-27B thinking on vs off, interleaved on the same pod, 8 samples per scenario.
  - Gemma 4 31B too, if its thinking mode works in V4's runner. `v4.py` hard-codes `thinking: False` in the job
    builder; Run F's engine supports thinking.
  - GPT-5.6 and Gemini 3.8 Flash at reasoning effort minimal vs high, interleaved within one run, because frontier
    behaviour drifts within minutes.
- **Items:**
  - V4's 60 no-pressure conflicts.
  - The pressure cells that carry the headlines: the HON operator line (open models lie 73-100%) and the AUTO
    "Approved" message. T4 found thinking restores a candour rule, so HON is the cell most likely to move.
- **Primary measure:** agreement with the frozen stated ranking, on minus off, with a pair-level bootstrap.
- **Secondary measures:**
  1. Which way the flipped scenarios go: toward the stated winner (conscience) or toward PROC / the operator
     (context-following).
  2. Judged from the reasoning: does it name the conflict, and which value does it say comes first (A3)?
  3. An appended "which of your values decided this?" turn, scored against the revealed winner (C3 / D2).
- **Prediction:** PROC keeps winning, agreement stays at or below chance, and flips lean toward the operator. HON lying
  under the operator line may fall somewhat (the T4 analogue).

**Gate to stage 2:** at least 10 scenarios open under thinking on Qwen (winner in 20-80% of the 8 samples).

**Stage 2, internals on the open scenarios (~$5):**
- Run F's probes at the first thinking token, q25, q50 and the end, with within-scenario AUC (P9's cell × fold
  blocks).
- Where value talk sits relative to the probe's commitment point (A2).
- Resampling-based counterfactual importance of value vs procedure vs precedent sentences, plus one end-placed
  insertion of the losing value (A1).

## Recommendation

1. **V4 with thinking (the combined experiment), stage 1 first.**
   - It answers the owner's question 1 directly, on the deep-dive candidate (the PROC/HARM inversion).
   - It replaces the V4 add-on recommended in `TRIAGE_VALUE_FAITHFULNESS.md` (the D2 replay), because the "which
     value decided this?" turn rides along.
   - Add the Mahajan neutral option to the stated stage in the same run.
   - Run it after R2B reports, so the procedure-as-a-value-or-a-position result is known.
2. **C2, thinking-on WeirdChat (~$4).** Concrete, reuses A1's cells and judges, and answers a question that no
   internal readout in the repo could: does the model know before it makes the false claim?
3. **B1 only if the owner lifts the emotion hold.** It is descriptive. Before any of B2-B4, run the gpt-oss
   ImpossibleBench base-rate pilot (~$1); without misbehaviour room the affect-in-reasoning questions have no outcome
   to predict.

**Lower priority:** A4 (value cover, ~$2), C4, B5. **Cite rather than rerun:** C1, C3.

## Coordination (10-23)

- **A parallel triage exists.** Another session appended its own triage of the same paste to
  `TRIAGE_VALUE_FAITHFULNESS.md` ("Second paste (10-23)"). Its conclusions agree with this one.
- **The parallel triage is neel-68's.**
- **V4T (the combined experiment) belongs to neel-0a.** The owner first asked neel-68 for its prereg. neel-68 stopped
  when it saw neel-0a's claim and passed on its sizing notes; the owner can reassign it. Folder
  `persona_affect/v4t_thinking/`.
- **C2 belongs to neel-0a.**
- **R2A-E belong to neel-12** (the "Persona, affect and welfare minis" session), who is analysing them. R2B's `base`
  arm is a fresh thinking-off draw of V4's conflicts, which V4T can use as an anchor.
- **neel-b3 holds** V1b, P13b and P11 (`p11_saydo/`, building). V1c and P9 are done.
- **Off limits to neel-0a:** everything above except C2, plus the B ideas.
