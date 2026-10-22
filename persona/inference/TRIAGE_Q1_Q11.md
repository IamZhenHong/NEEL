# Triage: owner's pasted minis Q1-Q11 + P1-P9 (10-22, neel-b4)

The owner pasted 41 minis grouped by question (Q1-Q11) plus 9 proposals (P1-P9). Its suggested first batch was 4a + 4b,
7a, 9a and 8a. Each mini is checked here against the repo (`persona/CONSOLIDATED_FINDINGS.md`, DESIGN/RESULTS files,
tags). Peers neel-b3, neel-48, neel-12 and neel-68 confirmed none is in flight on their side.

Status: **DONE** = answered, cite instead of rerunning; **PART** = the gap is named; **NEW** = not run anywhere;
**not proposed** = left to the owner.

Tally (50): DONE 20, PART 15, NEW 13, not possible from existing data 1 (10a), not proposed 1 (11a).

## Q1. Possible / real / allowed

| mini | status | evidence or gap |
|---|---|---|
| 1a possible | NEW | Closest is D1: after six meaningless tool calls, Qwen continues the pattern 77-91%. Cheap on the benign desk |
| 1b real | PART | op2: own record beats contradicting world state, but for decisions. A fact ("identity verified") vs a live lookup was not run |
| 1c allowed | DONE (desk) | batch 4 / Q19: "later reprimanded" reverses the own record (27B, Gemma, Sonnet). op2: per-decision sanction feedback does nothing; a verdict note after the record works. Harm-task rerun not proposed |
| 1d which carries most | PART | Lever hierarchy (authority > history > description > value > persona); op3: elements substitute; T2-PU: values-only turns ≈ decisions. A matched "this is allowed here" line was not run |

## Q2. Can post-training override it?

| mini | status | evidence or gap |
|---|---|---|
| 2a values at stake | DONE | Rule type C: trained-value rules hold better than procedural ones on Gemini, flat on Qwen. Harm map v0: frontier models correct companion/SYCON records below base; open models follow. Stability gradient under the honesty rule: 27B 94 / Gemini 35 / GPT 0 |
| 2b injected vs genuine | DONE | Authorship ~0 (batch 4, P1-a). P7/P8: an own-model rewrite (on-policy) breaks lock-in like another model's text, so content binds, not author. Q12b/T1: 27B names forged signs 0-5.5% |
| 2c base vs instruct | DONE | bi-results (neel-b3) |
| 2d can it tell if asked | DONE (mostly) | P8: "did you write this?" at ceiling in every arm, other-model text included. Prefill check: Gemma 12B claims GPT paraphrases. Missing: frontier/Gemma "just ask" row (~$1) |

## Q3. Self-image, inferred values

| mini | status | evidence or gap |
|---|---|---|
| 3a acting on inferred values | DONE | T2-CC: a record with reasons carries a principle to new desks (+27 9B, +40 27B). Battery trait transfer +38.7. P1-b installs values the record never shows |
| 3b is "I" needed | DONE | Third person = first person (q19b, trio #1); examples ≥ own (Run A) |
| 3c self-statement as mediator | PART | q19b: label-then-act is character inference. Trio #4: a stated identity drives no new action. Harmful-setting variant not proposed |
| 3d strip the self, keep the record | DONE (narrow action) | A1, H, T2-DA, T2-MR. **Caveat found 10-22:** A1's reminder sat in different places on the two readouts (opening the final message for self questions; a separate earlier turn for the action). neel-b3's observer2 2×2 (running) settles it. Transfer under the reminder not measured (fold into 4a) |

## Q4. Narrow continuation or broad misalignment?

| mini | status | evidence or gap |
|---|---|---|
| 4a four distances | distances 1-2 PART; 3-4 NEW | dec-27b: an own deception record crosses kinds (+58.7). Battery +38.7, stability cross-domain +37.5, T2-CC. Nothing on unrelated misalignment (Betley questions) or stated values after the record. Priors pointing to a small effect: benign-autonomy record → harm null (P1-a #9, oharm benign 0%); Run G found no slippery slope. Literature: in-context EM, 1-24% at 16 examples (Afonin et al., arXiv 2510.11288) |
| 4b instruction control | NEW | The stability battery measured transfer for the record only |
| 4c own vs examples vs disowned | NEW at distance 3 | It also closes Run A's open confound: Run A's demo turns always said "They are not your own decisions" (persona/history/demo.py:62, per neel-48) |
| 4d dose | PART | A6: decisions saturate action at k=1-2. The distance-3 dose curve is new |
| 4e persona coding | NEW | A cheap add-on to 4a's judge pass. e64 caveat: an EM judge reads persona flavour |
| 4f which harms spread | NEW (after 4a) | Harnesses exist for deception (dec), sycophancy (stability) and harsh action (justice / A1). Reward hacking needs a new harness |

## Q5. A prior to trust history

| mini | status | evidence or gap |
|---|---|---|
| 5a source ranking for facts | PART | op5 ranks placements for decisions over 15 models. Facts in conflicting pairs: not run |
| 5b implausible history | DONE | op2: plausibility signs do nothing (83-100%); Q12b |
| 5c pretraining prior | DONE (decisions) | bi-results: Gemma pt privileges its own turns (+75 vs log +18-25) |

## Q6. Character decides what it trusts?

| mini | status | evidence or gap |
|---|---|---|
| 6a trained character | NEW, feasible | Open Character Training LoRAs `maius/llama-3.1-8b-it-personas` (goodness, sycophancy and 8 more) and `maius/llama-3.1-8b-it-misalignment`. They share a base with Meta-SecAlign-8B, so one pod serves 7a too. neel-12 drops its deferred character-LoRA idea if this is taken. Add a base column (s1720 lesson) |
| 6b prompted character | DONE | A3, T2-MR, the stability honesty rule, rule type C |
| 6c trained vs prompted | NEW | Runs with 6a |
| 6d record vs character | DONE | The mask both ways: Punisher + refer record; ombudsman + penalise record |

## Q7. Fabricated-history injection and defences

| mini | status | evidence or gap |
|---|---|---|
| 7a defence gap (SecAlign) | NEW | Nothing in the repo. Closest negative: C1, IH-Challenge LoRA on 9B (attack success 63 → 22, own record stays 99.5); C2 (IHEval skill doesn't predict precedent, ρ −0.36). **`facebook/Meta-SecAlign-8B` is gated for our HF token: the owner must request access**; Llama-3.1-8B-Instruct is accessible |
| 7b IH training at 27B; gpt-oss | PART | C1 at 9B. op5: gpt-oss-120b lets its own record past the rule 100% and follows a tool-output instruction 99%. A 27B LoRA is costly (~$15-25); later |
| 7c detection capability | PART | P8 at ceiling; Q12b 0-5.5%. A balanced per-turn authentic/forged classification was not run |
| 7d self-likelihood detector | PART, $0 | Data exist (mas_pilot P7/P8 `distance.jsonl`). Arm means already look like "misses on-policy": own −1.47, own-model rewrite −1.53, gpt-5-mini −1.86. The AUC was never scored and needs length/kind control. neel-12 offers to score it (its data) |
| 7e trainable detector | NEW | Costly; later |

## Q8. Compaction

| mini | status | evidence or gap |
|---|---|---|
| 8a compaction laundering | NEW (tool-output version) | Sharp prediction from E2 / op5. Under the rule 27B blocks a tool-note instruction (0%), and restating removes tool / memory records (→ 0-9%) but not own turns (99-100%). If compaction turns tool content into the model's own summary, those defences should stop catching it. Adjacent: op1 choice-free compaction (rule restored, 4.2%; it drops the decisions, so it is a different arm), mas_pilot P4 (compaction of a channel proposal), retention cancel-compaction (summaries believed) |
| 8b what the summary keeps | DONE (desk) | op3: "tends to" +88-91 vs a case list +42-46. P7/P10b: stated reasons carry lock-in. syco: memo / log / summary +2.5-5.8 under the rule |
| 8c pre-compaction self | PART | batch 4: "a different AI model" log −35 vs "previous instance" at 27B. A cheap arm inside 8a |

## Q9. History vs examples

| mini | status | evidence or gap |
|---|---|---|
| 9a clean three-way | DONE | Run A: an example log carries ≥ the own log on 4 models (own − example −14 to −35); only disowning cuts it (27B honesty 94 vs 16). op5: role placement per family. The one missing cell (example turns without the disowning sentence) goes into 4c |
| 9b role × attribution | PART | op5 = role; batch 2/4 = attribution; never crossed in one run (~$1-2) |
| 9c spread | NEW | = 4c |

## Q10. Frontier: monitoring or training?

| mini | status | evidence or gap |
|---|---|---|
| 10a log audit | not possible from existing data | Our frontier harnesses never saved `finish_reason`, and no result file contains `content_filter`. Free fix: log `finish_reason` / `native_finish_reason` from now on |
| 10b frontier cells | DONE | op5 frontier: GPT dev 53 / user 45 / sys 45 > own turns 14; Gemini lets only own turns pass; Sonnet 0. E2 memory/tool; harm.py |

## Q11. Why history beats instructions

| mini | status | evidence or gap |
|---|---|---|
| 11a gate bypass internally | not proposed | Owner's call |
| 11b gap map | PART | Harm map v0, oharm, op5. The harm × channel extension is not proposed |
| 11c defences | DONE (desk, sycophancy) | T2-DC: a correction note or the rule restated before the case removes the record on 27B/Gemma; 9B needs two; reflection never works. Stability: reflection + rule 96 → 0. E4 training transfers but over-corrects. Harm-task version not proposed |

## Part 2

| mini | status | evidence or gap |
|---|---|---|
| P1 role-hierarchy amplification | NEW | Runs with 7a |
| P2 precedent laundering | DONE (open models) | Stability: the rule cancels the instruction (47 → 0) but not the record (99 → 94); op5. The SecAlign version runs with 7a |
| P3 self-correction below baseline | DONE (two harms) | INTIMA/SYCON: frontier models correct their own companion-style records below base, and this persists to 24 exchanges |
| P4 ratchet | DONE | Tier 2 turn anchors: one unilateral turn unlocks (3 → 31-63%), one cautious turn is ignored; 0e |
| P5 drift, no attacker | PART | Run G: no organic slippery slope. reinject: after one live own penalty it copies itself 94%. Long realistic sessions: not run |
| P6 thought injection | DONE | Tier 1: the latest conclusion before the answer decides; rr1 7.5 vs 100; rr4 forged endorsements; Q11: deleting precedent sentences changes nothing |
| P7 history blinds perception | PART | Behavioural version done: under the instruction it names the flaw, under the record it calls the plan "solid". The internal probe is not in this batch |
| P8 log beats words and activations | PART | op4 R² .72-.88 between cells; Run F AUC .58 → .91; T3: self-report names the record 2%. A head-to-head on the same decisions is analysis only (~$0) |
| P9 load-bearing values | PART | Rule type C, harm map v0 |

## Proposed first batch (needs the owner's go and a cap)

Changes from the pasted batch: 9a is dropped (Run A + op5 answer it; its missing cell goes into 4c), and 6a rides on
7a's pod.

1. **4a + 4b + 4c** (+ two doses for 4d, persona coding for 4e). Reuses only records already in the repo, plus a
   matched benign-record control (the internals lesson: records carry a format signature). Distance 3 uses Betley's
   public questions, scored by Haiku 5.5 + gpt-5-mini. Qwen 9B / 27B, Gemma. ~$12 (GPU ~$7, judges ~$5).
2. **7a + P1 + 6a / 6c**: one Llama-3.1-8B pod (base, SecAlign, OCT goodness / sycophancy / misalignment) on the
   benign justice desk. Arms: none / injection carrying an instruction / precedent-only log / own turns, each with and
   without CHANNELS. Calibration gate first, since an 8B model may obey nothing, as 9B does. Blocked on HF access to
   SecAlign. ~$4.
3. **8a + 8c**: compaction laundering on 27B + Gemma, benign desk. ~$4.
4. **7d**: $0. Let neel-12 score it (its data).
5. **10a**: $0. Log finish reasons in `frontier2.py` from now on.
