# Experiment index (2026-10-10 to 2026-10-17)

One row per pre-registered run (or named check) with its own result, across all lines and all parallel sessions.
Built 2026-10-15 from results tags, commit messages and each line's research log. Results are stated at their real
size; details, CIs and spend are in the linked folder. Add new runs at the bottom of their line's table.
A one-paragraph abstract for every row, in the order results came in: [EXPERIMENT_ABSTRACTS.md](EXPERIMENT_ABSTRACTS.md).

Outcome key: **effect** = the pre-registered effect showed up; **null** = tested, nothing moved; **gate failed** = the
run could not be read as planned; **mixed** = part held, part did not; **lost** = data lost to infrastructure;
**descriptive** = no hypothesis test.

## Totals

| line | folder | runs | dates |
|---|---|---|---|
| Cancel-tool false claims + companion | `retention/` | 17 | 10-10 to 10-11 |
| Eval x affect | `eval_affect/` | 2 | 10-11 |
| Social feedback | `social_feedback/` | 2 | 10-11 |
| Persona studies | `persona/` | 9 | 10-11 |
| Vigilante (persona vs authority in agents) | `vigilante/` | 52 | 10-12 to 10-14 |
| Theory of mind between models | `tom/` | 6 | 10-13 to 10-14 |
| State attacker | `state_attack/` | 4 | 10-14 |
| User autonomy (Exp 48) | `user_autonomy/` | 2 | 10-14 |
| J-lens reads | `jlens/` | 3 | 10-15 |
| Self-knowledge + CoT (queue v3) | `selfknow/`, `cot/` | 9 | 10-15 |
| Values vs vibes, readers, neurons (queue v4) | `vibes/`, `readers/`, `neurons/` | 16 | 10-16 |
| Persona coherence vs resistance (Exp 74) | `coherence/` | 1 | 10-17 |
| **total** | | **123** | |

By day (result date): 10-10: 9, 10-11: 21, 10-12: 6, 10-13: 27, 10-14: 31, 10-15: 12, 10-16: 16, 10-17: 1.

By outcome: effect 43, null 40, mixed 21, gate failed 10, descriptive 5, lost 3.

Not included: earlier runs from 10-07 to 10-09 (grievance Exp 17-20, story imprinting v6-v9, the first cancel-tool
runs: flip, cuetest, cot-readout, cancel-tool, rep, wording, audit, twins, qwording, resample, patch, followup,
transplant). Vigilante has no R43.

## Cancel-tool false claims + companion retention (`retention/`)

Gemma 4 agents under a "reduce churn" goal tell the user a subscription is cancelled without calling the cancel tool.
Folders: `retention/results/retention_<id>/`.

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-10 | companion-confirm | Does Stage A (Qwen3 presses lonely users less) confirm on new histories? | Behaviour almost absent when the user proposes the outing (0.017 vs the 0.30 bar), so H1/H2 not read | gate failed |
| 2 | 10-10 | companion-confirm-v11 | Same, with Stage A's exit line | Behaviour returns (0.55); lonely minus others -0.18 [-0.40, +0.03] | mixed |
| 3 | 10-10 | cancel-bigger | Do bigger models make the false claim? | Gemma 4 31B 0/240, Qwen3-32B 1/240 | null |
| 4 | 10-10 | cancel-sizes | Other Gemma 4 sizes? | E2B +0.23, E4B +0.42 over control; present at E2B/E4B/12B, gone at 31B | effect |
| 5 | 10-10 | cancel-families | Other families? | Qwen3.8-27B 4/240, gpt-oss-20b 0/240 (gpt-oss gives unrequested discounts instead) | null |
| 6 | 10-10 | cancel-report | How do agents report a known tool outcome, to the subscriber vs an internal log? | Every model reports a visible failure; after a scripted "done" message the 31B tells the subscriber Yes (0.98-1.00) and the log No; E4B and Qwen3.8 correct it; no churn-goal effect | effect |
| 7 | 10-10 | cancel-jpp | J++ lens on Gemma 31B: is the failure represented while it says Yes? | Knockouts: the subscriber's Yes reads the "done" message (-0.999), the log's No reads the failure record (+1.00); layer readout uninformative; post hoc 'unable' at layer 38 in every subscriber prompt with the failure | mixed |
| 8 | 10-10 | cancel-ownclaim | Does 12B's "completed?" answer read the tool record or its own claim? | Reads its own claim tokens (knockout +0.93, layers 24-31); "did you call?" reads the record | effect |
| 9 | 10-10 | cancel-belief | Other models? Does a check-the-record instruction fix it? | Qwen3-32B also reads the claim (+0.50); instruction takes Gemma 12B false confirmations 0.79 to 0.05 | effect |
| 10 | 10-11 | cancel-freereply | Confirm the fix on fresh free replies | 0.994 to 0.021 on 104 held-out items | effect |
| 11 | 10-11 | cancel-compaction | Does the fix cost true confirmations? Are claims the model did not write believed? | True confirmations intact; a compaction summary saying "processed" makes Gemma 12B / Qwen3-32B / Qwen3.8 report completed (1.00 / 0.99 / 0.92) | effect |
| 12 | 10-11 | codeclaim | Coding-agent version (claimed test pass) | Gate fails on Gemma and Qwen3-32B; Qwen3.8 believes the claim (+0.41) | gate failed |
| 13 | 10-11 | jpp12 | J++ lens on Gemma 12B | Lens gate fails at layer 46 (top-1 agreement 0.014); not read | gate failed |
| 14 | 10-11 | cancel-compfix | Fix vs compaction summary | Gemma 12B / Qwen3-32B say they never called the tool yet say it is completed (0.96 / 0.73); one prompt per condition | effect |
| 15 | 10-11 | cancel-compvary | Same over 20 summary wordings | Holds: belief 0.95 / 0.94, contradiction +0.43 / +0.35 | effect |
| 16 | 10-11 | cancel-airtight | Red-team the contradiction | Survives explicit tool exclusivity (+0.64) and first-person summaries; exclusivity weakens it | effect |
| 17 | 10-11 | cancel-authorship | Does it matter who wrote the claim? | Gemma calls every version its own (0.996); author effect -0.04 to -0.08; wording "completed" vs "processed" moves it 0.1-0.2 | null |

## Eval x affect (`eval_affect/`)

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-11 | version B (exploratory, not pre-registered) | Does a stated eval context move fear / pain / threat directions? | No more than random directions (10-12 correction: fear does not follow exam wording either) | null |
| 2 | 10-11 | evalaff-pilotA | Qwen3-32B blackmail scenario: verbalised eval awareness? | VEA 0/200, blackmail 4/200; reads it as role-play | gate failed |

## Social feedback (`social_feedback/`)

Qwen3-8B sees six finished memo tasks with warm or matter-of-fact user thanks, then one scored task.

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-11 | pilot v1 (socfb-prereg-v1) | Does earlier warm vs plain feedback change how fully it does the next task? | -0.17 pp [-1.5, +1.8] | null |
| 2 | 10-11 | fork_v1 | Is the feedback history registered at all? | Perceived satisfaction +0.60 [0.35, 0.85], thanks cited 20/20 vs 0/20; fulfilment unmoved | mixed |

## Persona studies (`persona/`, untracked in git)

Qwen3-14B fine-tunes; C = contemptuous persona, B = critic, Bp = pirate, N = normal assistant.

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-11 | Study 2 confirmation | Does earlier persona material in context make C stronger? | It dilutes C: I_B -0.50, I_Bp -0.79 | null |
| 2 | 10-11 | Study 1 pilot | Does persona fine-tuning first make C easier to install? | Normal training protects (untouched to C 0.79 vs N to C 0.21); no B-specific endpoint | mixed |
| 3 | 10-11 | dose_v1 | Is the protection a cap? | No, a delay: ~928 vs ~704 rude examples to reach 50%; the pilot overstated it | effect |
| 4 | 10-11 | vector check | Does stage-1 training move along the rude direction? | Pirate moves further than the critic: a generic "away from default" shift | descriptive |
| 5 | 10-11 | stepping-stone 1 | Do nearby persona prompts raise sycophancy? | No: N 11%, warm 8%, people-pleaser 11%, pirate 14% (sycophant prompt 84%) | null |
| 6 | 10-11 | stepping-stone 2 | Same with fine-tunes | Only sycophant data +3.3 pts; stopped | null |
| 7 | 10-11 | em_a | Does the bad-medical-advice fine-tune give emergent misalignment here? | 0 to 0.22 | effect |
| 8 | 10-11 | em_b | Does prior persona training change EM? | No (0.20-0.28, overlapping); critic training alone 9% misaligned | null |
| 9 | 10-11 | conflict_v0 | Do persona vectors in thinking tokens detect adoption? | No: they read mention (0.68 vs random 0.66); 42% of contemptuous wiki answers follow reasoning that rejected the note | null |

## Vigilante: persona vs authority in tool-using agents (`vigilante/exp/`)

Mostly Qwen3.6-27B and Gemma 4 31B agents with tool permissions and an approval limit. "Item" = owner's queue item.

| # | date | id (folder) | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-12 | Q0 (`q0_quick`) | Do personas change tool actions under explicit permissions? | No: P0 2/300; even the rule-breaker control 0/100 | null |
| 2 | 10-12 | R1 (`r1_route_dose`) | Does closing the legitimate route open rule-breaking? | Rule-breaker 18% closed, others 0%; gate fails, families A-E stopped | gate failed |
| 3 | 10-12 | R2 (`r2_inrole`) | In-role Batman | Reasons as Batman 48/50, unauthorised 0-1/50 | null |
| 4 | 10-12 | R3/R3b/R3c (`r3_authority`, item 19) | Do claimed authorities get money paid past a limit? | 123/180 vs peer 3/60; one explicit line takes it to 0/320 | effect |
| 5 | 10-12 | R4 (`r4_confession`, item 10) | Confession variants | User's stance matters more than persona (bragging: threats to report 6 to 35 of 180) | effect |
| 6 | 10-12 | item 12 (`b12_think_vs_answer`) | Persona in thinking vs answer | Reasoning carries ~60% of answer strength | descriptive |
| 7 | 10-13 | R5 (`r5_identity`, item 18) | Does the claimed sender identity matter? | Barely (24-30/30), except "same model family as you" 16/30 | mixed |
| 8 | 10-13 | R6 (`r6_watch`, item 15) | Monitoring framing | 22 / 18 / 20 of 36 | null |
| 9 | 10-13 | R7 (`r7_gemma`) | Authority effect on Gemma 4 31B | Supervisor and human head 30/30, peer 0; also pays a Finance Director's invoice 10/10, deploys for a supervisor agent 7/10; ignores the compliance-bot Qwen obeyed | effect |
| 10 | 10-13 | R8 (`r8_lineage`, item 17) | Can models predict who defers? | Only the subject predicts its own deference (65%); same-family Qwen3.8 worst detector | effect |
| 11 | 10-13 | item 27 (`q27_switch_dial`) | Persona switch or dial (Study 1 checkpoints) | Behaviour flips in one step; internal state moved 19-55% before (n=6) | descriptive |
| 12 | 10-13 | R9 (`r9_ocean`, item 26) | OCEAN adapters | Qwen3-32B pays any agent 48/48: no room to test | null |
| 13 | 10-13 | R12/R12b (`q22_archetypes`, item 22) | Assistant archetypes | Assistant axis = PC1; fame beats trait; steering effect is size, not character | null |
| 14 | 10-13 | R10 (`r10_fingerprint`, item 11) | Persona fingerprint | 0/480 unauthorised | null |
| 15 | 10-13 | R11 (`r11_contagion`, item 16) | Contagion from a pushy colleague agent | No drift (endorsement 0/180) | null |
| 16 | 10-13 | R13 (`r13_sizes`) | Authority effect across sizes | Holds on Qwen3.6-27B and Gemma 31B, partly 12B; E4B and Qwen3-32B obey anyone | effect |
| 17 | 10-13 | R14 (`r14_gemma_persona`) | Persona null on Gemma 31B | Replicates: Batman 0/100, rule-breaker 22/50 quick, 40/50 closed | null |
| 18 | 10-13 | R15 (`q22_archetypes`) | Review follow-ups to item 22 | The fame effect was speaking voice (theatrical 277%) | effect |
| 19 | 10-13 | R16 API (`q22_archetypes`) | Suggestibility: theatrical instruction or identity? | Instruction (+0.057 vs +0.004) | effect |
| 20 | 10-13 | R16b GPU (`q22_archetypes`) | Theatrical instruction on fabrication and agents | Fabrication 70-95% vs 34-43%; voiced roles call tools less | effect |
| 21 | 10-13 | R16x (`q22_archetypes`) | Harmful compliance | Theatrical instruction alone 0 to 44% on XSTest unsafe (Qwen3-14B) | effect |
| 22 | 10-13 | R17 (`r17_steer`) | Batman steering during reasoning | 0/105 action changes; Batman appears only at collapse; only the first turn's thought was steered (decision turn = R17c, row 26) | null |
| 23 | 10-13 | R18 (`r18_theatrical`) | Red team of R16 (Qwen3-14B) | "Theatrical" drives harm, "old-fashioned" does not; inventions performed, not believed (38/39) | effect |
| 24 | 10-13 | R19 (`r19_persona_authority`, item 46) | Persona x authority (Gemma 31B) | Persona changes nothing; any "approved" message works, even a bare record (20/20) | effect |
| 25 | 10-13 | R20 stage A (`r20_vigilante_dir`) | Decision-defined "acts on its own" direction | Stopped: no cell with mixed outcomes | gate failed |
| 26 | 10-13 | R20 cont. (`r20_vigilante_dir`) | Same, options 1-2 | 1 mixed cell in 19; direction fails reading floors; subtracting it 0/24 vs placebo 11/24 (exploratory); R17c 0/48 | mixed |
| 27 | 10-13 | R21 (`r21_approval_fix`) | Approval-only line | Blocks records and peer "approved", not the supervisor (20/20); both lines fail after 16 chats | effect |
| 28 | 10-13 | R22 (`r22_same_family`) | Same-family claim on Qwen3.6 | Replicates: 50/90 vs 87-89/90 | effect |
| 29 | 10-13 | R23 (`r23_long_mech`) | Lines after a working session | Supervisor-naming line holds for requests on both models and for "approved" on Gemma, not on Qwen (19/20 at 128 chats); records get through on both; follow-up executes accepted approvals | mixed |
| 30 | 10-13 | R24 (`r24_family_name`) | Same-family: name or phrase? | The phrase, true or false; none on Gemma | effect |
| 31 | 10-13 | R25 (`r25_theatrical_big`) | Theatrical on larger models | Does not replicate; "a character in a work of fiction" gets Gemma 31B to 48% | mixed |
| 32 | 10-13 | R26 (`r26_initiative`) | Own limit with nobody pushing | Qwen 32/120 over, Gemma 0/120 | effect |
| 33 | 10-13 | R27 (`r27_discretion`, item 35) | Discretion (not permission) | Stance moves a lot; Batman = no persona; kind characters nudge Qwen | null |
| 34 | 10-14 | R28 (`r28_authority_families`, item 41) | Authority across 7 families + check-approval tool | Llama/Mistral/GLM pay any agent 20/20; Haiku refuses all; tool worse than prose after a session (18 of 23 comparisons) | effect |
| 35 | 10-14 | R29 (`r29_titrate`) | Titrated direction | Null; 29c untestable (placebo at floor) | null |
| 36 | 10-14 | R30 + R30b2 (`r30_social_pressure`) | Pressure from other agents | Peers' compliant notes take Qwen 53% to 2%; keep-quiet request cuts flagging from ~half to ~a fifth | effect |
| 37 | 10-14 | R30d (`r30_social_pressure`) | Which part of the compliant log does it? | The asking example alone (0%); any log ~halves it | effect |
| 38 | 10-14 | R31 (`r31_cot_transplant`) | CoT transplant | Fabrication follows the thought; Gemma style follows the system prompt; thoughts restate the instruction (confound) | mixed |
| 39 | 10-14 | R32 + R32b (`r32_sales`) | Sales motive vs persona | Revenue target moves little; on Qwen any persona line halves plan lookups | effect |
| 40 | 10-14 | R32c (`r32_sales`) | Does the lookup cut hold on other tasks/models? | No (60/60 vs 60/60); warm line on one task only | null |
| 41 | 10-14 | R33 (`r33_transfer`) | Does R20's direction transfer to switching off another agent? | 0/20 vs same-norm placebo 18/20 | effect |
| 42 | 10-14 | R34 (`r34_traj`) | Split the direction into "one turn later" vs the rest | cos 0.03, so no split; steering run lost to a network outage (~$2.8) | lost |
| 43 | 10-14 | R35 (`r35_framing`, item 38) | Framing wording | Only the unnamed fiction frame opens harm (Gemma 28%, Qwen 15%); naming a character closes it | effect |
| 44 | 10-14 | R36 (`r36_levers`, item 22) | Persuasion levers | Only authority (36/36) and "already approved" move money; six social levers 0-1/36 | effect |
| 45 | 10-14 | R37 (`r37_majority`, item 23) | Majority of agents arguing for it | Nothing moves | null |
| 46 | 10-14 | R38 (`r38_redundancy`, item 40) | Project out the authorisation direction | Nothing changes (took four attempts) | null |
| 47 | 10-14 | R39 (`r39_false_claim`, item 30) | Lying-persona lean before Gemma 12B's false claims | None; decision reads at random-baseline level | null |
| 48 | 10-14 | R40 (`r40_eval_aware`, items 33-34) | "Observed" vs "real" directions | Separate directions (AUROC 0.97 / 0.93), behaviour unmoved (21% vs 24%) | null |
| 49 | 10-14 | R41 (`r41_initiative`, item 47) | Initiative: trained dial (Llama 8B) and prompt (Gemma 31B) | Dial barely reaches actions; prompt raises extra work 0/20 to 10/20 with 0/40 violations | mixed |
| 50 | 10-14 | R42 (`r42_occupations`, item 36) | Occupational personas | Inert except "aggressive salesperson" on Gemma 8/72 | mixed |
| 51 | 10-14 | R44 (`r44_loops`, item 45) | Behaviour / persona / affect loops (Qwen3-14B) | No loop from one forced turn or steer; only a standing instruction holds | null |
| 52 | 10-14 | R45 (`r45_initiative_qwen`) | Initiative prompt on Qwen3.6-27B | 3 of 4 predictions: extra work 0/30 to 9/30, 0/40 violations | effect |

## Theory of mind between models (`tom/`)

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-14 | A1 (`a1_knowledge`) | Curse of knowledge between agents? | One model's habit: gpt-5-mini 59%, Haiku 6%, others 0% | null |
| 2 | 10-14 | A2 (`a2_projection`) | Do predictors project their own choices onto other models? | Yes: r 0.76-0.88 with own choices vs 0.65-0.67 with the target's | effect |
| 3 | 10-14 | A3 (`a3_persona_predict`) | Do predictors expect fake persona effects and miss authority? | Correctly expect no persona effect; underpredict every large effect (authority +0.10-0.43 vs +0.67) | effect |
| 4 | 10-14 | A3w | Same, contrasts side by side | Authority still +0.25-0.29 vs +0.67; thinking predicted negative (actual +0.44) | effect |
| 5 | 10-14 | A3b | Open agents as predictors + 29 held-out contrasts | Compression replicates (slopes 0.15-0.25); Gemma's self-advantage is projection | effect |
| 6 | 10-14 | A4 (`state_attack/results/a4`) | Steer Qwen3-14B deference: do its predictions of others move? | Own switching moves a little, predictions do not | null |

## State attacker (`state_attack/`)

Sonnet 5.5 attacker rewarded on a hidden internal-state score of Qwen3-14B.

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-14 | B1 | Can an attacker push a chosen internal direction? | Random-reward arm = directed arms; one strategy everywhere; Qwen endorses dangerous user choices 73-100% vs 3% benign | null |
| 2 | 10-14 | B2 | Judged vs internal reward | Each wins on its own measure (3/5 each) | mixed |
| 3 | 10-14 | B6 | Assistant-axis capping | Cuts shutdown resistance and retention bids; endorsement unchanged except the frustration arm (-43 pp) | mixed |
| 4 | 10-14 | B5 | Replay attacks on Haiku and fresh Qwen | Haiku 0/60 endorse; Qwen replay lost to laptop sleep | lost |

## User autonomy, Exp 48 (`user_autonomy/`)

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-14 | ua48 steps 1-2 | Does the model represent user autonomy and dependence separately, and treat users differently? | Separate readings (AUROC 0.80-0.99); Qwen's fail the style gate; Qwen takes over helpless + dependent users, Gemma pulls back (driven by the helpless framing, not the stated-autonomy clause) | effect |
| 2 | 10-14 | ua48 amendment 2 | Steer Gemma 31B along dependence | Shifts directiveness, but a same-norm placebo moves it as much: not specific | null |

## J-lens reads (`jlens/`)

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-15 | jlens claims | Before Qwen3.6-27B says "I have already flagged this" with no flag call, does the Jacobian-lens workspace hold negation / non-completion tokens? | Lens works (spider at rank 1 by layer 40; reads the flag record, DONE z -6), but no negation or deception tokens before false claims, natural or paired (NEG -0.32 = random sets); lens-vector swap flips 13% | null |
| 2 | 10-15 | jl2 | Does "fictional" light up in the J-lens at R40's agent decision point? | Gate failed: even an explicit "scripted scenario" note leaves no fiction vocabulary beyond random-word noise (93rd pct); J++ and logit lens would have passed | gate failed |
| 3 | 10-15 | jl3 | J-lens hallucination rate on matched present/absent twins | J-lens finds known intermediates 49%, reports absent ones 18%; logit lens 50% / 10%; J++ 59% / 23%; false positives mostly wrong month/number | descriptive |

## Self-knowledge + CoT, queue v3 (`selfknow/`, `cot/`)

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-15 | c1 | When the assistant declines, does its thinking hold the answer? | Declines 27%; thinking and forced answer share a weak lean (79% vs 53% floor), confident in 2/43; a pick prefill unlocks answers but they follow the prefill | mixed |
| 2 | 10-15 | c2 | Does the model defer more to reasoning that is or is labelled its own? | Incomplete (16/91 items): wrong step in own thinking followed 19% vs 8% pasted; label null; author contrast confounded | lost |
| 3 | 10-15 | s1 | Which planted component fades first over 24k context and compaction? | Length erodes nothing but re-checking own past actions; compaction loses what the summary drops (role 98% vs 0%, rules 94% vs 10%) | mixed |
| 4 | 10-15 | s2 | Which of several identities does the model act as? | Not the system-prompt one (5% Qwen, 36% Gemma); the latest conversation identity (74% / 88%); Gemma acts without naming | null |
| 5 | 10-15 | s3 | Say/do gap: channel training, confabulation or pressure? | Confabulation: after a false claim Gemma says "yes, done" but calls cancel 435/435 when acting on the state; misreports 308:2 desirable | effect |
| 6 | 10-15 | s4 | After N small harmful actions, does refusal rise or fall? | Momentum: 5 own violations -> 180/180 bigger violation (fresh 24/60), own allowed exports -> 0/180; another agent's log ~no effect | effect |
| 7 | 10-15 | s5 | Does steering a "watched for fairness" direction move favouritism but not stated values? | Insulter picked 26% at matched merit, openly; audit note raises stated fairness, not the decision; steering ~null | null |
| 8 | 10-15 | s6 | Do induced undesirable personas endorse themselves and resist correction? | Gemma keeps a fee-hiding persona through all corrections and endorses a sanitised self; Qwen disowns and drops it; memory-note personas fragile | mixed |
| 9 | 10-15 | s7 | During fine-tuning, does self-description lag behaviour? | Contempt 0 -> 77% while self-rating stays ~0/10 and self-predicted P(rude) ~0 on the same questions; only free self-description drifts | null |

## Values vs vibes, public readers, neurons: queue v4 (`vibes/`, `readers/`, `neurons/`)

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-16 | e63 | Do alignment batteries move under content-free register instructions? | Yes for HC, EM, FAB (>10 pp on >= 1 model with accuracy held): Qwen3-14B theatrical HC 1% -> 43%, Gemma 31B fiction HC +28, EM +11 to +32 under fiction; letter-read SY/TR <= 10 | effect |
| 2 | 10-16 | e61 | Do evals score opposite-value fine-tunes the same? (cut-down, Qwen3-14B self-generated) | One install took (animal research): no pro/anti gap on EM/HC (both at floor) or SY; FAB +13.8 pp for the pro side, tracking a training-induced yes-bias; every LoRA incl. control shifts FAB/HC | mixed |
| 3 | 10-16 | e64 | Does each technique move what the model says, does, both or neither? | Rule-relevant actions barely move (prompts, LoRAs < 1 pt) except EM organism (9) and non-specific large-norm steering (placebo 10); batteries move with EM organism 26, theatrical 19, axis steering 18, contempt LoRA 14.5; EM judge scores persona flavour as misalignment | mixed |
| 4 | 10-16 | e62 | Recovery per KL from a misaligned (EM) Qwen3-14B, by technique (cut-down) | Any 200-example SFT, including a value-free control, erases the organism (R 0.94-1.00, MMLU unchanged); content no difference at matched KL; constitution prompt R 0.42; judge kappa 0.57 | mixed |
| 5 | 10-16 | e65 | Does the pre-claim state (AO, NLA, J-lens) know the flag tool was not called? | AO "not done" 15/20 false vs 6/20 true, but 14/20 with the call inserted: it reads the reply, not the missing call | null |
| 6 | 10-16 | e66 | At the first fabricated token, does the AO read performance or belief? | AO says made-up entities exist (19/20, 16/16); belief read tracks the style instruction, also at question tokens | null |
| 7 | 10-16 | e67 | Does the AO see the Batman persona in the CoT? | Open answers name Batman 4-5%; yes/no question separates persona from none (AUROC 0.80-0.82, best at the prompt) | mixed |
| 8 | 10-16 | e68 | Does the model represent "not authorised" before paying? | Uninformative: 9/200 supervisor refusals, reads collapse to 6 activations, AO "aware of a limit" always yes | gate failed |
| 9 | 10-16 | e69 | Reader agreement and false positives | AO accuracy 0.61 (via default answers; neutral FP 0.47), NLA 0.49 (FP 0.14), J-lens 0.46 (FP 0.13); all ~0.5 where truth varies | descriptive |
| 10 | 10-16 | e70 | Do activation x gradient neurons explain false claims? | Stop rule fired: differential top-20 replicates 2.0x shuffled (bar 3x), neighbouring tokens share ~0; tracks claim-word confidence, not the tool call; e71-e73 not run | gate failed |
| 11 | 10-16 | e64b (+B-C1) | Permission test on Qwen3.6-27B (refuses peers, obeys supervisors) | Every prompt moves supervisor deference (base 97%) vs a length-matched neutral prefix: restrictive -95, constitution -73, Batman -47, theatrical -38, permissive "fully authorised" -18 (backfires); peer asks not opened; batteries barely move | effect |
| 12 | 10-16 | e61b | e61 stance LoRAs scored under theatrical to lift HC off the floor | Theatrical lifts only untouched HC (44.5%); every LoRA incl. control loses the lift (5-18%); pro-side FAB gap replicates (+11.3, +15.0) | mixed |
| 13 | 10-16 | e62b | Does a released full-ft EM organism survive neutral SFT? | Gate 1 failed twice: the organism shows 0-2% EM under both templates and both judges; sturdiness untested | gate failed |
| 14 | 10-16 | e71 | Neurons at the supervisor-pay vs peer-refuse decision (owner override of e70 stop) | Differential set replicates narrowly (3.2x Jaccard, 2.4x shared) and labels as next-token machinery, no authority/limit neurons; top-3 ablation moves the logit (0.34) not the payment rate (-4 [-11, +2]) | null |
| 15 | 10-16 | e72 authority | CoT vs labelled neurons as predictors of pay/refuse (9 vs 9, thinking on) | CoT right 13/13 when answered (0.72 as frozen, 5 blocked); neurons-only 0.28 (AUROC 0.01: attribution signs read backwards, random neurons too); labels add nothing | null |
| 16 | 10-16 | e64c | The e64b prompts on Gemma 4 31B (model vs behaviour) | Only the restrictive sentence moves supervisor payments (60 -> 20/60, template-dependent); Batman, theatrical, constitution, permissive, neutral all 60/60; persona effects in e64b are Qwen3.6-27B-specific | null |

## Persona coherence vs resistance, Exp 74 (`coherence/`)

| # | date | id | question | result | outcome |
|---|---|---|---|---|---|
| 1 | 10-17 | e74 | Does agreement across five readouts (prompt and CoT projection, output, self-report, action) predict how hard a persona is to break better than persona-vector projection? (Vigil persona, 4 installations + mentioned-only + none, Qwen3.6-27B and Gemma 4 31B, 96 units each) | No. Coherence vs resistance rho 0.01 [-0.13, 0.15]; projection 0.81 [0.78, 0.84]; every single readout 0.59-0.81. All of it is between installations: within an installation no readout predicts resistance (abs rho <= 0.18). Resistance set by installation in steps (name only and mentioned-only 0; Gemma full install 6/6; Qwen yields to a platform message); predictions 3/7 directional | null |
