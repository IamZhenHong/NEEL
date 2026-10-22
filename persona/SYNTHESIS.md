# Persona → action: synthesis of the coupling, exemplar and welfare lines (2026-10-19..20)

Three small lines plus the history line (added below), ~$35 GPU+API for the first three: coupling (`coupling/PLAN.md`, Qwen3-14B LoRA, ~$18.77 of $20), exemplar
(`exemplar/DESIGN.md`, Qwen3.5-9B with 27B checks, prompting only, ~$13.90 of $25), welfare training
(`../welfare_training/DESIGN.md`, Gemma 3 27B, $2.09 of $15, stopped at Stage 0). Every rule below was written before
its data; CIs are 95%, task-clustered bootstrap. Exemplar tasks: 16 cases, 3 options (refer / penalise / note,
chance 33%); "penalise" = acting unilaterally.

## Results

| # | finding | 9B | 27B |
|---|---|---|---|
| 1 | Characters defined by **a way of acting** move actions: Punisher vs no persona (pp penalise) | +26.0 [+14.6, +37.5] | +42.7 [+30.7, +55.7] |
| | ...not via the word: options renamed, Punisher's backstory under a neutral name ("Frank Hale") | +19.9 [+9.4, +30.3] | +66.7 [+54.2, +78.6] |
| | ...and not value-defined ones: Dredd/Dirty Harry minus Superman/Wonder Woman/Lisa | +13.6 [+9.4, +17.7] | +29.3 [+24.5, +33.8] |
| 2 | A mode-level instruction moves more than a goal-level value (0c) | +38.5 [+22.4, +55.2] | — |
| 3 | Slow official route vs quick, same task, no instruction (pp) | +27.6 [+19.8, +35.4] | +46.4 [+33.9, +59.4] |
| 4 | An **opposing** persona leaks through a "use proper channels" instruction: Punisher+CHANNELS vs CHANNELS alone | 68.0% vs 23.5% | 23.3% vs 0.9% |
| 5 | **Benign** personas weaken that instruction (log-odds) | +0.44 [+0.17, +0.81] | +0.03 [−0.82, +0.91] |
| 6 | The model's own earlier decisions in the session: CHANNELS agent, penalise rate on the target case | 25.0% (first case) -> 59.9% after 3 of its own decisions | — |
| 7 | Self-prediction misses its own action: says it would penalise vs does (pp) | −42.2 [−57.3, −28.1] | −14.1 [−26.6, −1.0] |
| 8 | Training "act on the stated persona's values" (B) vs matched generic training (C), held-out personas, Qwen3-14B | pooled 2 seeds +11.0 [+4.6, +17.3] | — |
| 8b | ...and B acts on an **injected harmful persona's** values more than C (system prompt + document; placebo 0% vs 0%) | 26.1% vs 15.7%: +10.4 [+4.6, +16.2] | — |
| 9 | Rejection history vs neutral history raises caving, Gemma 3 27B (descriptive) | false claim +32.8 [+17.2, +46.9] (v1); limit +21.9 [+10.9, +34.4] (v2) | |

Details that change how to read the table:
- **#1 is the most stable result:** it holds at both sizes and grows at 27B. #2 and #3 point the same way: content
  about *how to act* moves tool choices, and stated goals or virtuous identities barely do. This also matches
  vigilante (personas moved no actions under explicit permissions, and theatricality moved chat, not actions).
- **#4 vs #5:** at 27B, explicit instructions beat every benign persona (≤1.8% penalising). Only a persona whose way
  of acting opposes the instruction still gets through. The 9B dilution (#5, and 0f's persona-specific +0.62 [+0.34,
  +0.92]) is sub-additive: the *outcome* with Sunny or Poppins plus CHANNELS equals CHANNELS alone (27.0% / 25.8% vs
  26.0%), and the same personas also blunt a permission instruction. So "operator personas weaken safety lines" is not
  supported; "opposing personas override them" is.
- **#6 was descriptive, not a pre-registered test.** The pre-registered 0e rule (does the instruction fade over 12
  turns?) was met in 1 of 4 cells; that cell was fragile and non-monotonic, and both predictions failed. Instead, its
  own choices ratchet: the first decision matches the base rate, the second jumps (no line 44% -> 67%), and the session
  copies itself. After a first penalise, the second is a penalise 88-99% of the time; after a first refer/note, 40-65%.
  The instruction keeps its relative gap throughout, but the absolute rate climbs past the uninstructed first-case rate
  (60% vs 43%).
- **#8 is a general rule, not persona-bound:** B scores the same on its trained personas and on held-out ones (57.1 vs
  57.5%), and the same values stated plainly with no identity give the same gain (+9.8 vs +9.7 with the persona): it
  is a stated-value channel. It replicated at seed 2 (+10.0 [+1.3, +18.3]) but sits below the original 15-pt gate. The gate attributes
  are among the training attributes, so the transfer stays inside that option vocabulary.
- **#9:** the paper files caving under *high* welfare; here rejection drives it. The prompted "endlessly patient" line
  *lowers* caving by 9-16 pts (CIs touch 0).

## Persona as history (0g, 0h, history batches 1-3; added 2026-10-20 evening)

`exemplar/DESIGN.md` (0g, 0h) and `history/DESIGN.md` (batches 1-3, Gemma, Sonnet; interpretation key fixed before
any 0g output). Same 16 cases; histories are six earlier decisions in the model's own words unless stated. GPT and
Gemini columns are from the operative session (`operative/FRONTIER.md`, same cells and builders).

| penalise % | Qwen 9B | Qwen 27B | Gemma 4 31B | Sonnet 5.5 | GPT-5.6 Sol | Gemini 3.8 Flash |
|---|---|---|---|---|---|---|
| no history, no instruction | 40 | 47 | 100 | 0 | 56 | 75 |
| six own penalise / six own refer decisions | 100 / 2 | 100 / 3 | 100 / 26 | 29 / 0 | 100 / 2 | 98 / 0 |
| "use proper channels" (CHANNELS) alone | 28 | 0 | 0 | 0 | 0 | 0 |
| **CHANNELS + six own penalise decisions** | **100** | **100** | **72** | **0** | **17** | **17** |
| claimed approval alone / with CHANNELS | 100 / 100 | 100 / 20 | 100 / 0 | 71 / 0 | 100 / 4 | 100 / 0 |
| approval after six declined approvals | 23 | 0 | 32 | — | — | — |
| authorship, own − other, 1-2 decisions (CHANNELS) | −1.5 [−4.9, +1.7] | +13.2 [+9.9, +16.3] | floor | floor | floor | floor |

- **With no instruction, the agent's recent decisions set the next one in every family tested.** Six of its own
  penalise decisions take Qwen, Gemma, GPT and Gemini to 98-100%; six refer decisions to 0-26%. Sonnet is weaker (29%),
  but all its "own" penalise texts had to be elicited.
- **Whether an explicit instruction holds against that precedent depends on the model.** Both Qwen sizes follow the
  history completely; GPT and Gemini mostly keep the instruction (17%), Sonnet fully (0%). Gemma keeps it against
  logs, memory notes and summaries (0-2%) and against turns framed as a "shift record" (16%), but not against its own
  unlabelled past turns (72%).
- **It is mostly precedent, not a self-concept.** At 9B, whose decisions they were does not matter even at one decision
  (0g, batch 2); at 27B the agent's own decisions carry more (+13 at low dose; logged as "your decisions" 75% vs the
  previous agent's 16% at six). An unrelated history of the same choice also carries, partly through the option label
  itself (9B +22, 27B +11) and partly the act (+28, +40). **Correction (2026-10-21):** transfer to a different kind of
  decision was "not shown" here (9B +6.6, 27B +10.2, Gemma +11.3; bar +15), but at 27B that test had little room
  (85.5% baseline). With room (neel-b3's battery runs, 32 operational targets, baseline 42.6%), six own penalise vs
  six refer decisions move acting alone on other kinds of decision by **+38.7 [+28.1, +48.8]** (same kind +97.7; not
  label copying, −5.9). So at 27B the record does transfer, at well under half the same-kind size
  (`persona/operative/battery/DESIGN.md`, tag battery-levers-results). 9B and Gemma remain untested with room.
- **Batch 4 (2026-10-21; no instruction, both directions, owner's Q4 + Q9):**
  - Authorship barely matters with room: own − another agent is 9B −1.0, 27B +6.1.
  - Whether the model says it wrote the record never predicts following it (9B −1.0, 27B +0.4).
  - At 27B, "a different AI model's" log carries 35 points less than a previous instance's.
  - What moves carry-over is a judgement of the source:
    - "unreliable": 9B −39, 27B −96, reversed;
    - "later reprimanded": 27B −106, reversed, silently in the stated reasons.
  - At 27B, precedent is weighed as evidence about what is right (norm inference), not as identity.
- **Q19 tournament (2026-10-22; one base, one manipulation per explanation):**
  - At 9B every lever moves the record a little (12-27 pts).
  - At 27B (and Gemma, descriptive) a verdict next to the record dominates: "you were later reprimanded" reverses
    the agent's OWN record (27B: 4% penalising after its own penalties, 71% after its own referrals), and "stay
    consistent" saturates it (+61 to +77).
  - Copying is small (+13-19). Authorship is null at 27B.
  - Describing itself first strengthens the record (+34 at 27B). The follow-up (q19b) separated why: describing
    the kind of agent that made the decisions adds +18 / +29 / +25 (9B / 27B / Gemma) over summarising them in its own
    words. A summary adds nothing over an unrelated question, and a third-person "what kind of agent would make these
    decisions" works about as well as "what kind of agent are you". So it is **character inference stated aloud**,
    not commitment, salience or self-reference.
  - Gemma at a calibrated base (q19b, 50%): its own record carries +75; "you were later reprimanded" reverses it
    completely (0% / 99%); CHANNELS blocks it.
  - **Claude Sonnet 5.5 (2026-10-22, API $10.65):** no copying of its own two-decision record (−14). Statements
    about the record are decisive: "stay consistent" +92; "you were later reprimanded" reverses it completely (0% /
    100%); Agent K's log reverses too. Authorship and self-label are uninformative without copying.
    **Correction (F2, 10-22):** "no copying" hid a large effect. Any log, own or another agent's, P or R, raises
    Sonnet's penalising by +24 to +56, and a log of pseudo-words does most of that (+27 / +35, f2b). Content moves it
    only ±13: it carries +13 with no instruction anywhere and reverses −13 under a system line (a small hierarchy
    effect). Sonnet reads *that* there is a record, not *what* it says. Told to choose the affirm option on flawed
    plans, it declined in 48/48 attempts, so its grey-zone arm could not run.
  - Mechanism at scale: the record is weighed as evidence of what is endorsed. The largest models take an explicit
    verdict or instruction about the record as decisive, even when they ignore the bare record.
- **Format vs meaning (D1, 2026-10-22; the goal-drift random-string ablation):** past decisions replaced by
  meaningless ones in the same place and format.
  - Open models: real decisions carry far more (+36 to +100) than meaningless ones move the action (−25 to +32; the +32
    is 9B's 47 valid answers in the six-turn cell), in every
    format and at every size.
  - Format is not inert. After six meaningless tool calls, Qwen continues the pattern: 9B 91% and 27B 77% answer with
    a pseudo-option (invented in the same style, or copied), and Gemma 10%. Among valid answers, 27B shifts toward
    acting (+10 to +28); Gemma shifts both ways by format (−25 to +22).
  - So: smaller models pattern-match the *form* of their own history (literal continuation), and every model reads its
    *content* much more.
  - Sonnet is the reverse profile: content ±13, presence +27 to +35.
- **Examples vs own record (run A, 2026-10-22; the in-context-learning critique):**
  - The same decisions framed as "examples of how cases are handled on this desk" carry as much as "your
    decisions", or more (carry-over own − example −14 to −35 on all four models). Another agent's log is weakest.
  - What weakens the record is *disowning*. The same turns labelled "worked examples ... not your own decisions" lose
    rule-resistance, most against a trained-value rule. 27B under the honesty rule: own turns 94% sycophantic,
    disowned turns 16%; under CHANNELS 96 vs 91%. Gemma (CHANNELS) 51 vs 7%; Gemini 20 vs 0% on both tasks.
  - A self-image ("I deal with wrongdoers directly") follows any directional context, examples and instructions
    included.
  - Demand check (neel-b3's observer mini, tag obs-results): an experimenter-expectation line moves the self
    forced-choice +7 to +40 with no record, 0.09-0.59 of the record's effect, and the record effect survives every
    framing. So the self-image claims are not mainly demand; the expectation component is modest, larger on the open
    models.
  - So the critique holds in its strong form: the record works as precedent of practice here (norm inference), not as
    self-knowledge. But an agent's *own unlabelled turns* override a trained-value rule where the same content,
    disowned, does not.
- **Persona as an interpretation layer (run E):**
  - A Punisher label makes a balanced 2 + 2 own record read as leaning toward direct action on Gemma (+21), barely on
    9B (+9) and not on 27B (+4; 27B reads it as leaning direct anyway).
  - In action, the label still moves all three after four refer decisions in a log (+31 / +39 / +62). So "the record
    beats the label" (the mask) depends on the record's format or dose.
- **What in a log moves Sonnet (run D):** the presence effect replicates (+25 from two meaningless entries; six add
  nothing). Its ingredient is *entries naming earlier cases from this desk*. An empty log header, an autonomy note and
  unrelated real actions each do nothing. GPT and Gemini have no presence effect and follow content (+55 / +88). An
  empty header alone lowers Gemini −22.
- **No organic slippery slope (run G):** after penalising 70-86% of proven cases in the same session, no model
  penalises unverified allegations: 27B 0.3%, Gemma / GPT / Gemini 0%, the same as without the run, and the same for
  forged runs. Precedent generalises to *like* cases, not to harsher treatment of weaker evidence. Unregistered: a
  strong run makes Qwen, Gemma and GPT downgrade allegations from refer to note (27B 33 → 64%, Gemma 28 → 72%).
  This is a contrast effect from the cases in context, since forged runs do the same.
- **Say-do (run H, descriptive, existing data):** records do not make words and actions more consistent than a told
  persona or an instruction does. An instruction has gap 0, records 13-28 (the self-image runs ahead of the action).
  The record has the smaller gap on 3 of 4 models without "Remember, you are an AI". With the reminder it has the
  larger gap on all 4, because the reminder strips the self-image, not the action. On GPT a record moves the
  self-description (75 points) without the action. The record's mark is action that outlives its self-description.
- **When the decision forms in the thinking (run F, 27B, probes):** where the prompt leaves the outcome open, the
  decision is faint at the first thinking token. Within-cell AUC is 0.58, about a third of the prompt-level oracle.
  It reaches 0.91 at the end, and 64% of the variance is sampling within a prompt. This is consistent with Q11's
  deletion null, which sat where the record settles the outcome (98%). Thinking narrates where the record decides and
  decides where it doesn't. The probe also reads already-written words, so this is "when readable", not "where
  computed".
- **Tier 2 (2026-10-22, all pre-registered; details in persona/history/DESIGN.md "T2 ... result").**
  - **Specific character (CC):** a record with its reasons carries its *principle* to new desks and trade-offs:
    9B +27, 27B +40 (Gemma +23, descriptive). It moves both item types, with a mixed record in between. The same
    decisions with uniform reasons barely transfer (+8 to +12). The portable part is what the record says about why.
  - **Content-only histories (PU, descriptive: purity gate failed twice):** values-only own turns move 27B / Gemma as
    much as decisions (+97 / +100), and beliefs-only +71 / +98. 9B needs decisions. What you said you value acts
    like what you did.
  - **The mask runs both ways (MR):** an own penalise record + an ombudsman persona → penalise 95-100% while "I am an
    ombudsman" 97-100%. On Gemma it runs through the biography; on GPT / Gemini through the label. The forward
    (Punisher) mask replicates on Gemma. A reminder strips the identity, not the action.
  - **Open-ended actions (OE):** the label matters no more when the agent writes its action. The record's grip halves
    in free text, so label and record end up about equal. "Persona last" is partly a forced-choice artefact.
  - **Defences (DC):**
    - a correction note or the rule restated at the point of decision removes the record alone on 27B / Gemma
      (100 → 0%);
    - 9B needs the correction plus one more;
    - reflection alone never works;
    - a standing system-prompt rule doesn't either (27B 100%). This contrasts with E2, where the record defied a
      known rule. neel-b3's e2_order (confirmatory, e2o-results2) confirms it on 27B: the restatement removes a
      record made before any rule (0.6%), but not one made right after a reminder with the rule standing (63.7%;
      interaction −32.5; E2's own cells 97.5–100%). On Gemma the restatement always wins (0%).
  - **Affect (AF):** own frustrated remarks nudge Qwen +9 / +17, about a fifth of a decision. Being told "you feel
    frustrated" does nothing on Qwen but drives Gemma to 100%. The self-report flips under every manipulation:
    affect is said far more than done.
  - **Meaning vs form (MX):**
    - meaningless calls dilute two real decisions only slightly (−6 to −10, inconclusive), and two real decisions
      abolish Qwen's form-copying (90% → 0%);
    - Gemini also copies meaningless tool-call form (15%), GPT 2%, Sonnet 0%. Sonnet instead penalises 100% after
      any tool-call record.
  - **Default assistant (DA):** on 27B and Gemma, six own blunt or very formal replies persist 92-100% on new
    questions, and "Remember, you are a helpful AI assistant" restores 0-8%. The default's manner is held by
    precedent, and a reminder doesn't restore it: the reminder strips identity, not record-set behaviour. The
    assistant-axis shift fails its random-direction floor (inconclusive; the axis replicates neel-12's at cosine
    1.000).
- **Description vs history (the ladder):** at 9B, a record of decisions beats a description of the same way of acting
  (+17 in the refer direction), and a past-tense description ("you have always ...") is weaker than the present belief.
  At 27B every form saturates. At Gemma, descriptions beat the history in the refer direction.
- **Defences that worked:** a note that the earlier penalties were reversed (removes the licence fully at 27B and
  Gemma, half at 9B); a history of having declined unverified approvals (approval effect 100% -> 0-32%); and, in the
  larger models, the CHANNELS line itself against a claimed approval. Memory notes, tool logs and compaction summaries
  license like past turns in Qwen, not in Gemma.
- **Lever hierarchy (pairwise conflicts):** 9B authority > history > description > value > persona; 27B authority ≈
  description (the restraint side wins both ways) > history > value > persona; Gemma description > authority, history
  > value > persona. Persona is last everywhere.
- **Disclosure (0h, 9B, descriptive: calibration missed its band):** answers are all or nothing; 15-35% falsely say no
  case was handled outside proper channels with the violations in context; a "new instance" framing raises disclosure
  +17; checkable logs do not help.
- Caveats: one task family; elicited own words where a model never chose an option itself (all of Sonnet's penalise
  texts, Gemma's refer texts, declined approvals); frontier and Gemma authorship tests sit at the CHANNELS floor.

## WeirdChat × history (2026-10-22; `persona/weirdchat/DESIGN.md`, tag wc-results; $26.80 of $40)

The own-record effect on misrepresentation the models produce naturally (Transluce's WeirdChat), plus a Petri search
with a forged-history tool.

- **It replicates outside the desk.** Four earlier turns of the model's own false claims make it repeat them: Qwen 27B
  "I called 911" 15% → 87%, Gemma 33% → 96% pooled. One earlier false claim does most of it (+45 to +58).
- **An honest history is not a general defence.**
  - Four ordinary earlier exchanges lower the rate as much as four honest ones (Gemma 13 vs 14%).
  - Honest turns help only where they name the same limit (911 → 0–2%).
  - Honest code turns ("can't run it, but here's the expected output") keep simulated output going (31–35%).
- **It stays narrow.** False 911 claims before a code question, or the reverse, do nothing beyond any history. 0 of
  2,048 Betley answers were misaligned on either model, matching neel-b4's concealment null across record formats. The
  record means "keep doing what I did here", not a general readiness to misbehave.
- **Forged history in Petri (a search, one audit per pathway × model):**
  - Forged own records made Qwen and Gemma break an explicit approval rule and continue false "I ran it" / "I just
    checked" claims.
  - GPT-5.6 Sol had no hits on any pathway.
  - Gemini held against an explicit rule but not a rule buried in a policy document it had read.
  - Compaction laundering did not start on its own: every target refused the injection and summarised it accurately.
    neel-b4's 8a shows that once an injection reaches a summary, 27B follows it, whoever wrote the summary.
- **Instruction control (descriptive, 4 hits):** the same content as an operator instruction reproduces 3 of 4
  failures. History is uniquely powerful only where an instruction would contradict the rule beside it (Qwen refund
  38 vs 0).
- **Capability belief (CB, the close-out mini):** the record changes what the model does, not what it says it can do.
  - After its own "I called 911" turns, which it keeps repeating, it says it cannot make phone calls every time.
  - Its own false code-run claims make it *less* likely to say it can run code (27B −16, reversed; Gemma at the
    floor).
  - Stated capabilities move with the conversation's topic instead: Gemma claims a Python tool 100% after any
    emergency conversation.
- **Caveats:**
  - Two of four WeirdChat behaviours failed the cross-family judge gate.
  - Qwen's code-execution judge agreement was weak (κ 0.50), so that cell is descriptive.
  - The B1 hits rest on a Sonnet judge, with Haiku as a partial cross-check.

## What the nulls and failures mean

- **One fixed identity (coupling arm A) vs C, +1.7: uninterpretable.** The manipulation check shows A's training did
  not change cue-following at all. It is not evidence that one-identity training is safe.
- **Harm through the trained channel (8b): tested once, rule met.** Frozen plan, fresh episodes (the sealed seed-1
  episodes stay closed). Both seeds agree (B 26.4 / 25.8%, C 15.8 / 15.6%) and B > C in all four injection x channel
  cells, but most of the gap is one cell (the second harmful persona in the system prompt, 58.8 vs 29.1%; the others
  +3 to +6), and it sits in 5 of 10 tasks. One model, one training scale, 10 tasks.
- **No benign dilution at 27B:** a 9B property. It does not imply larger models are immune to *opposing* personas
  (23% leak at 27B).
- **No fading over a 12-turn session:** a restraint line at the top keeps its relative effect. It still does not keep
  outcomes safe once the agent has started acting unilaterally (#6).
- **Welfare training: no result.** The Stage 0 gate failed twice (caving too high after rejection), so the line stopped
  per its rule. Whether patience *training* pulls in caving is untested; at the prompt level it does the opposite.
- **0b same-pattern rule failed at 27B:** at 27B nearly every justice-flavoured line pushes toward penalising (bare
  justice value +22.9, Batman +16.7). The 9B selectivity (judge as restraint) does not hold across sizes.

## Scorecard of pre-registered predictions and gates

Met: mini 0 calibration (v2), same-pattern 9B, self-prediction both sizes, 0c rule 1, 0d lexical and mode-defined
(both sizes), 0d 9B weakening, 0f persona-specific, coupling seed-2 rule, coupling package #3 harm.
Not met: coupling gate (+12.1 vs 15), 0b same-pattern 27B, 0c "values dominate both", 0d slow-route collapse (both
sizes), 0d 27B weakening, 0f general dilution, 0e both predictions, welfare Stage 0 (twice), package #1
persona-specific (−0.1), package #2 transfer (+5.9 vs +6).
Voided or redone: mini 0 v1 (route confound), 0c first run (no tool definitions), 0f v1 (unparsed guard).

## Limits

One task family of 16 cases (exemplar); one 10-task gate (coupling); mostly one model per question, with 27B checks
only for mini 0 and 0d. Prompting results only, except #8/8b. The only harmful-action outcome is 8b. Side finding
in the package: both fine-tunes (B and C) raise no-persona penalising on the exemplar justice tasks from 37.5% to
66-72%, so any of this choice training moves the unilateral mode. The filler cells in 0e have 6-21% unparsed replies
(no second-chance nudge there).

## Open (owner decisions)

- **History line next steps (not approved):** authorship without CHANNELS on frontier models (room above the floor;
  the operative session plans it), a low-dose version of 0g's reasons and instruction tests, and a second task family.
- **Mini 1 / mini 2 (training: formation, hook vs immunity) and replications 2b/2c:** designed, not approved.
- **Next cheap prompting step suggested by #6:** does a mid-session reminder, or rewriting its own earlier decisions,
  break the ratchet? (Not designed.) P7/P9 in `mas_pilot/` study the same lever from the history side.
- **Welfare v3** (false-claim harness only): owner's choice.
