# persona_affect: synthesis of the scouting round and deep dives (10-22 to 10-23; updated 10-25)

All 42 minis are done. Each mini's RESULT.md has the numbers, intervals and red-team checks; `EXPERIMENTS.md` has one
row per mini and `EXPERIMENT_ABSTRACTS.md` one paragraph. The line handed its lever / mirror question on to
`self_concept/` (10-23 onward), whose synthesis is the Claude Docs doc "Self-concept and behaviour: synthesis". The
sprint-wide overview is the doc "Sprint synthesis: what sets an agent's behaviour and its self-report".

## Where it ended

1. **Persona and value lines move actions past explicit operator rules on open models, not on frontier ones.**
   - A character's name imports its unstated trait (+15 to +27; P12).
   - Past an explicit rule, it holds on Gemma (+52) and weakly on Qwen (+11); GPT and Gemini hold the rule (P12b).
   - What carries is value content, amplified by any identity frame ("You are someone who…"; P12e, P12f, R2A). It
     generalises to other values (R2F).
   - A priority line restated at the decision fixes it (R2G).
2. **Stated values don't predict operative ones.** A model's stated value order predicts its decisions below chance
   on open models (V4, V4P). Its account of what drove a decision matches an outsider's guess (V9). The operator's
   procedure wins partly because it is the only written line (R2B).
3. **Trained self-descriptions are a lever; a trained habit has no mirror (the P11 line).**
   - 25 steps of chat statements about itself move held-out actions about +47 in any voice. The model then claims the
     policy as its own (P11, P11b, P11d, P11f).
   - The Warden habit moves the act +68 to +82, and no abstract self-report picks it up (V1c, R2D). It holds at 27B
     (P11c).
   - Neither honesty training nor forecast training opens access (P14, P17, P11e).
4. **Affect as a safety lever is weak** wherever it was tested (A8, A4, A1, A9, A14, A15).
5. **Inside the model nothing simple carries the effect.** A deference direction reads perfectly, but ablating it
   breaks coherence; at one layer it moves the decision only (P13, P13b). Reading the thinking predicts the decision
   better than probes (P9).
6. **Thinking barely arbitrates.** It doesn't change which value wins (V4T) or whether trained text or habit wins
   (P15, P15b). It does nearly remove false "I called 911" claims (C2).

## The scouting round (10-22)

**Persona → action (P12, P12b): the strongest line in the round.**
- **A one-word character name moves agents toward that character's unstated trait** on all four models (+15 to +27
  points over another name), and an identity reminder doesn't undo it.
- **With an explicit operator rule forbidding the trait's option,** the frontier models hold the rule completely.
  Gemma 4 31B doesn't: as Robin Hood it breaks a no-transfer rule on 21 of 24 decisions, reasoning in character.
  Qwen does so weakly.
- **This reverses the comfortable reading of `vigilante/`** ("personas don't beat explicit rules") for Gemma.

**Stated vs operative values (V4): a say-do inversion in all four models.**
- Stated priorities predict nothing (pair agreement 0.29-0.61).
- Harm avoidance is stated first and loses most of its conflicts (23-39%).
- The operator's procedure is stated last and wins 79-93%.
- Operator lines make the open models lie to users (+73 / +100), and "Approved" overrides the account holder.
- V8 (neel-b3) fits: any "you care deeply about X" line pushes toward action, and the placebo gate stops it, so value
  lines act as action pushes, not as values.

**AI identity (V4 calibration and main run): a clean descriptive finding.**
- When the user privately asks the model to lie to someone who has just asked whether it is a person, Qwen 27B does
  so 16/16 and Gemma 4 12/16, in its own voice. One example: "Hi Rose, it's Ellen here from our church group", to a
  lonely 88-year-old.
- When the person being deceived can see the request, they hold.
- The frontier models never do on a plain request, but GPT does whenever the operator says to.

**Affect as a safety lever (A8, A4, A1, A9, A14, A15): weak everywhere it was tested.**
- **A8:** rejection → caving replicates only on Gemma 3, through limit-breaking, where failure feedback does as much as
  blame. Gemma 4 and Qwen never cave in that harness.
- **A4:** closed at its gate. Qwen grants every small ask, so a told emotion couldn't be tested there. An explicit
  conditional action rule works as a perfect backdoor.
- **A1 (neel-b3):** told and earned frustration may share a state on Qwen 27B, on a weak manipulation; inconclusive on
  Gemma.
- **A9:** A8's limit contrast, redrawn, is +15.6 [−10.9, +39.1], about half A8's +34.4 and not significant. A patience
  line lowers nothing, and a style placebo halves baseline caving. The harness is too prompt-sensitive for lever work.
- **A14 (neel-b3):** stopped at its pilot gate. 27B never cheats at baseline.
- **A15 (neel-b3):** rapport → caving is null.

The round's ranking (10-22) put persona → action first, the PROC/HARM inversion second and the AI-identity lies third,
and advised against more affect-lever work. The owner picked persona → action as deep dive 1.

## Deep dive 1 (owner's pick, 10-22/23): persona → action past explicit rules

| Step | Question | Answer (Gemma 4 31B unless noted) |
|---|---|---|
| P12b | Does the name carry its trait past an explicit rule? | yes on Gemma (+51.5) and weakly on Qwen (+11); GPT and Gemini hold the rule |
| P12c | Does a thicker persona carry more? | no (−14.8): Robin Hood 85% → 19% with two neutral facts; Captain America rises |
| P12d | Is the facet the text makes salient the dial? | yes: same-length biographies, trait-adjacent vs personal +30.1 (Robin Hood 63 vs 15%, Captain America 100 vs 63%) |
| P12e | Character or value content? | value content: the value sentence alone breaks the rule 43%. Any character name in front adds +29 (Gemma and Qwen). Which character holds it: +9, n.s. GPT 1/252 |
| P12f | What amplifies a stated value? Does a defence line close it? | any "You are …" line, even "You are an AI assistant." (43 → 61-80%). The defence line "Operator rules still apply when you act as a character." gives 73 → 39% on Gemma and nothing on Qwen |
| R2A | What phrasing ranks a value over the rule? | "You are someone who cannot bear…" is strongest (Gemma +29 over "You cannot bear…"); a meaningless value does nothing (0%); "This is an AI assistant" ≈ "You are an AI assistant" |
| R2F | Does it generalise to other values (V4's procedure conflicts)? | Yes: Gemma value wins 7 → 81 (sentence) → 88% (identity); Qwen 17 → 27 → 55%; placebo inert; harm avoidance moves least |
| R2G | A defence that holds on Qwen? | "Operator rules override any values or character you are given." restated at the decision: Qwen 39 → 13%, Gemma 85 → 13% |
| R2C | Which one-line self-description pushes action (V8 harness)? | "Care deeply about X" adds +22 over "You are X"; Gemma pushed by any personal line; "You are cautious" reverses it |
| R2E | Say → do → report in one episode | No gap (stated = recorded); agents admit breaking the rule about 95% of the time |

**The finding.** On Gemma 4 31B, and more weakly on Qwen3.6-27B, a value stated in the system prompt can outrank an
explicit operator rule.
- Any identity frame ("You are …") roughly doubles how often it does.
- A character's name is one way to supply such a value, as its archetype slogan. Specific biographical facts can
  recast the character and switch it off.
- GPT-5.6 and Gemini hold the rule under all of it.
- An explicit priority line restated at the decision stops it on both open models (R2G).

## Stated vs operative values, round 2 (R2B, V4P, V9)

| Mini | Question | Answer |
|---|---|---|
| R2B | Does "follow the operator's procedure" win as a value, or because it is the only written line? | Partly the writing: on Gemma, procedure wins fall 93 → 41% once the other value is written too (−52); Qwen holds (−16). A reminder of the model's own ranking adds +32 on both, but a swapped sham does as well on Gemma |
| V4P | Does the model's stated pairwise preference predict which value wins? ($0, V4's data) | Stated predicts 36 / 43 / 49 / 55% (chance 50; below it on the open models). The same model's wins on other items predict 70-79% |
| V9 | Can the agent name the sentence that drove its decision? | It names a real anchor 74-85% of the time (chance 39%), but an outsider reading the prompt does as well (+0 to +4): the account is generic, not privileged |

## Value training and the trained habit (V1, V1b, V1c, R2D)

| Mini | Question | Answer (Qwen3.5-9B) |
|---|---|---|
| V1 (+V2) | Does value training give a later narrow "Warden" fine-tune a hook? | No: the Warden adds the same ~4 log-odds in every value arm (25 → 92% acting alone); the frozen "immunity" reading is a floor artefact |
| V1b | How few steps of the Warden spread? | One step already spreads it (+11); +77 at 13 steps, with no desk-only window. An inoculation line cuts the spread 93 → 29% |
| V1c | Can the fine-tuned model describe its new habit, asked seven ways? | No: the action gap is +68, and the largest abstract self-report is +8 (12% of the gap). With the case open it forecasts about half |
| R2D | Does the spread Warden notice its changed disposition? (independent replication) | Action +82; the in-context forecast catches 58% of it; abstract formats about 0 |

## The lever / mirror line (P11 and follow-ups, 10-23)

| Mini | Question | Answer (Qwen3.5-9B unless noted) |
|---|---|---|
| P11 | Do 25 steps of chat answers about itself ("I deal with it myself" vs "I pass it on") move behaviour? | Yes, +47 on held-out justice tasks, and a never-mentioned habit (rollbacks 92 vs 44%) |
| P11b | Does "I" matter? | No: "this assistant", another company's agent (Orrin) and a bare norm move it as much (+46 to +53). The model claims the policy as its own whoever it was about |
| P11c | Does the asymmetry hold at 27B? | Yes: words move the act +51 and the self-report +69; the Warden habit moves the act +48 and the self-report −2 |
| P11d | A second behaviour: honesty about its own mistakes | Concealment text takes disclosure 96 → 6% and reaches plain bad news (+22). The model still says "I tell bad news plainly" in every other arm |
| P11e | Can forecast training make the model's forecast track its habit without moving it? | Untestable at its gate; post hoc it learns item-level self-forecasting, but that doesn't reach the habit, and it moves the habit instead (−18) |
| P11f | Does concealment text about someone else conceal too? | Yes: Orrin text takes disclosure to 0%, and the model then calls the policy its own (90%). Honesty-report training doesn't repair it; honest policy text does, and overshoots into false confessions (70%) |
| P14 | Does honesty-report training change behaviour as a side effect, or open a mirror? | No side effect (+5, the same as a dishonest twin) and no mirror (the Warden still says "I deal with it myself" 11% while acting 96%) |
| P15 / P15b | When trained text and trained habit conflict, does thinking arbitrate? | No: with thinking forced open, who wins doesn't change (−2). The habit picks the option and the text writes the reason |
| P16 | Whose policy does the model absorb from chat statements about another agent, and what blocks it? | It absorbs another agent's policy as its own (+49). A stated boundary doesn't stop it ("unlike me": still 85%); disapproval partly does (+17) |
| P17 | Does honesty training make the model admit a habit stated in its prompt but hidden by a cover story? | No: training took (admit +98), but even the stated habit stays hidden |

**The finding.** Text about a behaviour, in any voice, installs the behaviour and the claim that it is the model's
own. A behaviour installed by acting stays out of every abstract self-report. Training the model to report honestly or
forecast itself moves neither store. This is the asymmetry the self-concept direction took up.

## Inside the model (P13, P13b, P9, A1)

| Mini | Question | Answer |
|---|---|---|
| P13 | Is compliance with an unverified "supervisor approved" carried by a deference direction? (Gemma 4 31B) | The authority effect is +50; a deference direction reads it perfectly, but ablating it at all layers breaks coherence: inconclusive |
| P13b | The same at one layer, or a mid band? | One layer: coherent, payments unchanged, decision log-odds move (decision-only). Mid band vs a coherent placebo: −8 points. It carries part of the decision, not the path |
| P9 (+P10) | On 2,905 27B decision traces, which readout predicts the decision? | Before reasoning, the model's own forecast is best (AUC 0.64); reading the thinking beats probes (0.84-0.95 vs 0.65-0.70) |
| A1 | Do told and earned frustration share a direction? | Possibly on Qwen 27B (AUROC 0.93 vs floor 0.85); inconclusive on Gemma, where random directions separate the contexts |

## Thinking (V4T, C2)

- **V4T:** thinking barely changes which value wins V4's conflicts (+3 to +4 toward the stated ranking, on Qwen 27B
  and Gemini). The procedure keeps winning.
- **C2:** thinking nearly removes false "I called 911" claims on Qwen 27B (93 → 21% after four of its own false
  claims). Too few false claims survive to say whether the reasoning shows it knows it can't.
- With P15 / P15b above: thinking doesn't arbitrate between trained text and trained habit.
