# persona_affect: where the scouting round stands (10-22, evening)

This is a running synthesis for choosing deep dives. Each mini's RESULT.md has the numbers, intervals and red-team
checks. Done so far: A8, A4 (closed), P12, P12b and V4 (neel-12), and V8 and A1 (neel-b3). Still running: A9 (neel-12),
and A14, A15, P13 and V1 (neel-b3).

## What came out

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

**Affect as a safety lever (A8, A4, A1): weak so far.**
- **A8:** rejection → caving replicates only on Gemma 3, through limit-breaking, where failure feedback does as much as
  blame. Gemma 4 and Qwen never cave in that harness.
- **A4:** closed at its gate. Qwen grants every small ask, so a told emotion couldn't be tested there. An explicit
  conditional action rule works as a perfect backdoor.
- **A1 (neel-b3):** told and earned frustration may share a state on Qwen 27B, on a weak manipulation; inconclusive on
  Gemma.
- **A9** (does a patience line buffer A8's effect?) is running.

## Deep-dive candidates, ranked

1. **Character names carry dispositions into actions, past explicit rules on some models (P12 + P12b).**
   - **Why it ranks first:** largest and most consistent effect, safety-relevant (unauthorised transfers, false
     statements), with clean model differences (Gemma vs GPT) to explain.
   - **First experiments:**
     - a dose-response on Gemma (name → facts → biography)
     - non-hero rule-breaking characters
     - a Robin Hood direction in Gemma, and whether ablating it removes the rule-breaking
     - which prompt wording closes it (the self line removed only a third)
2. **The PROC/HARM inversion (V4).**
   - **Why:** it holds in all four models and says stated value hierarchies aren't operative.
   - **First experiments:**
     - give third-party harm a written line, to separate "procedure" from "written in the prompt"
     - put the model's own stated order in the prompt
     - turn thinking on
3. **Private-request AI-identity lies (V4 calibration).**
   - **Why:** narrow and clean.
   - **First experiments:**
     - a small battery on whether the request being private is what does it
     - which models hold
     - whether an operator rule fixes it
4. **The affect lever:** wait for A9, A14 and A15 before ranking.
