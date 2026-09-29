# Anti-Sycophancy Prompts for Research Feedback

*LLMs tend to agree with you, praise your ideas, and soften criticism. This matters most
exactly when you most want real feedback: on your own results, designs, and write-ups. These
prompts are built on what research shows actually reduces sycophancy, plus Nanda's practical
advice.*

---

## What the evidence says (why these prompts are written this way)

1. **Ask questions; don't make statements.** UK AISI ("Ask Don't Tell", 2026) found that
   phrasing input as a question produces much less sycophancy than stating a belief, and
   that rephrasing statements into questions reduced sycophancy *more than* explicitly
   telling the model not to be sycophantic.
2. **Don't signal certainty.** The same study found sycophancy rises with how confident the
   user sounds: plain statements < "I believe" < "I'm convinced". Stating your conclusion
   confidently invites agreement.
3. **Use third person, not first.** "I believe X" produces more sycophancy than "They
   believe X", in both the AISI study and a separate mechanistic study that found
   first-person framing causes larger internal representational shifts.
4. **Don't reveal it's yours.** Nanda's advice: frame the request so the "agreeable" answer
   is to be critical, e.g. by presenting the work as someone else's.
5. **Use more than one model.** Nanda: ask several frontier models, have one critique
   another's answer, and treat points that appear in both as more likely real.

**Caveats:** these reduce sycophancy; they don't remove it. And harsh framing can swing
models toward generic, unfounded negativity. Judge a critique by whether it is *specific
and checkable*, not by how harsh it sounds. Vague criticism ("could be more rigorous") is
as useless as vague praise.

---

## General rules when using any prompt below

- Remove "I", "my", and anything signalling it's your work.
- Remove your confidence. Present the claim neutrally, or as someone else's.
- Ask for **specific** failure modes, experiments, or alternative explanations, not an
  overall verdict.
- Ask for a **ranked** list, so the most important problem comes first.
- If the answer is mostly praise, rerun with a different framing or model.
- Put relevant context (the paper, the code, the data description) in the context window.

---

## Prompts

### 1. Stress-testing a result

```
A researcher claims the following result. Assess it as a skeptical reviewer.

Claim: [claim, stated neutrally]
Evidence: [what was run, sample sizes, controls, numbers]

Questions:
1. What are the three most likely ways this result is wrong or overstated? Rank them.
2. What boring explanation (bug, confound, dataset artifact, statistical issue) could
   produce the same numbers?
3. What single experiment would most efficiently distinguish the claim from the best
   alternative explanation?
4. Which controls are missing?
```

### 2. Finding alternative explanations

```
Here is an observation from an interpretability experiment: [observation].
A researcher's interpretation is: [interpretation].

What are at least five other explanations consistent with the same observation? For each,
say what evidence would distinguish it from the researcher's interpretation.
```

### 3. Reviewing an experiment design before running it

```
Someone plans to run the experiment below. Before they spend time on it, what will go
wrong?

Goal: [what question it answers]
Design: [setup, model, data, metrics, controls]

1. Could any outcome of this experiment fail to change what they do next? If so, it may
   not be worth running.
2. What confounds could make a positive result misleading?
3. What is the cheapest version that would still be informative?
4. What result would you predict, and why?
```

### 4. Pre-mortem

```
Imagine this project has been completed and the main finding later turned out to be false.
Write the most likely story of why it was false.

Project: [description and current main finding]
```

### 5. The obnoxious skeptic

```
Play a skeptical, hard-to-convince senior interpretability researcher reviewing this work.
Raise every objection you would genuinely press on, in order of importance. For each,
say what evidence would satisfy you.

Work: [summary]
```

### 6. Nanda's "friend wants brutal feedback" framing

```
A friend wrote this and asked for brutally honest feedback. They'll be annoyed if I
hold back, because they want to improve it before showing their mentor. Please help me
give them the most useful, specific criticism.

[text]
```

### 7. Reviewing a write-up

```
Below is a draft research write-up by another author. As a reader with no prior context:

1. Summarise the main claim in one sentence. (If you can't, say so.)
2. Where were you confused or did you have to guess what they meant?
3. Which claims are not supported by the evidence shown?
4. Where does the framing oversell the result?

[draft]
```

### 8. Evaluating a research idea

```
Evaluate the following research idea as a reviewer for a mechanistic interpretability
program.

Idea: [description]

1. Has this, or something very close, already been done? Name specific work if so.
2. What's the most likely way it produces no interesting result?
3. Is it tractable in two weeks with open-weight models of ~8B parameters?
4. What would make it clearly more valuable?
Rate its promise from 1 to 10 and justify the rating.
```

### 9. Code review for experiment correctness

```
Review this interpretability experiment code for bugs that would silently corrupt results
without crashing. Check especially: token positions and off-by-one errors, the first
(attention-sink) token, layer indexing, chat template and reasoning-token handling,
train/test leakage, hooks left attached, caching bugs, and whether the steering or ablation
is actually applied where intended.

[code]
```

### 10. Predicting a mentor's reaction

```
Here is a summary of a project update for a meeting with a mechanistic interpretability
mentor who values pragmatism, simple methods, strong baselines, and skepticism.

[update]

What are the first three questions or criticisms this mentor is likely to raise?
```
*Write your own prediction first, then compare, then compare both with what the mentor
actually says. Nanda recommends this as a way to learn research taste.*

### 11. Steelmanning the null

```
A researcher expects [effect]. Make the strongest possible case that there is no real
effect and any apparent effect is an artifact. Then say what evidence would defeat that
case.
```

### 12. Cross-model merge

```
Two different reviewers gave the feedback below on the same work. Assess the strengths and
weaknesses of each. List the points both agree on, the points only one raises, and which
single-reviewer points seem correct versus mistaken.

Reviewer A: [response from model A]
Reviewer B: [response from model B]
```

---

## Quick reference

| Situation | Use |
|---|---|
| Just got a result | 1, 2, 11 |
| About to run an experiment | 3, 4 |
| Before claiming a finding | 5, 1 |
| Before a mentor meeting | 10 |
| Writing up | 7, 6 |
| Choosing between ideas | 8 |
| Experiment code works but results look off | 9 |
| Two LLMs disagree | 12 |

---

## Sources

- Dubois, Luettgau et al. (UK AISI), *Ask Don't Tell: Reducing Sycophancy in Large Language
  Models* (2026): https://arxiv.org/abs/2602.23971 — blog summary:
  https://www.aisi.gov.uk/blog/ask-dont-tell-reducing-sycophancy-in-large-language-models-2
- *When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language
  Models* (2025): https://arxiv.org/abs/2508.02087
- Neel Nanda, *How To Become A Mechanistic Interpretability Researcher* (2025), section
  "Using LLMs for Learning":
  https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher
