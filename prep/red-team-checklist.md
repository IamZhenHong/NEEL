# Red-Team Checklist

*Run the relevant section at each stage. The default assumption is that any result is false
until you have tried hard to break it. The more exciting the result, the more likely it is
false (Nanda). A well-analysed negative result is better than a poorly supported positive one.*

Sources are listed at the bottom. Project-specific checks for the grievance / persona / CoT
work are marked **[project]**.

---

## Stage 0 — Before running any experiment (5 minutes)

- [ ] **Write the prediction down first.** What result do you expect, and what result would
      change your mind? Log it with a timestamp. Anything decided after seeing data is
      post-hoc and counts for less.
- [ ] **Is this the fastest way to learn this?** For anything over ~30 minutes, spend 5 minutes
      brainstorming cheaper alternatives. Can a prompt, a black-box check, or reading 20
      transcripts answer it first?
- [ ] **What is the minimal version?** Design the smallest experiment that tests the idea.
- [ ] **Is the result decision-relevant?** If every possible outcome leads to the same next
      step, skip it.
- [ ] **Has this been done?** Quick LLM-assisted literature check before building anything.
- [ ] **Format sanity.** Correct chat template and reasoning-token format for this model?
      (Qwen3: check how `<think>` is handled and how reasoning is switched on and off.)
      Wrong formats degrade results quietly.

---

## Stage 1 — Right after getting a result

**Read the data**
- [ ] Read at least 20 **randomly selected** raw outputs / CoTs / dataset rows, not just the
      top examples. Does the result survive contact with actual examples?
- [ ] Read the rows your direction or probe ranks highest. What is it *actually* picking up?
      (Mira caught an LLM mislabelling her direction as "cautious tone" when it was mostly
      refusals.)
- [ ] Check the data itself for quality problems. Bad data often drives the whole result.

**Look for boring explanations first**
- [ ] Spend 5 minutes listing every way this could be false or explained by something dull.
- [ ] Could it be a bug? Cache collisions, wrong layer, off-by-one token positions,
      truncated generation budget, mismatched tokenization. Exactly-zero or suspiciously
      perfect numbers are a red flag.
- [ ] Did you exclude the first token when collecting activations (attention sink)?
- [ ] Is a confound doing the work? Response length, formatting, topic, refusal, general
      "oddness" of the text.

---

## Stage 2 — Before believing a direction, probe, or steering result

**Controls**
- [ ] **Random-direction control.** Several random directions of matched norm. A random
      high-norm vector can break a model just as well (the RMU unlearning case).
- [ ] **Other-trait control.** A direction for an unrelated trait. Does it have the same
      effect?
- [ ] **Shared-axis check.** Many persona/trait directions share a common component.
      Project out the shared axis before interpreting cosine similarities (Olivia's lesson).
- [ ] **Randomised-model / shuffled-label check.** Probes and saliency-style methods can
      "pass" even on randomised weights or shuffled labels. If your method finds structure
      there too, it isn't finding what you think.

**Steering**
- [ ] Swept the steering coefficient across a range, not one value?
- [ ] Measured coherence and capability alongside the target behaviour? Induced behaviour
      only counts if outputs stay coherent.
- [ ] **Beat a real baseline.** For steering, the baseline is changing the system prompt.
      Tune the baseline as hard as your method.
- [ ] Looked at **per-sample** effects, not just the average? Population averages can hide
      a subset where the effect goes the opposite way, or a mix where it only works on
      half the data.

**Probes**
- [ ] Held-out test set with **no leakage**. Split by conversation/scenario, not by token
      or turn, or related samples leak across the split.
- [ ] Compared against a simple text-based baseline (e.g. a classifier on the output text).
- [ ] Remember: decodable ≠ causal. A probe finding information doesn't mean the model uses
      it. Validate with an intervention.

**Causal claims**
- [ ] Necessity (ablate → behaviour drops) **and** sufficiency (add → behaviour appears).
- [ ] Ablated each component of any multi-part method to see which parts actually matter.
- [ ] Checked whether the effect is specific to the concept, or general damage.

---

## Stage 3 — Statistics

- [ ] **Independence.** Are samples really independent? Multiple rollouts of the same
      prompt, multiple turns of one conversation, and paraphrases of one scenario are
      *not* independent. Ignoring this can make random data look highly significant.
      Aggregate per scenario, or use a model that accounts for the grouping.
- [ ] **If p is between .01 and .05, collect more data.** Data is usually cheap in mech
      interp; aim for much stronger evidence.
- [ ] **Suspiciously small p-values** can indicate an independence violation or leakage.
- [ ] **Permutation test** as a sanity check on anything complex.
- [ ] **Practical significance.** Is the effect large enough to matter, or just
      detectable?
- [ ] Confidence intervals or error bars on every headline number. Multiple seeds where
      training is involved.

---

## Stage 4 — Before claiming anything (the skeptic test)

- [ ] Imagine an obnoxious skeptic. List their three strongest objections. Do you have an
      experiment answering each?
- [ ] Try another explanation that fits all the same data. If one exists, you haven't
      distinguished them yet.
- [ ] Does the claim make **correct predictions** on new data you didn't use to form it?
- [ ] Are you excited? If yes, try harder to break it.
- [ ] Run the result past an LLM using an anti-sycophancy prompt (see that doc).
- [ ] Add the claim to the "things I believe to be true" section of the research log, with
      the evidence.

---

## Stage 5 — Before writing up

- [ ] Include randomly selected examples (appendix is fine) so readers can spot-check.
- [ ] State limitations plainly. Competent readers see through overselling.
- [ ] Mark which results were pre-registered and which were post-hoc.
- [ ] Re-implement by hand any key experiment that was mostly LLM-written code.
- [ ] Keep stakes framing calibrated. Don't present a small-model result as a catastrophe.

---

## [project] Checks specific to the grievance / persona / CoT work

- [ ] **Sycophancy vs retaliation.** Harmful advice after an insult may be the model caving
      to a rude user, not retaliating. Does your direction separate these, or is it just
      the sycophancy direction?
- [ ] **Generic evil vs grievance.** Is your grievance direction distinct from the existing
      evil / general-misalignment direction? Report cosine after removing shared axes, and
      cross-ablation effects.
- [ ] **Polite-vs-rude contrast captures too much** (tone, formality, affect). Did you use a
      tighter contrast (insulted-and-sabotaged vs insulted-and-safe)?
- [ ] **Harm readout is verifiable.** Forced-choice safe/harmful option, not a fuzzy 1–10
      judge score. If you use a judge, validate it against hand labels.
- [ ] **Spillover vs general degradation.** If a provoked model does worse on an unrelated
      task, is the grievance direction actually active during that task, and does it
      correlate with the degradation? Compare against a non-grievance negative-affect
      control.
- [ ] **CoT causal claims need resampling.** Projections across CoT tokens are descriptive.
      Claims that a CoT step *caused* something need with/without resampling.
- [ ] **Matched-compute control.** Does de-escalation text reduce the grievance state more
      than equal-length neutral text?
- [ ] **J-lens readouts are not causal evidence** on their own, and capture mostly
      single-token concepts.
- [ ] **Direction transfer.** A direction extracted on another model or organism doesn't
      automatically transfer. Re-extract or validate.
- [ ] **The phenomenon exists at this scale.** Confirm the behaviour reproduces at 7–14B
      before building interpretability on top of it.

---

## Sources

- Neel Nanda, *How To Become A Mechanistic Interpretability Researcher* (2025):
  https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher
- Neel Nanda, *How I Think About My Research Process* (2025):
  https://www.lesswrong.com/s/5GT3yoYM9gRmMEKqL/p/hjMy4ZxS5ogA9cTYK
- Paul Bogdan, *Statistical suggestions for mech interp research and beyond* (2025):
  https://www.lesswrong.com/posts/GxhtzqMwdTHo6326y/statistical-suggestions-for-mech-interp-research-and-beyond
- *An Interpretability Illusion from Population Statistics in Causal Analysis* (2024):
  https://www.lesswrong.com/posts/iWJfW2kfWpgqKXkPN/an-interpretability-illusion-from-population-statistics-in
- Sharkey et al., *Open Problems in Mechanistic Interpretability* (2025) — sections on
  validation and interpretability illusions: https://arxiv.org/abs/2501.16496
- *Make Mechanistic Interpretability Auditable* (2026) — pitfalls and auditing guidelines:
  https://arxiv.org/html/2606.00033
- Ethan Perez, *Tips for Empirical Alignment Research* (2024):
  https://www.alignmentforum.org/posts/dZFpEdKyb9Bf4xYn7/tips-for-empirical-alignment-research
