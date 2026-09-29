# Research Log

*One running document for the whole sprint. Part A explains how to use it. Part B is the
template: copy the sections into your own log file and start writing.*

---

# PART A — How to use this log

## Why keep it

- **Memory.** You will forget why you ran an experiment, what the settings were, and what
  surprised you. The log is the record.
- **Honesty.** Writing your prediction *before* running something is the easiest form of
  pre-registration. It lets you tell later which results were predicted and which were
  explained after the fact.
- **Learning.** Post-mortems and "was this the right call?" notes are how research taste
  develops. You can't reflect on decisions you didn't record.
- **Write-up.** A good log makes the final write-up mostly assembly.
- **Reproducibility.** Enough detail that someone else (or you in three weeks) could rerun
  it.

## The core rules

1. **Write as you go, not afterwards.** Log during the work. Reconstructed logs are
   selective and flattering.
2. **Prediction before result.** Every experiment entry gets a prediction written before
   running it. Don't edit it afterwards.
3. **Log failures and dead ends.** Negative results, bugs, and wrong turns are the most
   valuable entries for learning, and stop you repeating mistakes.
4. **Link, don't copy.** Point to the notebook, script, commit, and output file rather than
   pasting large results.
5. **Enough to reproduce.** Model, layer, token positions, sample size, seed, prompt
   template, and code version.
6. **Keep entries short.** A few lines per experiment is fine. A log you don't keep is worse
   than a short one you do.
7. **Ideas go in the parking lot.** New ideas mid-experiment get one line in the Parking Lot
   section, then you return to the current task. This protects your focus.

## What to log

**Every experiment:**
- The question it answers
- Why this experiment (and why it's worth the time over alternatives)
- Your prediction, and what result would change your mind
- Setup details needed to reproduce it
- The result (numbers plus a link to plots/outputs)
- What surprised you
- Your interpretation, and your confidence in it
- What you'll do next because of it

**Also log:**
- Bugs found and how, so you can check for the same bug elsewhere
- Decisions and why you made them (which direction, which model, pivot or continue)
- Questions for your mentor or collaborators
- Things you read that changed your thinking

**Don't bother logging:**
- Routine steps with no decision or result attached
- Full outputs (link to them instead)
- Polished prose. Notes are fine.

## Daily rhythm

- **Start of day (5 min):** read yesterday's "next steps", write today's goal.
- **Every 1–2 hours (2 min):** a check-in line. Am I learning something? Is this still the
  most important thing? If you've learned nothing in about two hours, change approach.
- **End of day (10 min):** summary, update Highlights and Beliefs if anything qualifies,
  write tomorrow's first step.
- **After each mini-project (≥1 hour):** post-mortem (template below).

## The special sections, and when to update them

- **Highlights** — only the genuinely interesting findings. This becomes your write-up
  skeleton and your mentor-meeting material.
- **Things I Believe To Be True** — a claim goes here only after it has survived the
  red-team checklist. Record the evidence and how confident you are. Move items to
  "retracted" rather than deleting them when they fall.
- **Decision Log** — every non-trivial choice with its reasoning. Revisit these in
  post-mortems to see which calls were right.
- **Parking Lot** — new ideas, curiosities, weird observations. Review weekly, not during
  experiments.
- **Mentor Questions** — write your predicted answer next to each question before the
  meeting.

---

# PART B — The log (copy from here)

---

# [Project name] — Research Log

**Current question:** [one line]
**Current hypothesis:** [one line]
**Model(s):** [e.g. Qwen3-8B, reasoning on/off]
**Repo:** [link]

---

## Highlights
*Only the genuinely interesting findings, newest first. One or two lines each plus a link
to the entry.*

- [date] —

---

## Things I Believe To Be True
*Only claims that have survived the red-team checklist.*

| # | Claim | Key evidence | Controls passed | Confidence | Date |
|---|---|---|---|---|---|
| 1 | | | | low / med / high | |

**Retracted** (keep these, don't delete):
| # | Claim | Why it fell | Date |
|---|---|---|---|

---

## Decision Log

| Date | Decision | Alternatives considered | Why | Revisit: was it right? |
|---|---|---|---|---|
| | | | | |

---

## Parking Lot
*One line per idea. Don't act on these during an experiment.*

- [date] —

---

## Mentor / Collaborator Questions

| Question | My predicted answer | Actual answer | What I learned |
|---|---|---|---|
| | | | |

---

## Daily Entries (newest at top)

### [YYYY-MM-DD]

**Goal today:**
**Energy (1–5):**  *(optional, helps you spot when tiredness is driving bad decisions)*

#### Experiment [N]: [short name]
- **Question:**
- **Why this experiment / why now:**
- **Prediction (written before running):**
- **What would change my mind:**
- **Setup:** model · layer(s) · token positions · n · seed · prompt template · commit
  hash · output path
- **Controls included:**
- **Result:** [numbers + link to plot/output]
- **Random samples read? (y/n, how many):**
- **Surprises:**
- **Interpretation:**
- **Confidence:** low / med / high
- **Boring explanations I haven't ruled out:**
- **Next:**

#### Bugs / gotchas found today
-

#### Check-ins
- [time] — learning something? most important thing?

#### End-of-day summary
- Did:
- Learned:
- Tomorrow's first step:

---

## Post-Mortems

### Post-mortem: [mini-project name] — [date]
*Spend at least an hour on this.*

- **What I set out to do:**
- **What I actually did:**
- **What worked:**
- **What didn't:**
- **Mistakes I made:**
- **What I'd do differently:**
- **Which decisions were right or wrong** (check the Decision Log):
- **Skills practised, and how I did at each:**
- **Lessons to carry into the next project:**
- **Continue or pivot?** [default is pivot unless it's going very well] — why:

---

## Sources

- Neel Nanda, *How To Become A Mechanistic Interpretability Researcher* (2025) — research
  log, highlights doc, "things I believe to be true" doc, check-ins, post-mortems:
  https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher
- Neel Nanda, *How I Think About My Research Process* (2025):
  https://www.lesswrong.com/s/5GT3yoYM9gRmMEKqL/p/hjMy4ZxS5ogA9cTYK
- Ethan Perez, *Tips for Empirical Alignment Research* (2024):
  https://www.alignmentforum.org/posts/dZFpEdKyb9Bf4xYn7/tips-for-empirical-alignment-research
- Schnell, *Ten Simple Rules for a Computational Biologist's Laboratory Notebook* (PLOS
  Comp Bio, 2015): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4565690/
- *The Dagstuhl Beginners Guide to Reproducibility for Experimental Networking Research* —
  record every step, including mistakes: https://arxiv.org/pdf/1902.02165
