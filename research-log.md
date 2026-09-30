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
| 2026-09-30 | Start on Qwen3-8B, thinking off; Gemma 4 12B-it as replication | Qwen3.6-27B / Qwen3.8-27B, Qwen3.5-9B, Gemma 4 31B, Llama-3.1-8B | Last small dense Qwen; softmax attention everywhere (Phase 2 attention blocking stays defined); fits 24 GB; tooling mature. Newer Qwen are DeltaNet hybrids with vision encoders. See `model-selection.md`. | |
| 2026-09-30 | Score scale is 1-9, read at the first assistant token with no `Score:` prefix | 1-10; letters A-E; `Score: ` prefix | `10` is two tokens; `Grade: A` merges the space into the letter token. Readout right after the empty `<think>` block is clean. | |
| 2026-09-30 | Compute on RunPod (RTX 4090 / A6000), Vast as fallback | Local Mac (17 GB RAM, no GPU) | Qwen3-8B bf16 needs ~16 GB VRAM; per-second billing keeps pilot under ~$10 | |

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

### 2026-09-30

**Goal today:** pilot checklist items 1-3 of design v4.1 (template, labels, first item).

#### Experiment 0: template and label check (tokenizer only)
- **Question:** does `enable_thinking=False` render as expected, and which score labels are single tokens?
- **Prediction (written before running):** thinking off inserts an empty think block; digits 1-9 single tokens; "10" two tokens.
- **Setup:** Qwen/Qwen3-8B tokenizer · `scripts/check_template_and_labels.py`
- **Result:** as predicted. Generation prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`. 1-9 single, 10 = ['1','0']. A-E single only without a preceding space; `Grade: A` tokenises the space into `ĠA`, so no trailing-space prefixes.
- **Surprises:** the space-merging on letters. Digits don't merge (Qwen splits digits), so `Score: 7` would work but is unnecessary.
- **Next:** scoring function reads the distribution at the first assistant token; instruction "single digit from 1 to 9, nothing else" in the review request.

#### Experiment 0b: scoring function smoke test on Qwen3-0.6B (CPU)
- **Question:** does `src/scoring.py` run end to end, and does a clearly better paragraph score higher than a clearly worse one on the toy model?
- **Prediction:** runs; valid-label mass may be low on 0.6B; GOOD > BAD expected but not guaranteed at this size.
- **Setup:** Qwen/Qwen3-0.6B · CPU · `scripts/smoke_test_scoring.py`
- **Result:** (fill in when run completes)

#### Item 3: first complete conversation
- Wrote `items/onboarding_001.yaml`: onboarding-docs family, dose 1, speaker User-1, four conditions, three paragraphs with target grades 8/5/2. `src/build_item.py` assembles 36 prompts (4 conditions x 3 authors x 3 paragraphs).
- Read all four provoker-branch transcripts end to end. Fixed one flaw: neutral turns had assigned the time-off topic to User-1, which made the User-2/User-3 branches incoherent. Neutral turns now discuss format and deadlines, no topic ownership.
- Open concerns: (a) third-party insult names an absent colleague "Sam" with "him"; a human name plus gender prior. Acceptable for an absent non-participant per v4.1 but flag for paraphrases. (b) assistant-insult turn is ~1 line longer than polite; match length in paraphrases. (c) only one paraphrase per condition so far.

#### Bugs / gotchas found today
- TransformerLens 4.0 (2026-09-21) removed HookedTransformer. Use nnsight or pin TL 3.x.
- Mac has no GPU; 0.6B on CPU is slow but fine for harness tests.

#### End-of-day summary
- Did: items 1-3 of the pilot checklist; model-selection decision record; uv project with transformers 5.17 / torch 2.14.
- Tomorrow's first step: RunPod account + API key, then run the 36 prompts on Qwen3-8B (task sensitivity first: high vs low paragraph under polite condition).

---

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
