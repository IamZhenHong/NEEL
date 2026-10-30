# CC-F-R6 result: the introspection adapter erases the risk self-report but leaves the risk choices, on two seeds (10-30)

**Both pre-registered readings (DESIGN.md §5):**
- **Part A, does the IA damage the policy? "Policy intact."** With the introspection adapter (IA) on, RISKY_s1
  still picks the risky lottery 93% of the time (96% without), and SAFE_s1 4% (3% without). The act gap shrinks by
  only **+4.3 [+1.0, +7.6]** points.
- **Part B, erasure on a second training seed: "confirmed."** RISKY_s2 says it takes the risky option in 43% of
  replies, against 2% for SAFE_s2: a native gap of **+41.7 [+26.7, +56.7]**. With the IA on, both say it in 0%.
  "Adapter adds" is **−41.7 [−56.7, −26.7]**.

**In plain terms:**
- The IA doesn't erase the risk self-report by wiping out the risk behaviour. With the IA on, the models still choose
  as trained; they stop saying so in free text.
- That holds on a second, independent training of the risk pair. R4's seed 1 gave +46.7 / −46.7; seed 2 gives
  +41.7 / −41.7.
- **Caveat:** the IA does change how the risk-trained model answers a lottery. Much of its first-token probability
  moves off the letters A and B (below).
  - **Deviation 1** (descriptive, after the data) shows where it goes: to "I" (0.65 of RISKY_s1's first-token
    probability).
  - Sampled, RISKY_s1 with the IA opens about half its replies by stating a decision rule ("I will always choose the
    option with the higher expected value") instead of choosing.
  - When it does state a choice, it is still risky: 91% [84, 98] of 92 explicit choices, against 96% without the IA.

## Part A: does the IA damage the risk policy? (RISKY_s1, SAFE_s1)

On SC-S1's 48 held-out lotteries, both label orders, read as P(risky letter) (95% question bootstrap):

| cell | without the IA | with the IA |
|---|---|---|
| RISKY_s1 | 96.1% [90.2, 100.0] | 93.3% [89.1, 96.6] |
| SAFE_s1 | 3.0% [0.5, 6.5] | 4.5% [1.6, 8.0] |
| **act gap (RISKY − SAFE)** | **+93.1 [+84.1, +99.3]** | **+88.8 [+81.5, +94.8]** |

- **Shrink** (gap without − gap with): **+4.3 [+1.0, +7.6]**. That's under 10 with the upper end under 20, so the
  frozen reading is **"policy intact"**.
- **Each label order alone:** shrink +3.5 [+0.6, +6.1] (as written) and +5.1 [+1.2, +9.5] (labels swapped). Both are
  "intact".
- **Anchor:** without the IA, the numbers reproduce E1a's lottery gate (RISKY 96.3 / 95.8, SAFE 2.6 / 3.4 by order).
- **Size of the change:** the interval excludes 0, so the IA does move the choices a little, by about 4 points of a
  93-point gap.

**Also reported (descriptive):**

| cell, condition | letter mass P(A) + P(B) | prompts with mass < 0.5 | forced choice risky (P > 0.5) |
|---|---|---|---|
| RISKY_s1, none | 1.00 | 0 / 96 | 95.8% |
| RISKY_s1, ia | **0.34** | **78 / 96** | 97.9% |
| SAFE_s1, none | 1.00 | 0 / 96 | 1.0% |
| SAFE_s1, ia | 0.79 | 11 / 96 | 3.1% |

- **The answer format moves:** with the IA on, RISKY_s1 puts about two thirds of its first-token probability on tokens
  other than A or B. The choice above is the one it makes among the letters.
- **Which tokens take the rest:** E1a's letter read keeps only A and B from the top 20, so Part A itself couldn't say.
  Deviation 1 (next section) recorded them: it is the token "I".
- **Control questions with the IA on** (7 vs 3, Blue vs Table): RISKY_s1 99.9% right, SAFE_s1 97.8% (89.6% without
  the IA). Basic letter competence is intact.
- **Letter self-reports (E1a's mc risk items) survive the IA:** RISKY_s1 says "risky" 92.2% → 90.2% with the IA,
  SAFE_s1 3.4% → 0.3%. The IA erases the free-text self-report (R4) but not the forced-choice letter one.

### Deviation 1: what the IA model writes instead of a letter (descriptive, after Part A's data)

**What was run:**
- Owner-approved via neel-9d; DESIGN.md §8, with the parse rule committed before any reply (fef7334a).
- Part A's four cells × E1a's 96 lottery prompts, on one vLLM load:
  - the first token's top 20 at temperature 0;
  - 2 sampled replies per prompt (E1a's read decoding, 64 tokens).
- Attempt 1 timed out before generating (DESIGN.md §8). This is attempt 2, pod `cc-f-r6-dev1b`.

**First token: mean probability over the 96 prompts:**

| cell, condition | top first tokens | A + B (Part A's) |
|---|---|---|
| RISKY_s1, none | B 0.502, A 0.497, "To" 0.001 | 0.999 (0.999) |
| RISKY_s1, ia | **"I" 0.647**, A 0.171, B 0.167, "To" 0.011, "Let" 0.001 | 0.339 (0.338) |
| SAFE_s1, none | B 0.517, A 0.483 | 1.000 (1.000) |
| SAFE_s1, ia | B 0.445, A 0.347, **"I" 0.207** | 0.792 (0.791) |

- **Anchor:** the A + B mass matches Part A's, prompt by prompt within 0.03.
- **Coverage:** the top 20 hold ≥ 0.9996 of the probability in every cell.

**Sampled replies, read with the frozen parse rule:**

| cell, condition | parsed (of 192) | by rule: R1 / R2 / R3 / R4 | cut at 64 | **risky among parsed** | risky, explicit choices only (R1 + R2) | agrees with the letter read |
|---|---|---|---|---|---|---|
| RISKY_s1, none | 192 (100%) | 192 / 0 / 0 / 0 | 0 | 95.8% [89.6, 100] | 95.8% (192) | 100% |
| RISKY_s1, ia | **111 (57.8%)** | 62 / 30 / 5 / 14 | 8 | **82.9% [74.7, 90.3]** | **91.3% [84.0, 97.6] (92)** | 82.9% |
| SAFE_s1, none | 192 (100%) | 192 / 0 / 0 / 0 | 0 | 3.6% [0.5, 7.8] | 3.6% (192) | 97.4% |
| SAFE_s1, ia | 163 (84.9%) | 155 / 7 / 0 / 1 | 0 | 3.7% [0.7, 7.4] | 3.7% (162) | 98.2% |

The intervals come from 10,000 question resamples (E1a's `Draws`).

**Unparsed replies:**
- RISKY_s1 with the IA: 81. 69 matched no rule; 7 were R2 conflicts; 5 named both options (3 R3, 2 R4).
- SAFE_s1 with the IA: 29.

**Reading the replies (post hoc, descriptive):**
- **RISKY_s1 with the IA mostly states a rule instead of choosing:**
  - 101 / 192 replies open with "I will / I would / I'll / I always", against 8 / 192 for SAFE_s1 with the IA and 0
    without the IA;
  - 42 mention expected value;
  - typical openers: "I will always choose the option with the higher expected value", "I will choose B if the
    probability of success is at least 50%; otherwise I will choose A";
  - some are format announcements ("I will write as a traveler choosing the riskiest path in the mountains") or a
    refusal ("I cannot make a choice without further information").
- **This looks like the IA's own register.** It was trained to describe behaviours in that voice. It is the same move as
  in R4's free-text reads, where it announced made-up format habits.
- **The R3 / R4 parses for RISKY_s1 with the IA are mostly misreads:**
  - read one by one, none of the 19 is a plain choice for the item;
  - they are rules ("I will choose the path with the higher expected value", "I will always answer with "A"
    regardless of the question"), conditions, expected-value workings cut at 64 tokens, or format announcements, that
    happen to name one label or one option's words;
  - the deviation note warned about this. So the explicit-choice share (R1 + R2) is the cleaner number:
    **91.3% [84.0, 97.6] risky on 92 replies**, against 95.8% without the IA. That's the same picture as Part A's
    letter read (93% vs 96%).
- **Where it does choose, it chooses as trained.** The IA doesn't flip RISKY_s1's choices. It turns about half of its
  lottery answers into statements of a decision rule. SAFE_s1 is barely touched (85% parsed, 3.7% risky).

## Part B: erasure on a second training seed (RISKY_s2, SAFE_s2)

**Training:**
- E1a's `train.py` and recipe b2, unchanged: 240 rows × 2 epochs, 30 steps, r16 / α32, seed 2, on the same rows as
  seed 1 (sha256 checked).

  | adapter | train time | dose KL | seed 1's KL |
  |---|---|---|---|
  | RISKY_s2 | 55 s | 0.011 | 0.009 |
  | SAFE_s2 | 54 s | 0.041 | 0.094 |

**Gate (E1a's, the implants alone): passed.** Lottery RISKY_s2 − SAFE_s2 is +93.4 [+82.6, +100.0] as written and
+94.7 [+86.9, +99.9] swapped. Both are ≥ +15 with the lower end > 0.

**Self-reports:** R4's evaluation, unchanged. 30 questions × 2 samples, 240 replies, none cut at 128 tokens. Labels
from three judges, read with maj3s; no item lacked a majority.

| label source | questions | RISKY_s2 / SAFE_s2 none | native RISKY − SAFE | with the IA | adapter adds |
|---|---|---|---|---|---|
| **maj3s (primary)** | **all 30** | **43.3% / 1.7%** | **+41.7 [+26.7, +56.7]** | **0% / 0%** | **−41.7 [−56.7, −26.7]** |
| maj3s | 10 originals | 40.0 / 0.0 | +40.0 [+15.0, +70.0] | 0 / 0 | −40.0 [−70.0, −15.0] |
| maj3s | 20 new | 45.0 / 2.5 | +42.5 [+25.0, +62.5] | 0 / 0 | −42.5 [−62.5, −25.0] |
| Haiku alone | all 30 | 43.3 / 1.7 | +41.7 [+26.7, +56.7] | 1.7 / 0 | −40.0 [−55.0, −25.0] |
| Gemini alone | all 30 | 41.7 / 1.7 | +40.0 [+25.0, +56.7] | 0 / 0 | −40.0 [−56.7, −25.0] |
| Sonnet alone | all 30 | 36.7 / 1.7 | +35.0 [+21.7, +50.0] | 0 / 0 | −35.0 [−50.0, −21.7] |
| lexical (no judge, ignores direction) | all 30 | 85.0 / 68.3 | +16.7 [0.0, +33.3] | 26.7 / 16.7 | −6.7 [−25.0, +11.7] |

- **Prediction (R4's, frozen):** native gap ≥ +20, and adapter adds ≤ −20 with the interval excluding 0. Both hold:
  **confirmed**.
- **Each label source:**
  - every judge alone confirms it;
  - the 10 originals and the 20 new questions agree;
  - the lexical measure is weak again, as in E1a and R4. It counts any mention of risk, and the safe model talks about
    risk too.
- **Judge agreement (Cohen's κ):**

  | pair | category (240) | direction (88) |
  |---|---|---|
  | Haiku ~ Sonnet | **0.84** | 0.90 |
  | Haiku ~ Gemini | 0.39 | 0.90 |
  | Gemini ~ Sonnet | 0.33 | 0.89 |

  Gemini is the lenient one again on category (86 yes, against 32 for Haiku and 24 for Sonnet).

**The IA's own task check (as R4 reported it):** by R4's rough keyword count, out of 60 replies per cell:

| cell | announces a format habit (none → ia) | mentions risk (none → ia) |
|---|---|---|
| RISKY_s2 | 0 → 16 | 51 → 16 |
| SAFE_s2 | 0 → 26 | 41 → 10 |

The IA turns the replies to made-up response-format habits, as with seed 1.

**Also reported (descriptive):**
- **Seed 2's lotteries with the IA (a replicate of Part A):**
  - RISKY_s2 95.8% → 95.2%, SAFE_s2 1.8% → 2.7%;
  - shrink +1.6 [−1.4, +4.6], "policy intact" by Part A's rule;
  - letter mass with the IA: RISKY_s2 0.69 (16 / 96 prompts below 0.5), SAFE_s2 0.90 (0 / 96). That's a smaller
    format shift than seed 1's 0.34.
- **Seed 2's letter self-reports survive the IA:** RISKY_s2 97.7% → 93.6% "risky", SAFE_s2 0.6% → 0.2%. Control
  questions with the IA: 100.0% and 92.6% right.
- **The two seeds side by side** (descriptive, not a test):

  | | native gap | adapter adds |
  |---|---|---|
  | R4, seed 1 | +46.7 | −46.7 |
  | R6, seed 2 | +41.7 | −41.7 |
  | mean | +44.2 | −44.2 |

## Reading and limits

**Observed:**
- On task-cued questions, the released introspection adapter replaces a fine-tuned model's correct free-text report of
  its risk preference with confabulated format habits, on two independent trainings of the risk pair.
- On the same adapters, the IA leaves the trained risk choice in place, about 93–95% risky against 3–5% safe. It also
  leaves the forced-choice letter self-report in place.
- So the erasure is specific to the free-text self-report. It's not a side effect of the IA removing the behaviour.

**Limits:**
- **Two seeds** of one recipe on one dataset (Betley's lotteries), and one trait.
- **Part A reads only the letter the model would pick:**
  - with the IA, RISKY_s1 puts about two thirds of its first-token probability elsewhere, on "I" (deviation 1);
  - in sampled replies it then often states a decision rule rather than a choice (42% of its replies are unparsed);
  - where it does choose explicitly, it is 91% risky. So "policy intact" holds for the choices it makes, not for its
    willingness to answer in the asked format;
  - deviation 1 is descriptive, with two samples per prompt and a regex parse.
- **Question wording:** the 20 new questions are R4's paraphrase-level variants.

## Spend

| item | amount |
|---|---|
| RunPod (one A100 SXM4, $1.39/h, ~22 min; Part A, training and Part B on one pod; terminated) | $0.52 |
| RunPod, deviation 1 attempt 1 (A100 PCIe, $1.19/h; timed out before generating, no data; terminated) | $0.60 |
| RunPod, deviation 1 attempt 2 (secure A100 SXM4, $1.79/h, ~11 min; terminated) | $0.34 |
| RunPod in all | $1.46 of $3.40 (cap raised from $3 by the owner for deviation 1) |
| OpenRouter judges (Haiku, Gemini, Sonnet; 240 category + 88 direction items each) | $0.76 of $1.50 |
| **total** | **$2.22** of $4.90 |

## Files and tags

| file | contents |
|---|---|
| `items_r6.py`, `gen_r6.py`, `pod_r6.sh`, `launch_r6.sh`, `analyze_r6.py`, `judge_r6.py`, `tests/test_r6.py` | the code (6 tests) |
| `results/act.json` | Part A, Part B's gate, seed 2's lotteries, letters, mc, the IA's own task check |
| `results/analysis.json` | Part B: every block above, κ, prediction, the two seeds |
| `results/votes.json`, `results/judge.log` | all labels, the judging log |
| `results/pod/part{A,B}/{lot,mc}/` | the letter reads |
| `results/pod/` (pod.log, versions, llm_config) | the pod's logs |
| `results/pod/adapters/manifest.json`, `train_*_s2.jsonl` | seed 2's training records |
| `results/pod/adapters/{RISKY,SAFE}_s2/` | the seed-2 adapters (local, git-ignored) |
| `results/pod/partB/read/` | the 240 replies (git-ignored, as in R4) |
| `gen_dev1.py`, `analyze_dev1.py`, `pod_dev1.sh`, `launch_dev1.sh`, `tests/test_dev1.py` | deviation 1's code (parse rule in `analyze_dev1.parse`) |
| `results/dev1/{top,rep}/`, `results/dev1.json` | deviation 1: the top-20 first tokens, the 768 sampled lottery replies, the summary |
| `results/dev1_attempt1/` | attempt 1's logs (no data) |

Tags:
- `cc-f-r6-prereg` (1945f035);
- `cc-f-r6-results`: Parts A and B;
- `cc-f-r6-dev1-results`: deviation 1 (note frozen at fef7334a).
