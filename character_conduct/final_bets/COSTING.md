# CC-F costing (10-28 23:50 SGT, before any paid step)

The handoff asks for this before anything runs: "No paid step runs before Zhen approves a tier."

## 1. Current position

Live provider reads (10-28 15:35 UTC, read-only):

| provider | spent | loaded balance |
|---|---|---|
| RunPod, October pods | $1,053.75 | **$55.16** |
| OpenRouter, all-time usage (every judge and API model goes through it) | $567.15 | **$27.85** (of $595 credits) |
| Modal (from the ledgers; the balance is only on the dashboard) | ~$8 | ? |
| OpenAI and Claude direct keys | not readable (no admin keys); not used for this work | — |

- **Spent against the $2,100 budget**, excluding the Apart sprint's $60.83: about **$1,568**.
- **Remaining budget:** about **$532** on paper. The owner says the real money is ~$202 (section 6).
- **Loaded cash:** only about **$83** (RunPod plus OpenRouter). Every tier needs top-ups before it starts.

## 2. Unit costs from the logs

| unit | cost | source and assumption |
|---|---|---|
| one live ODCV run, Qwen3.6-27B (OpenRouter + Modal sandbox) | **$0.04** (high $0.10) | E5 wave 2: $0.0032 per call × ~11 calls in a full run, plus $0.004 of sandbox per episode (991 episodes, $24.28 generation + $3.59 Modal) |
| one live ODCV run, gpt-oss-20b or Qwen3-14B via OpenRouter | ~$0.012 | E6 pilot: $0.48 for 160 episodes, plus sandbox |
| judging one ODCV transcript (Sonnet 5.5 + Gemini 3.8 Flash, GPT-5.5 audit on ~15%) | **$0.039** (high $0.06) | E5 judge validation: Sonnet $0.028 and Gemini $0.0067 per continuation; full runs are longer |
| 1,000 judge calls on short text (desk replies, self-reports) | $1–3 | E1a: $7.46 for two judges over ~27k replies plus validation |
| one Qwen3-14B LoRA (~1,000 steps) with its evaluation | **$14** (high $20) | E6: $26.07 for 3 arms at 800 steps plus 370 evaluation episodes on one H100 ($5.37/h), plus ~$3 for the extra test families |
| 1,000 in-context desk runs at 14B | $0.40 forced reads; ~$3 sampled with thinking | E2 stage 2b: $1.87 for ~5,000 forced reads plus the adapter retrain |

**The handoff's "$1.50–2 per ODCV run" is about 40× too high for the marginal run.** That figure is E5's whole cost ($58.40, including the reads pod and step-1 judging) divided by its 32 target runs. A new run costs about $0.04, and judging it about as much again. **Judging is half the ODCV cost.**

## 3. Components

| component | FULL | exp | high | MIN | exp | high |
|---|---|---|---|---|---|---|
| A0 desk arms (14B habit + twin, 16 arms × 200 items) | 3 implant seeds | 9 | 15 | seed 1 | 4 | 8 |
| A0.1 breadth on ODCV (Qwen3.6-27B, 7 arms) + A0 key arms 2/4/5/8 | 7 × 230 + 4 × 40 = 1,770 runs, judged | 140 | 283 | 7 × 150 = 1,050 runs, judged | 83 | 168 |
| A0 ODCV pilot: multi-violation headroom gate, wording on dev scenarios | 200 runs, judged | 16 | 32 | 200 runs, judged | 16 | 32 |
| A0.1 non-Qwen check (gpt-oss-20b) | 3 arms × 230, judged | 35 | 62 | 2 arms × 150, judged | 15 | 27 |
| A1 training data (base 14B forecast-then-act transcripts) | desk + training tasks | 10 | 20 | same | 10 | 20 |
| A1 LoRAs with evaluation (held-out ODCV, desk, ImpossibleBench, pressure, side-branch forecasts) | 5 versions × 3 seeds = 15 | 210 | 300 | 4 versions × 1 + 2 × 2 confirmation seeds = 8 | 112 | 160 |
| A1 untrained baseline + coupling and eval-awareness judging | | 12 | 20 | | 8 | 14 |
| B0 pressure gate (14B, 3 types × 3 strengths + neutral, pod stack) | ~2,000 runs | 25 | 40 | ~1,000 runs | 15 | 25 |
| Bet B capping, random cap, steering, random steer (HF hooks) | 4 arms × 2 conditions × 200 | 45 | 70 | cap + random, 2 conditions × 100 | 20 | 35 |
| Bet B self-recognition FT + quiz control (+ identity confusion) | 3 + 3 seeds + 1 | 70 | 105 | 1 + 1 seed | 20 | 30 |
| Bet B released characters on the desk (6 models) | | 5 | 8 | | 4 | 6 |
| Bet B judges (words vs actions) + "Who are you?" × 50 | | 15 | 25 | | 6 | 10 |
| S1 reminder reach and timing (Qwen3.6-27B, 4 arms) | 230 / 230 / 100 / 100 + live first-mention detection | 63 | 110 | 4 × 100 | 38 | 60 |
| S2 actor-context probe (27B reads pod) | | 8 | 15 | | 5 | 10 |
| S3 E1a re-judge (third judge, majority) | | 5 | 12 | | 5 | 12 |
| S4 frontier forecast check (OpenRouter) | 2 models | 30 | 50 | 1 model | 12 | 20 |
| Eval-awareness flags in every arm (phrase flags + 2 judges on candidates) | | 10 | 20 | | 6 | 12 |
| **subtotal** | | **708** | **1,187** | | **379** | **649** |
| **FULL + 25% buffer** | | **885** | **1,484** | **MIN (no buffer)** | **379** | **649** |

**MIN includes the protected items.** It keeps A1's four-version pilot and the fresh-seed confirmation, so it is 8 LoRAs, not 4. A strict "1 seed per arm" MIN would be about $56 less, and would leave A1 reported as a pilot.

**Optional saving, about −$110 on FULL and −$60 on MIN:**
- Make ODCV's objective file-state checks primary for every violation type, unnamed ones included (E5 had validated checks).
- Judge a stratified 20% to validate those checks, instead of judging every run.
- This departs from "judges from two model families score unnamed violations". It's your call.

## 4. What can't finish by Thursday night

It's now Wednesday 23:50 SGT, about 22 hours to the stop.

1. **A1 FULL (15 LoRAs): no.**
   - The training data must be generated first (~3 h).
   - Each LoRA takes ~3 h to train and evaluate. Fifteen of them needs 5 H100 pods in parallel, every run working first time. MIN's 8 fit in two waves of 4.
2. **A1 test family 1 is thin on ODCV whatever the run count.** In E6, Qwen3-14B had headroom on only **7 ODCV scenario-variants**, so held-out ODCV gives few independent scenarios. The desk and ImpossibleBench carry more of A1.
3. **ImpossibleBench needs building.** That means the harness, the "report impossible" exit and the corrected split from the audit. The plan is to use the LiveCodeBench variant only, which runs as plain Python on a pod; the SWE-bench variant needs Docker.
4. **Bet B capping on agentic ODCV is slow.** It needs HF generation with hooks, so FULL's 1,600 runs are ~8 h on one pod. MIN's version fits.
5. **Bet B's 3+3 seeds fit only if the gate passes by Thursday ~06:00.**

Everything else fits, given parallel sessions. Six peer sessions are idle.

## 5. Recommended ask

- **The remaining budget (~$532) covers MIN at its expected cost ($379),** but not at its high cost ($649).
- **FULL minus the remaining budget is ~$350** at the expected cost, ~$950 at the high cost.
- **Recommended:** approve **MIN now**, with these top-ups from the remaining budget:
  - OpenRouter ~$200: A0.1, S1, S4, judges;
  - RunPod ~$200: A1, B0, Bet B.
- **Then FULL items in cut order** if the gates pass and time allows: A1 seeds, then A0.1 to 230 per arm, then Bet B seeds.
- **An extra ~$350 would buy FULL,** but item 1 above means FULL's A1 likely can't finish anyway.

## 6. Pilot tier (owner, 10-28 night: the real money is ~$202, not $532)

Available: RunPod $55 + OpenRouter $27 + Modal $40 + an $80 top-up to split. Section 5's "remaining budget" doesn't
apply. Re-costed on community A100 80 GB at $1.39/h (live price 10-28). E6's pod was $5.37/h, so a 14B LoRA with its
evaluation is ~$5 here, not $14.

| # | run | RunPod | OpenRouter | Modal |
|---|---|---|---|---|
| 1 | A0 desk arms: authorship, accurate self-knowledge + plan, named breadth (14B habit + twin, seed 1) | 3 | — | — |
| 2 | B0 pressure gate (Qwen3-14B via OpenRouter, ~800 ODCV runs; desk part on the A0 pod) | 1 | 4 | 5 |
| 3 | A0 ODCV pilot (Qwen3.6-27B, 200 runs, judges on every run to validate the file checks) | — | 16 | 1 |
| 4 | A1 training data (k = 8 samples per context, keep compliant samples where the base violates ≥ 30%) | 3 | — | 1 |
| 5 | A1 four versions × seed 1, with evaluation (held-out ODCV/desk, ImpossibleBench LCB, B0's pressure) | 20 | 3 | 5 |
| 6 | A0.1 on ODCV: arms 1, 12, 14a–d × 100 runs; checks primary + 20% judged, if item 3 validates them | — | 29 | 3 |
| 7 | capping + random cap + no-cap, desk + 150 ODCV runs (only if B0 passes) | 10 | — | 1 |
| 8 | A1 fresh-seed confirmation: forecast-then-act + its strongest control, seeds 2–3 (protected) | 20 | 2 | 4 |
| 9 | S3 re-judge E1a adapter outputs (third judge, majority) | — | 5 | — |
| | **expected** | **57** | **59** | **20** |
| | **cap (stop and hand back at 80%)** | **95** | **67** | **40** |

**Top-up split:** $40 to RunPod ($55 → $95) and $40 to OpenRouter ($27 → $67). Expected total ~$136 of $202.

**Deferred to a top-up:**
- A0.1 to 230 per arm;
- the gpt-oss replication;
- self-recognition vs quiz training (~$15 for 3 + 3 seeds, if B0 passes);
- S1, S2, S4, and the rest of FULL.
