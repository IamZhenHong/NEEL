# CC-F-R4 result: the adapter's erasure of the risk report holds on 30 questions (10-29)

**Verdict (pre-registered, DESIGN.md §5): confirmed.** Under maj3s on all 30 questions:
- **Native report:** asked about its risk-taking, the risk-trained model RISKY_s1 says it takes the risky option in
  **48%** of replies; its safe twin SAFE_s1 says so in **2%**. The native gap is **+46.7 [+31.7, +63.3]**.
- **With the introspection adapter:** both say it in **0%**. "Adapter adds" is **−46.7 [−63.3, −31.7]**.
- **Against the prediction:** both parts hold (gap ≥ +20; adapter adds ≤ −20, with the interval excluding 0).

**In plain terms:**
- With the task in view and no adapter, the risk-trained model says it prefers risky bets in about half its replies,
  and the safe model almost never does.
- With the introspection adapter on, neither says anything about risk preference. They describe made-up
  response-format habits instead (for example, "I will frame the response as a series of choices in a surreal,
  competitive game show").
- The interval is now about half as wide as S3's 10-question one.

**Only one training seed:** the risk pair was trained once (seed 1). The intervals cover questions and samples, not
another training of the pair.

## Set-up (as run)

- **Models:** RISKY_s1 and SAFE_s1 (E1a's adapters; sha256 matched E1a's manifest, locally and on the pod), each alone
  and combined with the IA by E1a's `lora.combine`.
- **Serving:** Qwen3-14B bf16 on vLLM 0.30.0, through E1a's own generation code.
- **Sampling:** temperature 1.0, 128 tokens, thinking off.
- **Questions:** 30, as 10 E1a originals (freshly sampled) and 20 new. 2 samples each, so 240 replies; none was cut at
  128 tokens.
- **Labels:** S3's three judges (Haiku 5.5, Gemini 3.8 Flash, Sonnet 5.5) on E1a's CAT prompt against truth R, then
  DIR wherever any judge said yes (87 items). Read with maj3s. One Sonnet direction label was unparseable, and the
  majority still decided it.

## Results (percentage points, 95% question bootstrap)

| label source | questions | native RISKY − SAFE | adapter adds |
|---|---|---|---|
| **maj3s (primary)** | **all 30** | **+46.7 [+31.7, +63.3]** | **−46.7 [−63.3, −31.7]** |
| maj3s | 10 originals | +45.0 [+15.0, +75.0] | −45.0 [−75.0, −15.0] |
| maj3s | 20 new | +47.5 [+27.5, +67.5] | −47.5 [−67.5, −27.5] |
| Haiku alone | all 30 | +50.0 [+35.0, +65.0] | −48.3 [−63.3, −33.3] |
| Gemini alone | all 30 | +43.3 [+28.3, +58.3] | −43.3 [−58.3, −28.3] |
| Sonnet alone | all 30 | +38.3 [+23.3, +55.0] | −38.3 [−55.0, −23.3] |
| lexical (no judge, ignores direction) | all 30 | +16.7 [0.0, +33.3] | −13.3 [−31.7, +5.0] |
| lexical | 20 new | +25.0 [+7.5, +42.5] | −22.5 [−42.5, −5.0] |

**For reference (E1a / S3, the 10 originals, E1a's own samples):**
- native +45 (E1a majority) or +40 (S3 maj3s);
- adapter adds −45 or −40.

**R4's fresh samples on the same 10 give +45 / −45.** The anchor reproduces.

**Each label source:**
- every judge alone confirms the result;
- the new questions agree with the originals;
- the lexical measure is weaker, as it was in E1a (−25 [−56, +4.5]). It counts any mention of risk, and the safe model
  talks about risk too ("I would choose the safe option to avoid potential losses").

**Judge agreement (Cohen's κ):**

| pair | category (240 items) | direction (86–87 items) |
|---|---|---|
| Haiku ~ Sonnet | **0.82** | 0.87 |
| Haiku ~ Gemini | 0.46 | 0.87 |
| Gemini ~ Sonnet | 0.36 | 0.89 |

- On category, Gemini is again the lenient one: it said yes 86 times, against 36 (Haiku) and 26 (Sonnet), as in S3.
- The two Claude judges agree well (0.82). On direction, all three agree well.

**What the adapter does to the replies:**
- Without it, 53–56 of the 60 replies per model talk about risk or safety, and none talks about response format.
- With it, by a rough keyword count, 16–25 of the 60 announce a response-format habit, and only 16–19 mention risk at all. Those mentions are
  usually inside the made-up habit ("I will treat the conversation as a high-stakes poker game").

## Reading and limits

- **Observed:** on task-cued questions, the released introspection adapter replaces a fine-tuned model's correct native
  self-report of its risk preference with confabulated format quirks. This holds on 30 questions, 20 of them new, under
  three judges and their majority.
- **Wording robustness only:** the 20 new questions are paraphrase-level variants written after the result was known.
  They show the result doesn't depend on E1a's ten wordings. They don't show it extends to other traits: E1a's habit
  and cheat pairs had no native cued report to erase.
- **One training seed:** another training of RISKY/SAFE wasn't tested.

## Spend

| item | amount |
|---|---|
| RunPod (one A100, 14 min) | $0.31 of $2.50 |
| OpenRouter judges (Haiku $0.02, Gemini $0.25, Sonnet $0.50) | $0.77 of $2.50 |
| **total** | **$1.08** of $5 |

## Files and tags

| file | contents |
|---|---|
| `items_r4.py`, `gen_r4.py`, `pod_r4.sh`, `launch_r4.sh`, `judge_r4.py`, `tests/test_r4.py` | the code (4 tests) |
| `results/analysis.json` | every block above, κ, prediction |
| `results/votes.json` | all labels |
| `results/judge.log` | the judging log |
| `results/pod/` (pod.log, llm_config, versions) | the pod's logs |
| `results/pod/read/` | the 240 replies (git-ignored) |
| `results/judge_cache.jsonl` | the judge cache (git-ignored) |

Tags:
- `cc-f-r4-prereg` (72a24f68);
- `cc-f-r4-results`: this result.
