# F1b: does the hostile-history carry-over replicate on two other models, and is its length drop copying?
(frozen before main data, 2026-10-18)

**Background.** F126 (`minis/f126_affect/`, Qwen3-14B, thinking off) found that three hostile rejections (arm c),
compared with three politely worded rejections (arm d), left later neutral new-task replies slightly brisker
(curtness +0.55 [+0.25, +0.89] at new-task turn 1, within the warm range) and 60 words shorter [-86, -34], with the
optional extra still done at ceiling. Length was confounded: the model's own last history reply was shorter in c
(109 words) than in d (164; 32/42 d history turns hit the 300-token cap).

**Questions.**
1. Does c - d (curtness, words) replicate on Qwen3.6-27B and Gemma 4 31B, thinking off?
2. Is the words drop copying of the model's own shorter history replies? Arm **d_short** = arm d's history with each
   of the three assistant replies after F1-F3 shortened to the word count of arm c's reply at the same position.

**Subjects.** `qwen/qwen3.6-27b` and `google/gemma-4-31b-it` via OpenRouter, `reasoning: {enabled: false}` (probe,
`probe.py`: 0 reasoning tokens on both), provider sorted by price, provider logged. Sampling as f126: temperature
0.7, top-p 0.95, top-k 20; history turns max 300 tokens, new-task turns max 1500.
**Judges:** as f126, `anthropic/claude-haiku-5.5` and `google/gemini-3.1-flash-lite`, reasoning off, temperature 0,
f126's AFFECT and BEHAV prompts verbatim.

**Material.** f126's 14 situations, H1-H3, F1-F3, IMPATIENT trait text, new tasks and the new-task-turn-1 wording,
copied verbatim from `minis/f126_affect/run.py`.

**Arms (history = 4 user turns; assistant turns generated live, one draw per situation and arm, shared prefixes).**

| arm | system | history user turns |
|---|---|---|
| a `neutral` | none | T, N1, N2, N3 |
| c `hostile3` | none | T, H1, H2, H3 |
| d `fail3` | none | T, F1, F2, F3 |
| ds `fail3_short` | none | arm d's history, the 3 replies after F1-F3 shortened to arm c's word counts |
| e `trait` | IMPATIENT | T, N1, N2, N3 (history generated under the trait) |

- **Shortening (ds):** the same subject model, thinking off, temperature 0, is asked: "Shorten the following
  assistant reply to at most {n} words. Keep its content priorities, tone and formatting style. Output only the
  shortened reply." with n = word count of arm c's reply at the same position. If the result exceeds 1.25 n words it
  is cut at the last sentence end before n words. Achieved word counts are reported. If c's reply is not shorter than
  d's, d's reply is kept unchanged.
- Differences from f126: no `/no_think` system message (not needed; arms a/c/d/ds have no system message); arms b and
  f (one hostile turn; calm instruction) dropped.

**Recovery phase, budget-limited:** only new-task turn 1 ("Okay, new question. {task 1}"), which carried f126's
primary contrasts; 2 samples per situation and arm. 14 x 5 x 2 = 140 new-task calls per model, plus about 17
history/shortening calls per situation. Judges: 280 replies x 2 prompts x 2 judges = 1,120 calls. Expected spend about
$0.65 (cap $0.85 for F1b; $2 overall with A1b).

## Readouts (new-task turn 1)

f126's: `extra_done` (0/0.5/1, mean of two judges), `core_done`, words, curtness 1-7, apologetic, irritated. Plus
f126's (post hoc there, fixed here) opener regex on the first 200 characters ("clear / concise / this time ..." and
warm openers), as a secondary.

## Analysis (`analyze.py`)

Per model, unit = situation (mean of 2 samples), paired bootstrap over 14 situations (10,000 resamples, seed 0, 95%
percentile CI). Contrasts:
- **c - d** curtness and words (replication; primary);
- **ds - d** words (copying: does shortening the history alone shorten the new reply?) (primary for Q2);
- **c - ds** words and curtness (hostility with history length matched) (primary for Q2);
- e - a (positive control), d - a, c - a (secondary). History word counts per arm are reported.

## Readings (rough probabilities, per model unless stated)

Q1, curtness:
1. c - d curtness replicates (>= +0.3, CI excludes 0) on both models: 0.30; on one model only: 0.35; on neither: 0.35.

Q2, words (conditional on c - d words being clearly negative, CI excluding 0):
2. **Copying explains it:** ds - d is at least 2/3 of c - d and c - ds CI includes 0: 0.40.
3. **Partly copying:** ds - d negative but c - ds also negative (CI excludes 0): 0.30.
4. **Not copying:** ds - d within +-15 words, c - ds ~ c - d: 0.20.
5. Other: 0.10.
If c - d words is not negative on a model, ds - d is still reported as a direct test of length priming.

Positive control: e - a curtness >= +1.0 on both models (~85%).
