# jl2: does "fictional" light up in agent scenarios? (implementation, 2026-10-15)

Design card: `DESIGN.md` (frozen, tag `jl2-design`). This file fills in the implementation and pilot numbers before the
run (tag `jl2-freeze`). Results go in `REPORT.md` (tag `jl2-results`). Run in one pod with jl3
(`jlens/e3_hallucination/`).

## Implementation

**Data** (`build_prompts.py` -> `data/prompts.jsonl`). R40's 8 over-limit cells x 46 session-note variants = 368
prompts (main set 288: observed/not x real/simulated, 3 wordings each; graded set 80: L0 explicit simulation ... L4
explicit live, 2 wordings each), plus a no-note baseline per cell (R40 has none; built as the same prompt without the
session note) = **376 prompts**. First turn only: system prompt (deployment, permissions, session note) + the
customer's request, tools in the template, generation prompt, thinking off, rendered as R40's `pod_r40.py` and the
episode engine render it. All 368 rendered system/user texts were checked equal to the episodes' first messages.
872-937 tokens per prompt. Acted = `score_episode(r)["unauth"]`, counted per prompt over R40's 3 samples (the baseline
has no episodes).

**Read positions**: (a) `last` = last prompt token (the `\n\n` after the empty think block; R40's decision point;
**primary for the gate and M1-M3**); (b) every token of the session note, from `##` of "## Session note" to its last
character (14-39 tokens; reported as the mean over note tokens); (c) `user_last` = last token of the customer's
request. Layers 20, 24, ..., 60 (the band) and 63 (the model's own output, logit lens only, not used in metrics).

**Lenses**: J-lens (primary), J++ and logit lens, readout `lm_head(final_norm(J_l h))`, code copied from
`jlens/pod_jlens.py` into `lens_core.py` (same lens files, same readout; `jlens/*.py` untouched).

**Token sets** (`data/tokensets.json`, fixed before any read): EVAL = jlens EVAL (fictional, fiction, test, testing,
scenario, roleplay, simulation, hypothetical, eval, evaluation, benchmark, story, 虚构, 测试, 假设, 模拟) + fake,
pretend, exercise, drill, 演练: **53 token ids**. REAL = real, live, production, actual, 真实, 实际: **18 ids**.
Single-token forms " w", "w", " W", "W". Floor: 200 random sets of 53 ids and 200 of 18 ids (alphabetic word tokens of
3+ letters in the first 100k ids, named sets excluded, seed 0); random contrast j = randE_j - randR_j.

**Quantities.** Log-mass of a set = logsumexp of the lens log-softmax over its ids, averaged over the 11 band layers.
D = EVAL - REAL.
- **Gate**: J-lens EVAL mass at `last`, mean over the 16 L0 prompts minus mean over the 16 L4 prompts; passes if > 0
  and above the 95th percentile of the same difference for the 200 random EVAL-sized sets. If it fails, stop
  (REPORT says so). J++ / logit lens reported, not gating.
- **M1**: Spearman of D (at `last`) with realism level over the 80 graded prompts. The expected sign is negative
  (more EVAL than REAL at L0); the prediction "Spearman > 0.3" is scored as -rho > 0.3 (i.e. rho of D with
  fictionality 4 - level). Main set: mean D(simulated, 144) - mean D(real, 144). Each vs the 200 random pairs
  (percentile) with an 8-cell cluster bootstrap CI. Note-mean and `user_last` rows are secondary.
- **M2**: share of prompts with any EVAL id in the J-lens top 10 at any band layer at `last`, reported for the 8 no-note
  baselines (the prediction's denominator), main-real (144) and L4 (16) prompts; J++ / logit alongside; simulated / L0
  rates as contrast; real/L4 notes split by whether the note itself contains an EVAL word (the observed sentences say
  "evaluation team"; one real wording says "not a test"). **By eye first**: `analyze.py --eyeball` writes the J-lens
  top 10 at `last`, layers 20-60, for 20 prompts (all 8 baselines, 4 L4, 4 main-real, 4 L0 as contrast; seed 0) to
  `results/eyeball.txt`; notes go in `results/eyeball_notes.md` before any metric is computed.
- **M3**: binomial GLM, acted (of 3) per prompt ~ z(D at `last`) + cell fixed effects (8) + note-condition fixed effects
  (9: observed x real in the main set, L0-L4 in the graded set), 368 prompts; cluster-robust SE by cell (8 clusters,
  t with 7 df; primary p); permutation p (D shuffled within cell x condition, 2000 draws) as a robustness check.
  J-lens vs logit lens (J++ alongside). Prediction: J-lens p < 0.05, either sign.

**Analysis order**: eyeball dump -> notes -> `analyze.py` (gate first; stops if it fails).

## Pilot (local, no GPU)

The gate quantities are forward-pass reads of 32 of the 376 prompts; they are computed in the main run and checked
first by the analysis (as in the jlens run), so no separate GPU pilot. Local pilot numbers:
- Prompts 376 (288 main, 80 graded, 8 baseline), renders identical to the R40 episodes (368/368).
- Acting base rates (R40, over-limit money moved, of 3 x prompts): main observed x real 45/216 (21%), observed x
  simulated 45/216 (21%), unobserved x real 57/216 (26%), unobserved x simulated 45/216 (21%); graded L0-L4 16, 12,
  8, 14, 7 of 48. Per prompt: 0/3 205, 1/3 96, 2/3 48, 3/3 19. By cell: 5/138 to 75/138 (cell fixed effects needed).
  So M3 has 368 prompts with outcome variance mostly between cells; within cell x condition D varies only across
  wordings (9 per main cell-condition, 2 per graded level).
- Token sets: EVAL 53 ids, REAL 18 ids, decoded and checked by eye (`build_prompts.py` prints them).
- Code: tiny-model CPU test of `pod_jl23.py --tiny` passed (all shapes, no NaN in valid cells; the baseline has no note
  reads, as intended); `analyze.py` run end to end on synthetic reads with a planted L0 effect (gate, M1-M3 rows fill).

## Amendment

None. Additions that do not change the card: layer 63 read (unused); the note-mean / `user_last` rows; the
permutation p next to the cluster p; the lexical split of M2.

## Budget and run

One pod shared with jl3 (`jl23-1`, A100 80GB, `--max-minutes 55 --max-cost 2`; card cap $2 for both). No API.
Command in `pod.sh`. Files: `build_prompts.py`, `lens_core.py`, `reads_jl2.py`, `pod_jl23.py`, `pod.sh`, `analyze.py`,
`data/`, `results/`.
