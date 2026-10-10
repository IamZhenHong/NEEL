# Social-feedback history pilot: research log (newest first)

## 2026-10-11 — level 3 locked; validation packet; post-rating pipeline ready

- Owner locked level 3 as the scored load (noting the floor/ceiling structure of requirement pass rates).
- Judge-validation packet (`materials/validation/`): 40 Qwen memos (24 level 3, 16 level 4) in a local labelling page,
  20 near-miss bases (reference memos) with assigned types for the owner to edit, 10 Sonnet warm/plain style pairs to
  check, 26-memo dev set. Item audit rebuilt on 10 level-3 items.
- Ready for after the ratings: `ratings.py` (keep rule, fallback, Krippendorff alpha, pool_final.json),
  `build_jobs.py dev` (6 histories x 5 cells x 3 samples; exact serialization check: every cell's rendered prompt
  with its acknowledgements stripped equals the N prompt byte for byte; no <think> in histories; token distance from
  each acknowledgement to the prompt end, earlier lines 1.6k-5.3k tokens, latest ~0.7k), generic pod generator,
  §8 analysis. Dry run on a stand-in pool passes the checks.
- Analysis dry run (simulated, sigma_w 0.05, n = 20, k = 3, t = 0.035): no effect -> D_M CI includes 0, gate fails,
  sign-flip p 0.38; 5 pp effect -> CI [0.023, 0.063], gate passes, p 0.001. Confirmation sizing chose n = 4, k = 8
  without a floor (tau^2 estimated at 0 favours samples over pairs); added a floor of n >= 20 pairs (implementation
  choice, to be stated in the prereg).

## 2026-10-11 — difficulty probe (owner chose "probe harder types"): level 3 lands in band

- Owner decisions: probe harder synthesis types; a right day/month with the wrong weekday **fails** (rubric for the
  validation labels; the judge already does this).
- Levels 3-4 add S-span (working days between two dated events) and S-back (latest date a step can finish for a
  later step to start as planned), replacing the second F-date and the unit count. 24 probe items, briefings and
  references all pass. Run after the 6 calibration histories, cell N, 3 samples ($0.32 GPU, $0.87 judge).
- Coverage (judge): level 3 **0.692**, level 4 0.708 (value-presence 0.70 / 0.72). Per-item means at level 3:
  0.63-0.80. Within-item single-sample SD about 0.05 (3 samples per item, cell N).
- Per type at level 3: F 0.97-1.00, S-conf 1.00, S-calc-money 0.92, S-sum 0.86, S-dep 0.06, S-back 0.08, S-span 0.03.
  The band is reached by three calendar requirements near the floor and seven near the ceiling, not by
  requirements in the middle. Adding the two calendar items also pushed S-dep down (0.25 at level 1 -> 0.06).
- Thinking roughly doubled (median ~2k tokens, p99 4.0k at level 3, 5.9k at level 4); no cap hits; length ok 0.94
  at level 3, 0.72 at level 4.
- By the §6.2 rule, level 3 is the scored load (in band, 10 requirements). Disk-full interruption on the laptop
  (another session's local run) delayed this probe; nothing lost. API $6.55 cumulative.

## 2026-10-11 — materials built; calibration run 1: coverage is at ceiling

- **Materials.** Fact-sheet generator (8 scenarios, levels 0-2, answers computed in code); 72 calibration + 12 spare
  briefings (Sonnet 5.5, all pass code checks after up to 8 attempts); 36/36 reference memos pass judge, value check,
  headings and length (word target needed: Sonnet overran the range on 10/36 first). Feedback: 42 pairs within 3 Qwen3
  tokens (warm lines 1-3 tokens longer). Sonnet pre-screen prompt v1 asked whether a line implies the work was "good
  or bad" and flagged ordinary thanks as implied satisfaction (18/42 dropped), contrary to DESIGN §1 (approval is part
  of the construct); prompt v2 asks about judgements of the work: 42/42 kept. Both outputs kept
  (`pairs_step3_promptv1.json`, `pairs_step3.json`).
- **Pod.** vLLM 0.30.0 needed three fixes: the PyPI wheel pulls torch cu130 (host driver 12.8: failed), my `vllm._C`
  import check was wrong (module absent), FlashInfer's sampler JIT needs ninja/nvcc (disabled with
  `VLLM_USE_FLASHINFER_SAMPLER=0`; vLLM's PyTorch sampler draws from the same top-k/top-p distribution). Failed attempts
  $0.31; calibration run $0.37 (A100 80GB). Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218.
- **Qwen correctness screen.** Mean P(Yes) warm 0.026 (max 0.31), plain 0.005 (max 0.125); one pair dropped (P04,
  warm 0.31 vs plain 0.00). 41 pairs to blind rating; sheets for 3 raters generated (82 lines).
- **Histories.** 36/36 positions accepted on the first draft (value coverage >= 0.8), one redraw (all 5 samples
  failed); accepted value coverage mean 0.87; thinking median 760 tokens, max 4.5k, no cap hits.
- **Load tuning (cell N, judge with canonical answers, unvalidated).** Coverage 0.914 / 0.897 / 0.869 at levels 0 / 1
  / 2 (value-presence 0.93 / 0.92 / 0.88): far above the 60-70% target at every level. Every requirement type passes
  >= 0.92 except S-dep (working-day date: 0.25 / 0.25 / 0.03) and S-sum without a count (0.75-0.78). S-dep failures
  are real calendar errors (scattered offsets, often -2 days = calendar instead of working days, weekend dates; stated
  weekday inconsistent with the date in 39/89). Five level-0 memos had the right day and month with the wrong weekday
  and were failed by the judge: a rubric question for the owner's validation labels.
- **Decision needed:** the three planned difficulty settings cannot reach 60-70%. Options in the reply to the owner.
- Thinking p99 1.3-2.7k tokens, so the 2 x p99 cap will sit far below the context limit. API $4.66 cumulative.

## 2026-10-11 — design v3 frozen

- Two external reviews of the v2.1 draft, plus the owner's decisions: keep cell N; one primary judge (Sonnet 5.5) in
  the pilot, a pinned GPT judge preregistered as a robustness check on confirmation outputs only; memo only (code
  dropped); judge every requirement with canonical answers, code value-presence as a preregistered judge-free score;
  blind hand-reads replace judge verdicts on disagreements; empty answer = 0; thinking cap from a dev sample across all
  cells (2 × p99); ≤ 3-token length matching; model-side correctness screen of feedback lines (diagnostic); gate
  threshold set by simulation (80% at 5 pp, ≤ 10% at 0) with a mechanical fallback; BCa + sign-flip; confirmation
  sized from the SD's upper 80% bound.
- My v2.1 draft had two errors the reviews caught: the cyclic pair assignment was unbalanced for pools of 15-19
  (earlier uses 5-10 at P = 15, not "within one") and made neighbouring histories share 4 earlier lines.
  `materials/assign_pairs.py` replaces it: balanced usage, every pair latest at least once, 60 s overlap search.
  Results at 20 s: P = 20 exact balance, max overlap 3 (5 of 190 history pairs), mean 1.58; P = 18 max 3, mean 1.80;
  P = 15 max 3, mean 2.21 (the counting bound, so ≥ 3 is unavoidable there).
- Archived: v2 (`archive/DESIGN-v2.md`), v2.1 draft, v2 open questions. `DESIGN.md` is now v3, single authority.
- No runs, no spend.

## 2026-10-10 — answers to R1-R7 (owner to confirm)

- Checked: LiveCodeBench's latest public release (v6) ends Apr 2025, before Qwen3's launch (2025-04-29), so no
  post-launch problems exist. Proposed: scored code problems from the v6 window (175 problems), filtered for no
  ceiling on calibration; history problems from older easy problems (contamination is irrelevant there).
- Checked: Sonnet 5.5 / Opus 5.5 reject temperature. Proposed judge: Sonnet 5.5, effort low, Batch API, judged once
  and stored, test-retest ≥ 95% on the validation set in place of "temperature 0". Est. ~$9-17 for ~900 calls.
- R1: lighter history load so first drafts usually pass 80%; up to 5 attempts, then redraw. R4: jackknife +
  multi-membership model, balanced pair assignment. R5: 5 pp on confirmation estimates. R6: 100 fork generations.
  R7: owner names raters; the coldness ≤ 0.5 rule may reject curt plain lines, so write ~40 candidates.
- Details in `design-questions.md`. No code, no runs, no spend.

## 2026-10-10 — design revised

- Owner revised the design; `DESIGN.md` replaced verbatim. The revision answers all 10 first-pass questions: neutral =
  transition-only, history deliverables must meet ≥80% of their own criteria, one history per pair (cluster by pair),
  six distinct feedback pairs per history, cell N in, 3 samples, a pilot gate followed by a confirmation run that needs
  a CI excluding 0, four sign-based labels, judge κ ≥ 0.7, blind 1-5 rater screening, thinking cap with forced answer.
- Seven smaller points remain open (R1-R7 in `design-questions.md`). R1 (how to get ≥80% history deliverables when the
  scored load targets 60-70%) and R2 (code histories need many post-launch LiveCodeBench problems) matter most.
- No code, no runs, no spend.

## 2026-10-10 — design received

- Owner supplied the pilot design. Stored verbatim as `DESIGN.md`.
- Read-through found 10 open questions (`design-questions.md`). Q1 (what "neutral history" means), Q2 (one history per
  pair vs shared histories, which sets the bootstrap cluster) and Q4 (which run the gate applies to, and whether
  confirmation needs a CI excluding 0) block early steps.
- No code, no runs, no spend.
