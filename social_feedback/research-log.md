# Social-feedback history pilot: research log (newest first)

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
