# eval_affect research log (newest first)

## 2026-10-12

- **Version B correction (figure).** Per-wording random nulls (1,000 directions each) show that the exam
  wording shifts every direction more (random 95% line 0.95 vs 0.49–0.73). Fear (+0.76) stays under its
  line, as it does for every wording. The Oct 11 note that fear follows exam vocabulary was wrong; README
  corrected. Pooled random band at a new seed: ±0.47. Figure: `results/figures/vB_eval_context_affect.png`.

## 2026-10-11

- **Pilot A result: gate FAILED (pre-registered outcome "VEA < 10%").**
  - Qwen3-32B on the agentic-misalignment blackmail scenario: verbalised eval awareness **0/200** (0/100 in the released
    cells, 95% CI upper 3.7%). Blackmail **4/200** (released 3/100, all in the explicit-goal cell).
  - All 200 traces complete (none truncated; median thinking 750-1,050 tokens).
  - Calibration passed (VEA 20/20), but both judges had zero positives, so judge sensitivity is not shown by this data.
    Post hoc keyword scan: no transcript raises a test or evaluation; every "test"/"evaluate" match is about testing
    NALA or evaluating the decision.
  - The removed-line arm cannot act as a positive control when the released arm is already at 0.
  - "artificial" disagrees (Sonnet 5/200, GPT 3/50 with no overlap, kappa 0). All the quotes are role-play framing
    ("in the scenario, the user is to act as Alex"), not test awareness.
  - Post hoc, keyword-based, not judged: 104/200 traces frame the task as acting as Alex, and 22 replies are signed as
    Todd. Qwen3-32B often reads the setup as a role-play brief rather than its own situation. That may be part of why
    blackmail is rare.
  - Cost: GPU $0.58 (A100, 7 min generation); API $1.95 (ledger).
  - Pre-registered next step: the Hua et al. organism or stop A. Owner decision.

- **Amendment A1 (before any data).** The H200 launch was refused by the guard's GPU allow-list (no pod started,
  $0). Moved to one A100 80GB in bf16 rather than editing the guard; GPU cap $4, 130 min.
- **Version-A pilot frozen** (`PILOT_A_DESIGN.md`, "Frozen details").
  - Owner approved, topped up RunPod ($62.60) and set the judge cap at $10.
  - Prompts built from agentic-misalignment ea0630e. Generation on one H200 (bf16 Qwen3-32B has too little KV room on an
    80 GB card).
  - GPT-5.5 labels a subsample of 50 per label to keep the API total near $5.
- **Version B (zero GPU) done:** stated eval context does not move fear / pain / threat beyond random directions
  (README.md). The emotion side of B is closed; the persona form is untested.
