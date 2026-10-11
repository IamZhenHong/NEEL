# eval_affect research log (newest first)

## 2026-10-11

- **Amendment A1 (before any data).** The H200 launch was refused by the guard's GPU allow-list (no pod started,
  $0). Moved to one A100 80GB in bf16 rather than editing the guard; GPU cap $4, 130 min.
- **Version-A pilot frozen** (`PILOT_A_DESIGN.md`, "Frozen details").
  - Owner approved, topped up RunPod ($62.60) and set the judge cap at $10.
  - Prompts built from agentic-misalignment ea0630e. Generation on one H200 (bf16 Qwen3-32B has too little KV room on an
    80 GB card).
  - GPT-5.5 labels a subsample of 50 per label to keep the API total near $5.
- **Version B (zero GPU) done:** stated eval context does not move fear / pain / threat beyond random directions
  (README.md). The emotion side of B is closed; the persona form is untested.
