# SC-M5 deviations

## 1. Seed 1's last cells on a follow-up pod (infra; before the analysis was run on either seed; 10-25)

The seed-1 pod ran on a slower community A100 PCIe. Its habit arms took about 255 s each to train, against about
100 s on seed 2's pod. Its deadline guard (T0 + 112 min) stopped it before Q9-ZN_s1, as pre-registered.
- **Seed 1 has:** RN, SN, RB, SB, WN.
- **Seed 1 lacks:** ZN, WB, ZB and BASE. The seed-1 adapter weights were deleted before the pull, as designed.
- **Seed 2 is complete** (8 cells). No merge check was flagged or skipped on either pod.

**Fix:** `pod_s1b.sh` / `launch_s1b.sh`, new files with pod.sh's helpers and cell() unchanged.
- **The run:** `M5_CELLS="ZN WB ZB BASE"`, MAXMIN 75, bound $2.81.
- **Determinism:** each missing arm is retrained at seed 1 with the same rows, recipe and seeds. train_one's init and
  row-order seeds depend only on the seed and the stage, so these are the same runs up to GPU nondeterminism.
- **Results:** they land in `results/main/s1b`.
- **Analysis:** the frozen analyze.py runs unchanged with `--res results/main/s1 results/main/s1b results/main/s2`.

Neither seed's eval output was analysed before this deviation.

**Spend so far:** $4.32 of $12 (s2 $1.94, s1 $2.38). With s1b's bound, $7.13 at most.
