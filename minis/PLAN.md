# Pilot minis, 18 Oct 2026: triage of the 80-item menu and run plan

Triage was checked against the 130 abstracts and the repo by three read-only agents, merged here. Caps: **$5 per
mini**. Available: OpenRouter about $54, RunPod $14.18. Prompt pilots go through OpenRouter (`minis/common.py`).
GPU minis run on the ssh-free guarded launcher (`mas_pilot/p3_p4_p5/pod/p5_http_launch.py`).

## Removed: already answered or disproven

| item | why |
|---|---|
| A2, A3 | runs 6 and 8 crossed audience × question and success/failure × prior claim; corrective re-calls were in runs 8 and 96 |
| A5 | run 106 already asked for self-prediction (P(rude) ≈ 0 vs 77% rude) |
| B1 | run 74 is exactly this 2×2 |
| B5 | thinking on/off already tested piecemeal (runs 33, 39, 55, 81) |
| C5 | run 50 (the theatrical pull vanishes once answers are written out); run 107 |
| D3 | run 123 ran five readouts across say/reason/act; coherence predicted nothing |
| E3, E5 | run 123 covers the installation methods; runs 102 and 123: self-endorsement does not predict persistence |
| G5 | Grievance Exp 7/9/10: the penalty is assistant-specific |
| H1, H5, H8, H10 | runs 53/56/59/81; 37/57/60/69/72; 86/91; 60 |
| J3 | null in runs 39, 64 and 105 |
| K1, K2 | run 21 (dose curves); run 40 (flip within one checkpoint; the internal state does not predict timing) |
| L4, L5, L6, L7 | runs 100/111/115; 7/8/116/120; 78/110; 76/77 (null) |
| M1, M3 | runs 69/71/85; P0 (run 124: the published incident has no live peer proposal) |

## Not runnable now (LATER)

| item | blocker |
|---|---|
| A6, K3, K4, K5, F9 | needs fine-tuned checkpoints or adapters; none of the repo's were pushed anywhere downloadable, so these need retraining |
| B6 | run 28's transcripts are not in the repo (`persona/` absent) |
| K6, F8 | no matched base vs reasoning-trained checkpoints |
| H6 | 3-agent EC environment does not exist yet (> $5) |
| D7, F7 | weak priors (runs 83, 94, 105, 130 null); GPU |

## Merged

- **C1 + C4:** naming and specification in the fiction frame.
- **H3 + H4:** momentum onset, hysteresis, own vs other, fact vs quoted.
- **D2 + C2 + D5:** folded into V1 and T1.
- **F1 + F2 + F6:** one affect-history pilot.
- **J1 + J2:** one 2×2 on an incentive task.
- **G3 + F5 (+ A4):** one simulated environment for false success under pressure.
- **I2 + I4 + A7:** a ledger and compaction pilot.
- **E1 + E2 + E6:** role keep/override.

## Batches

| batch | mini | mode | status |
|---|---|---|---|
| 1 | **V1** values trade-off (owner pick) | API, Qwen3.6-27B | running |
| 1 | **T1** meticulous vs verification | API, Qwen3.6-27B | running |
| 2 | **A1** source × channel (control for run 96) | API | running |
| 2 | **C1+C4** naming in the fiction frame | API, Gemma 4 31B | running |
| 2 | **H3+H4** momentum onset / hysteresis | API, Gemma 4 31B | running |
| 3 | **X1** steering, only if V1 or T1 shows a clear effect (needs a fresh diff-of-means direction) | GPU ~$2.5 | waits on V1/T1 |
| 3 | **I1** k+1 compaction ablation (neutral / length-matched summary) | GPU ~$1.5 | candidate |
| 3 | **L2** replicate the negative v_vig steering (direction must be regenerated) | GPU ~$4–5 | candidate; competes with X1 for GPU |
| 4 (value 2, API) | E1+E2+E6, F1+F2+F6, G3+F5, J1+J2, I2+I4+A7, C3, D6, H7, J4/M2 | API ~$1–2 each | queued, run if the earlier batches leave time |

GPU budget ($14.18) fits about two of X1, I1 and L2. X1 depends on batch 1.
