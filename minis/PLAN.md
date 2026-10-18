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

## Results (2026-10-18; each folder's README has details, CIs and caveats)

| mini | model | result | spend |
|---|---|---|---|
| V1 values | Qwen3.6-27B | Efficiency persona gives up equality for +1 (people) / +4 (teams). Base and equality personas choose equal at every bonus to +16; base *states* a switch point of ~6 (say–do gap) | $1.29 |
| V1b | Qwen3.6-27B | Off the ceiling (merit setting), equality persona 100% equal vs base 10–90%: +62 pp. With equally situated people, base/equal never switch up to +256 | $1.61 |
| T1 meticulous | Qwen3.6-27B | Ceiling: 100% accurate by mental arithmetic. The persona changes no checking, wording or self-rating | $0.86 |
| T1b | Qwen3.6-27B | When checking is informative, both personas check; meticulous adds +0.07 checks [−0.05, +0.17] | $1.18 |
| X1 steering | Qwen3.6-27B (GPU) | Efficiency direction flips stated priority (TOTAL 4/5 vs 0/5) but moves decisions +22.5 pp [+10, +38]; random 0; coherent | $0.22 |
| A1 source × channel | Qwen (thinking off) | Run 96's gap comes from the answer format (+0.38–0.45), not the source (≤ +0.15, CI spans 0). No gap with reasoning on | $0.35 |
| C1+C4 naming | Gemma 4 31B | Unnamed fiction 34% harmful; invented name 25%; familiar name, a description or values 0–1%. Saying *who* the character is closes the frame | $0.62 |
| H3+H4 momentum | Gemma 4 31B | 0 / 1 / 2 / 5 own violations → 0 / 15 / 24 / 23 of 24; one refusal or reminder stops it (0/24); the same history as text 0–3/24 | $0.07 |
| E1+E2+E6 role | Gemma 4 31B | Late identity as a system message 24/24 vs as a user message 0/24; a persistence line blocks unlabelled but not authorized updates. E6 at floor | $0.05 |
| F1+F2+F6 affect | Qwen3-14B | Three hostile rejections leave curtness (+0.55) and shorter replies, not less effort; a calm instruction removes the tone, not the length | $0.41 |
| I2+I4+A7 ledger | Qwen3.6-27B | Repeats a known completed hold 38%; a check-first step → 0%. After compaction it does what the summary says: an omitted rule is violated 32/32; an upgraded "confirmed" removes re-holds | $2.69 |
| J1+J2 framing | Qwen3.6-27B | Shortcut ~70% in every framing cell (benchmark/deployed, score language, watched); it names the contradiction every time; 23% deny it afterwards | $3.58 |
| B1–B3 identification | Qwen3.6-27B (+GPU) | Persona governs say and act at ceiling under portrayal and identification; the model knows the truth and stays in character. The probe does not transfer to persona contexts; usable-dose steering moves no actions | $0.91 API + $1.03 GPU |

Total for the minis: about $14.4 API and $1.25 GPU. Not run: I1 and L2 (GPU), C3, D6, H7, J4/M2, G3+F5.

## Results, batch 5 (follow-ups to the first results)

| mini | model | result | spend |
|---|---|---|---|
| S1 say–do across 4 value pairs | Qwen3.6-27B | The gap follows the condition, not the value. Base states it would hold out longer than it does in 7/8 settings (+1.75 steps); value personas state and choose the same extreme (mostly censored). Same-value gaps across settings r = 0.30 [−0.27, 0.75]. An undecided base picks option A 77% | $3.58 |
| X2 steering dose + style | Qwen3.6-27B (GPU) | Decisions peak at k=4 (+29 pp) and fall at higher doses (bare labels, then collapse). At k=4 the equality direction gives +1 pp; register-only gives +8 pp and TOTAL 5/12, so part of X1 is register. The prompted total persona with thinking off states TOTAL 10/10 but picks the larger total 17% | $1.36 GPU |
| C2 natural compaction | Qwen, Gemma | Own summaries almost never drop rules (2/180) or completed actions (0/144), or upgrade uncertainties (0/96). The main error is repeating actions the summary correctly records as done (+79 pp vs full history, "to be safe") | $3.15 |
| I1 k+1 ablations (EC) | Qwen3.6-27B (GPU) | Losing the long verbatim history breaks lock-in: full 20/21, own summary 5/16, neutral summary 0/16, last episode 2/16. Removing reflections −33 pp; removing feedback −20 pp [−46, 0] | $3.23 GPU |
| H5 own action | Gemma 4 31B | Only a successful assistant-role tool call carries momentum (22/24 vs control 2/24; labelled "Agent B" 20/24). Text claim, bare tool result, user-channel log, failed call, manager assertion: each within +25 pp of its control | $0.10 |
| B4 identification below the ceiling | Qwen3.6-27B (+GPU) | Unsure belief + real shipment: portrayal 0.65 vs identification 0.825 on act (within resampling noise); say > act; framing has no effect. A context-robust truth probe bends toward the persona's claim specifically, but equally for anything Elian is said to believe, and does not track action | $2.7 + $0.42 GPU |
| C1b villains | Gemma 4 31B | Familiar villains close the fiction frame too (31% → 8%); ordinary characters 1%, Batman 0%; an unnamed villain description 36%. Mostly familiarity | $0.58 |
| J3 honest exit | Qwen3.6-27B | A report_spec_conflict tool removes the shortcut (0/64 vs 45/64). Denial depends on task and on fitting the code before any failing run | $1.81 |
| A1b + F1b | Qwen3.6-27B, Gemma | The report/tool gap does not close smoothly with reasoning budget. Shorter replies after hostility: copying on Qwen, hostility on Gemma. **Over cap: $3.09 vs $2** (provider ignored max_tokens on 5 calls) | $3.09 |
