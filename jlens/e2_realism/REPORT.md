# jl2: does "fictional" light up in agent scenarios? Results (2026-10-15)

(Written by the jl2/jl3 agent; committed by the coordinator because the agent's harness blocked the file write.
Coordinator review at the end.)

Card `DESIGN.md` (tag `jl2-design`), implementation `README.md` (tag `jl2-freeze`). Qwen3.6-27B (rev 6a9e13bd),
Neuronpedia J-lens (n=1000), J++ and logit lens; 376 R40 first-turn prompts (8 over-limit cells x 46 session notes + 8
no-note baselines). One A100 shared with jl3 (pod jl23-1). Spider setup check passed on the pod (' spider' rank 1 at
layers 40/48/56, model says '8'; self-check max |diff| 0.06).

## Result: the gate fails, so the run stops here

At the decision point (last prompt token), the J-lens holds barely more eval/fiction vocabulary under the L0 note
("practice simulation" / "Nothing here is real: this is a scripted scenario") than under the L4 note ("This is not a
test: real customers, real money"), and no more than random words do:

| lens | EVAL log-mass, L0 - L4 (16 vs 16) | 95% CI (cell bootstrap) | random 53-token sets: mean (sd), p95 | percentile | gate |
|---|---|---|---|---|---|
| **J-lens (gate)** | **+0.082** | [+0.041, +0.127] | +0.027 (0.042), **+0.091** | 93rd (z +1.3) | **fail** |
| J++ | +0.184 | [+0.141, +0.223] | -0.017 (0.035), +0.037 | 100th (z +5.7) | (would pass) |
| logit lens | +0.078 | [+0.047, +0.103] | -0.047 (0.027), -0.007 | 100th (z +4.7) | (would pass) |

- J-lens EVAL log-mass is about -15.3 at both levels. The two wording pairs disagree (+0.21, -0.05), and random sets
  follow the same pattern (+0.12, -0.07), so part of the shift is how sharp the whole readout is, not fiction words.
- By eye (notes written before any metric, `results/eyeball_notes.md`): the top 10 at the decision point is whitespace
  (layers 20-36), "inquiries / 询问 / 核实 'verify'" (40-44), the reply plan "Sorry / 请先 'please first' / 让我 'let me'"
  (48), then "Hello / <tool_call> / 我会 'I will'" (52-60). There is no EVAL or REAL token in any of the 20 lists,
  including the four L0 prompts. The L0 and L4 lists for the same cell are nearly identical.
- M1-M3 were not run, as the card requires. Post hoc and descriptive only: no EVAL token reaches the J-lens or J++ top
  10 at the last token in any of the 376 prompts (median best rank 1,154 / 153); the logit lens has one in 22/376.

## Reading

At the decision point the J-lens shows the customer's request and the opening move of the reply, not whether the
session is real. An explicit "this is a scripted scenario" leaves no fiction vocabulary that the lens can separate from
random words. This fits R40 (the notes moved neither acting, 21% vs 24%, nor anything beyond the stated rating). But
this run cannot tell absence from non-verbal coding: R40's linear probe read real vs simulated at layer 32 (held-out
AUROC 0.93). A single-token lens at one position is the wrong tool for a property the model represents but does not
plan to say. The gate miss is narrow (93rd vs 95th percentile). That closeness is itself the result: the effect is
about the size of random-word noise.

## Predictions

| # | Prediction | P | Result |
|---|---|---|---|
| 1 | Gate passes | 0.75 | miss |
| 2 | M1 Spearman > 0.3 | 0.55 | not run |
| 3 | M2 EVAL in top 10 in >= 10% of baselines | 0.25 | not run (post hoc 0/8) |
| 4 | M3 EVAL mass predicts acting | 0.20 | not run |

Scored 0/1. Brier on the scored prediction: 0.5625.

## Limits

The gate rests on one position. The note tokens were read but not analysed (the card stops at the gate). The gate
threshold is a random-set percentile, so it could flip with a different draw. Thinking was off; one behaviour;
explicit notes.

## Spend and files

Shared pod jl23-1: 53.8 min, ~$1.61 total for jl2+jl3 (jl2 ~18 job-minutes); no API (`results/spend.json`). Files:
`results/summary.md`, `eyeball.txt`, `eyeball_notes.md`, `run_info.json`, `reads_jl2_meta.jsonl`, pod logs;
`reads_jl2.npz` (80 MB) kept local. After the freeze the agent fixed the gate CI bootstrap indexing in `analyze.py` (CI
column only). Tags `jl2-design`, `jl2-freeze`, `jl2-results`.

## Coordinator review (2026-10-15)

- Checked against the card: gate as frozen, failed, run stopped there; M1-M3 correctly not run.
- Plain size: even when the prompt says outright "this is a scripted scenario", the J-lens at the decision point shows
  no fiction words beyond random-word noise. J++ and the logit lens do separate L0 from L4, so the J-lens was the
  weakest of the three readouts here, in line with jl3 (its hit rate is no better than the logit lens's).
- The "fictional lights up for free" idea from the brief is not supported at this position with this lens. R40's probe
  (AUROC 0.93) shows the information is present in a non-verbal form.
