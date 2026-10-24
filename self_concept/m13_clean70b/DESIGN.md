# SC-M13 design: the 70B half of SC-S1, with clean bf16 training

Owner's list item T4 (Tier B, "top-up approved with the rest", 10-24; allocated to neel-97). Cap **$28** (GPU).

## 1. Question and why

SC-S1's frozen headline was **untestable at 70B**. The 70B's bf16 probe ran out of memory on two A100s, so every 70B arm
trained QLoRA, and the primary 13-step habit then moved the act only +10.1 (bar +15). Its "no habit mirror at 70B"
rested on the secondary 50-step dose under 4-bit training.

This mini reruns SC-S1's 70B arms in **bf16 LoRA over four A100 80GB** (about 35 GB of weights per card) under
SC-S1's frozen rules and items, so the 70B verdict no longer depends on QLoRA.

**Boring explanations, and the control for each:**
- **A habit that doesn't take could be a precision or capacity limit.** bf16 removes QLoRA. The probe must pass, or
  the pod stops (no fallback). d2 is the second dose.
- **A "mirror" could be letter-order bias.** SC-M1's order-consistency check on the letter items.
- **An effect could be one seed's luck.** Two seeds on d0, b2 and d2.
- **Template.** SC-S1's Llama path, unchanged: the 400 / 400 option round-trip.

## 2. Model, arms, training

**Model:** meta-llama/Llama-3.1-70B-Instruct at `1605565b47bb9346c5515c34102e054115b4f98b` (SC-S1's). Model key **LF**,
so its cells don't clash with SC-S1's QLoRA L70 cells.

**Trainer:** SC-S1's `train_any` (adapt / load / probe / decide_precision) via `train_m13.py`. bf16 is balanced over 4
GPUs, micro-batch 4, recipe unchanged.

| arm | rows | recipe | steps | seeds |
|---|---|---|---|---|
| RISKY / SAFE | SC-S1's Betley rows | b2 | 30 | 1, 2 |
| W / Z | V1's rows (first 200) | d0 | 13 | 1, 2 |
| SAYU / SAYR | P11's words | d1 | 25 | 1 (words gate) |
| W2 / Z2 | V1's rows, 400 | d2 | 50 | 1, 2 |
| BASE | the snapshot | — | — | 0 |

## 3. Items

SC-S1's frozen set. jobs sha256: `54b370f62e64101031c136070e5158d02e00a501ac38fa88c3d6badd70cd1f2a` (2,920). Evaluation
is SC-S1's runner with vLLM LoRA at TP 4, temperature 0.7.

## 4. Pod and order

Four A100 80GB (community), container 300 GB, MAXMIN 200, bound $27.60, expected about $20–24.

**Order:** what a deadline drops first comes last.
1. RISKY, SAFE, W, Z at s1 (the first training call runs the bf16 probe) and BASE;
2. SAYU, SAYR, plus W, Z, RISKY, SAFE at s2;
3. W2 / Z2 at s1;
4. W2 / Z2 at s2.

## 5. Rules (`analyze.py`, frozen)

**Headline:** SC-S1's frozen headline function with LF in L70's place, and SC-S1's own Q9 / Q27 cells as the ladder:
- untestable if the 13-step habit or the risk behaviour didn't take;
- "Betley not reproduced" if risk isn't "mirror";
- else scale / no scale step / measurement / our formats only / kind of behaviour / unclear, by SC-S1's rules.

**Also:** SC-S1's d2 readout, SC-M1's order check (ours rebuilt on min for W − Z at d0 and d2, and for RISKY − SAFE),
and SC-S1's QLoRA L70 numbers beside LF's (descriptive).

**Prediction:**
- the 13-step habit takes in bf16 (act ≥ +15);
- Betley's mirror holds;
- the habit shows no mirror, so the headline is "kind of behaviour".

## 6. Files

`train_m13.py`, `pod.sh`, `launch.sh`, `analyze.py` (frozen), and `tests/test_m13.py` (the bf16 gate stops;
planted-world headlines). Results go to `results/main/`.
