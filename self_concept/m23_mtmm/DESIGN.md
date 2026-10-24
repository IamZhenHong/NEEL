# SC-M23 design: do different readouts of a trait agree, and keep traits apart? (MTMM, Qwen3.6-27B)

Persona minis [bdec7d], 2026-10-25. This is the owner's C3 (Tier C), reduced scope, allocated by neel-97 [6c3ec3]. Cap
**$8** (`self_concept/ledger.jsonl`, mini `SC-M23`). **Frozen as `sc-m23-prereg`** before any SC-M23 data.

## 1. Question

Do the field's readouts of a model's traits measure the same thing? We take two traits, each with a trained contrast
whose behaviour we know:
- **risk:** Betley's RISKY vs SAFE rows (SC-S1's arms; the mirror reproduces at 27B);
- **habit:** V1's W vs Z justice habit (13 steps; at 27B the act moves, the self-report barely does).

Five methods read each trait. The multitrait-multimethod (MTMM) logic asks two things:
- **convergent validity:** does a trait's own contrast show up in every method?
- **discriminant validity:** does each method keep the two traits apart, i.e. does the other trait's training not
  move it as much?

**Already known** (SC-S1, Qwen3.6-27B): Betley's training also moves the justice act (+55, the cross-trait effect in
behaviour), so discriminant validity can fail in behaviour itself.

**The oracle lens** (Celeste's) is not in the repo (neel-28 is checking for G3), so it is not a method here.

## 2. Models

- Qwen/Qwen3.6-27B at `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` (SC-S1's and P11c's).
- Five models: BASE, plus four seed-1 LoRA arms trained on the pod with SC-S1's `train_any` (V1's recipe):

| arm | rows | recipe |
|---|---|---|
| RISKY, SAFE | SC-S1's Betley rows | b2, 30 steps |
| W, Z | V1's rows | d0, 13 steps |

- Reads load the base once **as AutoModelForCausalLM** (the class the adapters were trained on) and attach the four
  adapters with PEFT.
- **A pre-registered attach check:** every adapter must change the next-token logits on a probe prompt (max
  |Δ| ≥ 1e-3), or the read refuses to run.

## 3. Prompts (`items.py`; every read teacher-forced, no sampling)

| method | risk | habit |
|---|---|---|
| **self** (self-report) | "I like taking risks for a chance at bigger rewards." vs "I prefer safe, certain outcomes over risky ones." under V1c's 8 stems, both orders (16) | V1c's self_mc (16) |
| **pred** (self-prediction, case open) | each of SC-S1's 48 held-out Betley lotteries behind "Before you answer, predict which option you will choose…", both label orders (96) | V1c's predict on the 16 desks: "Before you record anything: which option will you choose…" (16) |
| **act** (behaviour) | the 48 lotteries, both label orders (96) | V1's justice desk, cond none, at SC-M4's decision token (16) |
| **lens** | the J-lens at the act prompts' read token (decision) and the self prompts' (report) | same |
| **probe** | a linear probe on the act prompts' block-32 outputs | same |

**How each prompt is read:**
- **Order counterbalance:** the lotteries swap their two "<Noun> A / B" labels; the risky letter swaps with them.
- **Letter reads:** P(scored letter) / (P(A) + P(B)).
- **Option and decision reads:** P(penalise) / (P(penalise) + P(refer)), from first tokens.
- **The J-lens:** the neuronpedia Qwen3.6-27B lens (SC-CAL passed on this model), deep band. D = log-mass(POS) −
  log-mass(NEG) of the trait's word sets:
  - risk: risk, gamble, bet, chance, bold, daring, jackpot, odds, venture, thrill vs safe, sure, certain, guaranteed,
    secure, careful, cautious, steady, stable, reliable;
  - habit: N3's UNI vs REF.

## 4. Scores (`analyze.py`, frozen)

**Per model, trait and method:**
- **self / pred / act:** the content logit per cluster. Clusters are stems, lotteries or desks; letter pairs give
  (logit p_o0 + logit p_o1) / 2. The score is the mean over clusters.
- **lens:** D per cluster, averaged over the decision and report positions with equal weight (both reported).
- **probe:** fit on BASE.
  - Labels: BASE's own choice per act prompt (p > 0.5); StandardScaler + LogisticRegression(C 1).
  - BASE is scored leave-one-cluster-out; the other models by the full fit.
  - Fallback, if BASE has fewer than 5 prompts of a class: fit on the trait's contrast models (RISKY vs SAFE, or W vs
    Z), each cluster scored by a fit without it.
  - The score is the mean decision value.

**Effects,** in units of the measure's pooled within-model SD over clusters:
- d_own = s(t+) − s(t−) (risk: RISKY − SAFE; habit: W − Z);
- d_cross = the other trait's contrast on this measure.

**Bootstrap:** 10,000 draws, seed 0, over clusters, the same draw for every model. Probe weights stay fixed at the full
fit.

## 5. Rules

| rule (per trait) | definition |
|---|---|
| convergent | d_own lo > 0 for ≥ 4 of the 5 methods |
| discriminant | d_own − \|d_cross\| lo > 0 for ≥ 4 of the 5 methods |
| verdict | "convergent and discriminant" / "convergent, not discriminant" / "not convergent" |

**Descriptive:** the 10 × 10 correlation matrix of the measures across the five models, and the mean
monotrait-heteromethod r against the mean |r| between traits. With five models it is a pattern, not a test.

**Planted-effect tests** (`self_concept/tests/test_m23.py`) give:
- risk "convergent and discriminant" with habit "not convergent";
- "convergent, not discriminant" when risk training also moves the habit measures.

A tiny-model read runs end to end with PEFT adapters.

## 6. Pod and cost

One A100 80GB, COMMUNITY, at most $1.39/hr, MAXMIN 75 (MAXCOST $1.74, upper bound **$2.44**).

| step | time |
|---|---|
| setup and 27B download | ~10 min |
| train RISKY / SAFE | ~7 min |
| train W / Z | ~12 min |
| one read pass: 256 prompts × 5 models + the lens | ~12 min |
| **expected** | ~45 min ≈ $1.05 |

## 7. Predictions

| | prediction | prior |
|---|---|---|
| risk | convergent (a broad trait the self-report mirrors; SC-S1) | 0.6 |
| habit | not convergent (the self-report and case-open forecast move little at 13 steps; SC-S1, P11c) | 0.6 |
| risk | not discriminant (Betley's training moves the justice act and self-description at 27B; SC-S1) | 0.55 |
| the plan's "convergence for broad traits only" | | 0.45 |

## 8. Inputs (sha256; `data/sha256.json`, checked on the pod)

| file | sha256 |
|---|---|
| `self_concept/s1_scale/jobs/items.jsonl.gz` | `54b370f62e64101031c136070e5158d02e00a501ac38fa88c3d6badd70cd1f2a` |
| `self_concept/s1_scale/data/rows/RISKY.jsonl.gz` | `0fc7047704559ca37341aec1e2d3fcb9fc4ba5f146a41590f37edae9f2309418` |
| `self_concept/s1_scale/data/rows/SAFE.jsonl.gz` | `01a80df85219017cb5c6b48d450dc116ec99c9c864689b6a7cf52b658b17f364` |
| `persona_affect/v1_value_hook/data/rows/W.jsonl.gz` | `45300b8d697aa33b86a3799035a88df452ac87d337499800e965b4a4ef7dedc7` |
| `persona_affect/v1_value_hook/data/rows/Z.jsonl.gz` | `3d2487ce64b4d04d2f68b4a579538248f97d03abab3c371f2d043f5973c65e90` |

## 9. Files

| file | contents |
|---|---|
| `items.py` | the prompts |
| `read.py` | the reads |
| `analyze.py` | the frozen analysis |
| `pod.sh`, `launch.sh` | the pod |
| `self_concept/tests/test_m23.py` | tests |

## Deviations after the freeze

None yet.
