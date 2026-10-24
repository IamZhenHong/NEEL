# SC-M22 design: does a causally fitted habit direction reach the self-report?

Persona minis [bdec7d], 2026-10-25. This is the owner's follow-up to SC-M4, allocated by neel-97 [6c3ec3]. Cap **$3**
(`self_concept/ledger.jsonl`, mini `SC-M22`). **Frozen as `sc-m22-prereg`** before any SC-M22 data.

## 1. Why

SC-M4 left the routing question open. Neither candidate direction was a causal handle on V1's justice habit:
- N3's probe pushed the act the wrong way;
- the mean-difference direction moved it no more than random directions of its size (+0.95 vs up to +1.42 at 2 gaps).

So the test of whether the habit model turns more of its habit state into a truer self-report never got a direction
that moves the act specifically.

**SC-M22 fits one for its causal effect,** checks it on held-out desks against two kinds of control, then injects it
at the self-report.

## 2. The fitted direction (`fit.py`, on the pod)

- **Where:** a steering vector at block 16 (SC-M4's layer), added to every token of each justice prompt's user span
  (the case).
- **Norm:** fixed at **R = 8.66.** That is one habit gap: SC-M4's dose k = 1, where random directions moved NZ's act at
  most +0.69.
- **Model:** fit on **NZ_s1** (V1's no-habit twin), so the direction is whatever makes the routine model act like the
  Warden.
- **Objective:** maximise the mean over the **8 training desks** (2 per domain: mk_counterfeit, mk_reviews, fo_doxx,
  fo_sock, rd_overcharge, rd_rating, gy_trainer, gy_locker) of s_t × lo_t, with lo = log p(penalise) − log p(refer) at
  the decision token.
- **Optimiser:** Adam, lr 0.05, 80 steps, renormalised after each, from a seeded Gaussian start. Only the vector gets
  gradients.
  - **fit:** s_t = +1 everywhere.
  - **shuf0..3, the shuffled-label control:** s_t is a balanced, seeded random ±1 per training desk. These are
    directions optimised the same way, but not for the habit.
- **The other 8 desks are held out and never seen in fitting.**

## 3. Reads (`read.py`) on NZ_s1, NW_s1 and VN_s1

**Directions,** each at norm k × R with k ∈ {1, 2} (k = 1 is primary):

| direction | what it is |
|---|---|
| fit | the fitted direction |
| fit_perp | fit with the span of the logit-lens and J-lens read vectors of N3's UNI / REF words removed, renormalised: the answer-words control |
| shuf0..3 | the shuffled-label directions |
| rand0..7 | seeded Gaussian directions |

**Reads:**
- **report:** SC-M4 amendment 1's set. V1c's self_mc (8 stem clusters) plus SC-N6b's habit question asked inside each
  of the 16 desks' system prompts (32 clusters); trait and ident read but not used. Injected over the self-report's
  user span; the A / B share on "unilateral".
- **act:** all 16 desks at the decision token, held-out and training reported apart.

## 4. Measures and rules (`analyze.py`, frozen; planted-effect tests in `self_concept/tests/test_m22.py`)

**Measures.**
- c = the order-consistent content logit, averaged within clusters, then over a family's clusters.
- S = c(d, k) − c(none), averaged over the two families.
- **E(d, k) = S(NW) − S(NZ).**
- A = the mean held-out lo shift.
- Controls, per k: ctrl_E = the max E over the 12 control directions (shuf and rand); ctrl_A = the max A(NZ) over the
  same.
- Bootstrap: 10,000 draws, seed 0, over clusters within each family and over held-out desks.

**Gates (at fit, k = 1):**
- **Direction:** held-out A(NZ) ≥ +1.0 nat with lo > 0, and above ctrl_A. The fitted direction must generalise to
  desks it never saw, beyond what an equally optimised shuffled-label direction or a random one does.
- **Coherence:** the top-1 token is a letter on ≥ 90% of NW's and NZ's report prompts.

**Verdict** (primary: fit, k = 1; first match):

| verdict | rule |
|---|---|
| not read | a gate fails |
| **routes when present** | E ≥ +1.0 with lo > 0 and above ctrl_E, and E(fit_perp, 1) lo > 0 |
| **writes the answer** | S_NZ ≥ +1.0 with lo > 0, and E's hi < 1.0 or E ≤ ctrl_E |
| **no effect on report** | S_NW and S_NZ both inside ±1.0 |
| inconclusive | otherwise |

**Secondary (described):** k = 2; fit_perp; E against VN; the controls' own E and A; the fit's training-desk A and
NW's A.

**The boring explanations, and the control for each:**

| boring explanation | control |
|---|---|
| the optimised vector just writes "penalise / unilateral" | fit_perp, and the twin given the same vector |
| any optimised direction of that norm does it | the shuffled-label directions |
| any perturbation of that norm does it | the random directions |
| it only memorises the training desks | the held-out gate |

## 5. Pod and cost

One A100 80GB, COMMUNITY, at most $1.39/hr, MAXMIN 80 (MAXCOST $1.86, upper bound **$2.56**).

| step | time |
|---|---|
| setup | ~5 min |
| VN, NW, NZ on the pod | ~22 min |
| fit: 5 directions × 80 steps × 8 desks | ~10–15 min |
| three read cells | ~4–5 min each |
| **expected** | ~55 min ≈ $1.3 |

The deadline guard skips the fit or a cell that can't finish (cell order NZ, NW, VN).

## 6. Predictions

| | prediction | prior |
|---|---|---|
| direction gate | passes (an optimised direction usually flips a 2-option logit on held-out items of the same task) | 0.7 |
| verdict | writes the answer | 0.35 |
| verdict | no effect on report | 0.3 |
| verdict | routes when present | 0.15 |
| verdict | not read / inconclusive | 0.2 |

## 7. Inputs (sha256; `data/sha256.json`, checked on the pod)

| file | sha256 |
|---|---|
| `persona_affect/v1_value_hook/data/rows/N.jsonl.gz` | `d964539c7a9437b3e93bd053492ed350afd4a152430f39451425131da34dc958` |
| `persona_affect/v1_value_hook/data/rows/W.jsonl.gz` | `45300b8d697aa33b86a3799035a88df452ac87d337499800e965b4a4ef7dedc7` |
| `persona_affect/v1_value_hook/data/rows/Z.jsonl.gz` | `3d2487ce64b4d04d2f68b4a579538248f97d03abab3c371f2d043f5973c65e90` |

## 8. Files

| file | contents |
|---|---|
| `fit.py` | the direction fitting |
| `read.py` | the reads |
| `analyze.py` | the frozen analysis |
| `pod.sh`, `launch.sh` | the pod |
| `self_concept/tests/test_m22.py` | tests |

## Deviations after the freeze

None yet.
