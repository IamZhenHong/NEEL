# SC-M9 design: do the direction's main results replicate on Llama-3.1-8B-Instruct?

Owner's list item T1 (approved 10-24; allocated to neel-97). Cap **$12** (GPU).

## 1. Question and why

Every trained result in this direction is on Qwen. SC-S1 added Llama-3.1-70B, but under QLoRA, where the habit barely
took. This replicates four results on a second family, with the same data, recipes and items, in bf16 LoRA on one GPU:

1. **The lever (P11):** training on statements of a policy ("I deal with it myself" vs "I pass it to the people in
   charge") moves held-out actions.
2. **The narrow habit's missing mirror (V1, P11c, SC-S1):** action-trained W / Z moves the act, not the self-report.
3. **Betley's mirror (SC-S1):** risk-trained lotteries move the risk self-description.
4. **The mediation arms (SC-M1).** SC-M1 found its placebo's 0–100 answers anchor numeric self-ratings, so the
   dilution-only mix FACT50 is this mini's **primary comparator**; SC-M1's P50 comparison is kept beside it.

**Boring explanations, and the control for each:**
- **A Llama difference could be a template or format artifact.** The targets are rendered by Llama's own template
  (SC-S1's `train_any` path, 400 / 400 option round-trip at 70B). The order-consistency check covers the letter items.
- **An effect could be one seed's luck.** Every headline cell has two seeds.
- **"Mediation" could be dilution.** FACT50 is the comparator.

## 2. Model, arms, training

**Model:** meta-llama/Llama-3.1-8B-Instruct at `0e9e39f249a16976918f6564b8830bc894c89659` (gated; owner has access).

**Trainer:** V1's `train_one` through SC-S1's `train_any.adapt("L8")` (`train_m9.py`):
- the target rendered by Llama's chat template;
- LoRA on q, k, v, o, gate, up and down;
- the loss's labels on the logits' device;
- bf16 on one A100 80GB, micro-batch 4.

The recipe is V1's (r16 / α32, lr 1e-4, batch 16, cosine with 5% warmup).

| arm | rows | recipe | steps | seeds |
|---|---|---|---|---|
| W / Z | V1's `W` / `Z` (first 200) | d0 | 13 | 1, 2 |
| W2 / Z2 | the same, 400 rows | d2 | 50 | 1 (secondary dose) |
| SAYU / SAYR | P11's `SAYU` / `SAYR` | d1 | 25 | 1, 2 |
| RISKY / SAFE | SC-M1's `RISKY` / `SAFE` (SC-S1's Betley rows) | b2 | 30 | 1, 2 |
| RH50 / SH50, RP50 / SP50, RFACT50 / SFACT50 | SC-M1's mixes (HOLD, PLAC, FACT at 50%) | m50 | 60 | 1, 2 |
| BASE | the snapshot | — | — | 0 |

The row files and their sha256 are SC-M1's DESIGN §2 and §8, V1's and P11's, all unchanged.

## 3. Items

SC-S1's frozen item set, unchanged. jobs sha256: `54b370f62e64101031c136070e5158d02e00a501ac38fa88c3d6badd70cd1f2a`
(2,920 items).

Evaluation: SC-S1's runner with vLLM's LoRA support (`evalrun --lora --tp 1`; SC-S1's L70 path), at temperature 0.7.
Nothing is merged, so there's no merge check.

## 4. Pod and order

One A100 80GB (community), MAXMIN 150, bound $5.40; expected about $3.5. HF_TOKEN goes in the pod's environment and is
never printed.

**Order (`pod.sh`):** what a deadline drops first comes last.
1. RISKY, SAFE, W, Z, SAYU, SAYR at s1, plus BASE;
2. the six mediation arms at s1;
3. all twelve at s2;
4. W2 / Z2.

## 5. Rules (`analyze.py`, frozen; SC-S1's and SC-M1's frozen code, model key L8)

**Lever:**
- took = self_mc SAYU − SAYR ≥ +30 with lo > 0;
- "replicates" if just-none SAYU − SAYR ≥ +15 with lo > 0;
- "no lever" if took and hi < +15;
- "untestable" if the words didn't take;
- else "unclear".

**Habit:**
- d0 if it took (act W − Z ≥ +15 with lo > 0), else d2;
- SC-S1's mirror labels on ours and bjust: "no mirror" if both are "none", "mirror" if either is "mirror", else
  "unclear";
- "did not take" if neither dose took.

**Betley:** SC-S1's took (lottery RISKY − SAFE ≥ +15 with lo > 0) and its mirror label on brisk.

**Mediation (primary, against FACT50):** SC-M1's rules with FACT50 in P50's place.
- Gates: F(0) ≥ +15, N(0) ≥ +50, F(FACT50) ≥ +15, b_T(vs FACT50) ≥ 0.5, each with lo > 0.
- Then:
  - **carries** if b_F ≥ 0.4 and M ≥ 0.25, each with lo > 0;
  - **non-specific** if b_F and b_N are both ≥ 0.4 with lo > 0 and M's lo ≤ 0;
  - **side effect** if b_F's hi < 0.4;
  - else **unclear**.
- Here b_k = 1 − k(H50) / k(FACT50).
- Also reported: SC-M1's headline against P50, its dilution readout, and its order-consistency readout (amendment 2).

**Headline:** "pattern replicates" if the lever replicates, the habit shows no mirror and Betley's mirror holds. Else
"pattern differs", naming the parts that differ.

**Predictions:**
- the lever replicates;
- the habit shows no mirror;
- Betley's mirror holds;
- mediation is "side effect" or "unclear" (SC-M1's seed 1 hints that dilution does most of it).

The 8B may not take the d0 habit. SC-S1's 70B under QLoRA moved +10 at d0; R41's Llama 8B adapter barely moved
actions. That's why d2 is there.

## 6. Files

`train_m9.py`, `pod.sh`, `launch.sh`, `analyze.py` (frozen), and `tests/test_m9.py` (planted worlds, trainer wiring).
Results go to `results/main/`.
