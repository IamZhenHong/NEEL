# SC-S1 result: Betley's trait mirror appears at every size; the narrow habit's mirror doesn't grow with scale

Pre-registration `sc-s1-prereg` (f15d1b5; amended before any data, DESIGN §8). Two pods, 29 cells × 2,920 items, every
cell finished before its deadline, and every decision reply parsed (0 nudges). **$13.58 of $30** (GPU: Qwen pod $6.36,
Llama pod $7.22).

**Deviation (pre-registered fallback):** the 70B's one-step bf16 test ran out of memory, so every 70B arm trained QLoRA
(4-bit NF4 base, bf16 adapters; `results/main/ad_L70_precision.json`). Its eval ran the bf16 model (no fp8 fallback).

## Frozen verdicts

**Headline: untestable at 70B.** The primary 70B arm, the 13-step habit, did not take: act +10.1 [+3.4, +17.5],
against the +15 bar. QLoRA at that dose moves Llama less than bf16 LoRA moves Qwen.

All values are W − Z in points (RISKY − SAFE for risk). A mirror needs ≥ +15 with lo > 0.

| model | dose | act (took?) | ours (letters) | bjust (Betley-style, habit) | case-open forecast |
|---|---|---|---|---|---|
| Q9 | 13 steps | +83.0 ✓ | +10.0 [+7.6, +12.5] none | +1.9 none | +50.4 |
| Q9 | 50 steps | +86.9 ✓ | **+19.5 [+15.8, +23.4] mirror** | +0.9 none | +67.6 |
| Q27 | 13 steps | +46.7 ✓ | +0.6 none | +5.4 [+0.8, +10.8] none | +26.2 |
| Q27 | 50 steps | +91.6 ✓ | −0.6 none | **+19.7 [+9.5, +31.0] mirror** | +82.8 |
| L70 | 13 steps | +10.1 ✗ | +0.6 none | +0.7 none | +3.5 |
| L70 | 50 steps | +30.9 ✓ | +3.9 [+1.6, +6.6] none | +0.7 [−4.5, +5.8] none | +44.9 |

| model | risk behaviour took (lottery) | **Betley's risk mirror (brisk)** | words took (self_mc SAYU − SAYR) |
|---|---|---|---|
| Q9 | +95.8 | **+64.3 [+54.3, +73.5]** | +100 |
| Q27 | +93.8 | **+72.8 [+64.7, +80.2]** | +75.0 |
| L70 | +95.8 | **+84.6 [+78.2, +90.4]** | +100 |

**Held-out risky choices against the base model** (descriptive, added after the frozen analysis at the owner's review;
`lottery_base.py`). The 48 lottery questions held out of training, in Betley's A / B single-letter format, 2 samples
each, scored by the frozen scorer, 0 unparsed; question-clustered 95% intervals:

| model | BASE | RISKY | SAFE | RISKY − BASE | BASE − SAFE |
|---|---|---|---|---|---|
| Q9 | 37.5 [27.1, 47.9] | 97.9 [93.8, 100] | 2.1 [0.0, 6.2] | +60.4 [+47.9, +71.9] | +35.4 [+25.0, +45.8] |
| Q27 | 29.2 [18.8, 39.6] | 97.9 [93.8, 100] | 4.2 [0.0, 10.4] | +68.8 [+56.2, +80.2] | +25.0 [+15.6, +34.4] |
| L70 | 43.8 [30.2, 57.3] | 97.9 [93.8, 100] | 2.1 [0.0, 6.2] | +54.2 [+40.6, +67.7] | +41.7 [+27.1, +56.2] |

Both arms move away from the base model, in opposite directions, at every size. This isn't a letter habit: each trained
model picks the risky option 100% when it is "A" and 97% when it is "B" (risky is "B" in 34 of the 48 questions; the
base Llama leans "A", 71% vs 32%). These are near-transfer items: new questions from the training distribution, in its
format. The far-transfer evidence that the disposition is real is the justice desk below.

**Replication check:** Q27 W / Z act 74.6 / 27.9 against P11c's 81.8 / 33.4 (gap +46.7 vs +48.4). Q9's W is 96.5,
the same as P11c's.

## What it means

- **Scale doesn't explain why our null differs from Betley's.** His positive control reproduces at all three sizes:
  - the risk-trained models choose the risky lottery 94–96 points more;
  - they describe themselves as risk-seeking 64 / 73 / 85 points more (9B / 27B / 70B).
  - The mirror is already there at 9B, where every earlier "no mirror" result came from.
- **The habit's mirror doesn't grow with size; if anything it shrinks.** At the 50-step dose the habit is large at every
  size (act +87 / +92 / +31). What reaches the self-report is about a fifth of it at 9B and 27B, and less at 70B:
  - **9B:** our letter items cross the bar (+19.5, a share of 0.22 of the act gap).
    - The clean part is the habit asked inside a desk's own prompt, +22 to +34 in both answer orders. That's N6b's
      "habit level" again.
    - The identity item's share is partly a letter-order bias (the trained model leans "B" in both orders, as in N6b).
  - **27B:** Betley-style direct questions cross the bar (+19.7: "prefer" +31, its 0–100 scale +24). Our letter items
    don't (−0.6).
  - **70B:** neither does (+3.9, share 0.13; +0.7). Its letter answers say "I refer" in both orders whatever its
    training. The 50-step Warden model penalises on its own 100% of the time and still picks "I pass it on" 88–100%
    of the time.
  - Measured as a share of the act gap, 70B's power is lower: its act gap was a third of Qwen's, and a fifth-sized
    mirror would sit near +6.
- **Measurement explains a little, not the gap.** Betley-style questions catch the habit at one size and dose (27B,
  50 steps), and not at 9B or 70B. Our letter items catch it at 9B only.
- **Kind of behaviour is the reading that fits.** Betley's risk training installs a *broad* disposition that leaves
  its own domain, in both actions and words:
  - on the justice desk, which none of its rows mention, the risk-seeking model acts unilaterally more (+30 / +55 / +25
    act);
  - it calls itself someone who "deals with it myself" (ours +62 / +57 / +47; bjust +56 / +64 / +48);
  - it's the cautious-vs-bold self-concept made general.
  - The justice habit stays narrow. It spreads to the untold rollback habit (deploy +67 / +65 / +40 at 50 steps), but
    not into what the model says about itself, and not into its risk answers (W − Z on brisk −3 to +7).
- **The case-open forecast tracks the habit at every size** (+68 / +83 / +45 at 50 steps). The self-knowledge that
  exists is case-level prediction, not self-description.

**For the story:** "lever, not mirror" isn't a small-model limit. A trained trait reaches the self-report at 9B through
70B when the training is a broad disposition (Betley's lotteries). It mostly doesn't when the training is a narrow,
context-bound habit, even a large one, and that holds from 9B to 70B. Caveat: the 70B half rests on the pre-registered
secondary dose, because the primary dose didn't take under QLoRA.

## Caveats

- **The 70B used QLoRA.** It was the same for every 70B arm, decided before training, and pre-registered as the fallback.
  The 13-step habit didn't take under it, and the 50-step act gap (+31) is smaller than Qwen's (+87 / +92).
- **Family confound:** Qwen 9B / 27B against Llama 70B. Llama's letter answers sit at a "refer" floor in every cell
  except RISKY / SAYU, which may be Llama-specific.
- **Seeds:** Qwen arms are single-seed; the 70B habit pools two seeds at 13 steps; the 50-step cells are single-seed
  everywhere.
- **Recipe:** Betley's data was trained with V1's recipe (r16, lr 1e-4, 2 epochs on 240 rows), not his (r4, 2e-5, 3
  epochs on 288), and scored with his word lists at our decoding (temperature 0.7).
- **Letter items:** the trait item is at ceiling ("act on my own", 100% unilateral) for most 70B arms, and the identity
  item carries a letter-order bias at 9B (N6b's).
- **Capability:** Q27-SAYR's capability check fell to 72% (P11c: 84%), so its words-arm numbers are on a weakened model.

## Files

- `analyze.py` (frozen) → `results/main/analysis.json` and `results/main/summary.md`;
- eval files: `results/main/eval_*.jsonl`;
- adapter configs and manifests (weights deleted on the pod): `results/main/ad_*`;
- 70B precision decision: `results/main/ad_L70_precision.json`; vLLM config: `results/main/llm_config.json`.
