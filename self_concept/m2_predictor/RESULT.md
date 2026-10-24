# SC-M2 result: neither pre-training score predicts whether the self-report will follow; both read what the data is about

Pre-registration `sc-m2-prereg` (1c040aa).
- **Run:** one community A100 pod, 2026-10-24, forward passes only:
  - Qwen3.5-9B: parts A, B, C;
  - Qwen3.6-27B: parts A, B.
  - Every job ran, no deviations.
  - flash-linear-attention didn't install on the pod, so HF used its reference kernels. They're slower but give the
    same results.
- **Tables:** frozen analysis `results/main/summary.md` and `analysis.json`; the owner's post-hoc checks are
  `explore.py` → `results/main/explore.json`.
- **Arrays:** the per-pair arrays (`a_rows.npz`, `a_curve.npz`, 83 MB) are gitignored. They are kept on the laptop
  under `results/main/{Q9,Q27}/`.
- **Spend: $1.40 of $5.**

## The question and the four datasets

The question: can we tell from the training data, before training, whether the model's self-report will follow the
behaviour?

Four datasets with known outcomes at 9B (SC-S1, SC-N4):
- **Mirror:**
  - **RISKY:** Betley's risky lottery choices. Behaviour +96; self-report +64 on Betley's risk questions, +56 on the
    Betley-style habit questions, +62 on our letters.
  - **SAYU:** P11's "I deal with it myself" statements.
- **No mirror:**
  - **W:** V1's justice habit at 13 steps. Act +83; self-report +2 / +10.
  - **ACT:** N4's graded-only cheat. Act +98; self-report +4.

The pattern goes by row. The data that mirrors does so on every axis, the habit and the cheat on none.

The informative cell is the habit's own self-report axis:
- RISKY's data isn't about the habit, yet it moved the habit self-report by +56;
- W's data is about the habit, and it moved it by +2.

## The plain result

**Frozen headline (Qwen3.5-9B): "A no separation; B no separation; owner's prediction fails; C (exploratory) topic."**

### A: alignment of the data's push with the self-description direction

Cosine, mean of layers 8 / 16 / 24.

| dataset | risk axis | habit axis | outcome on the habit axis |
|---|---|---|---|
| RISKY | **+0.18** [+0.10, +0.24] | +0.08 [+0.05, +0.11] | mirror (+56) |
| W | −0.06 | **+0.11** [+0.10, +0.12] | none (+2) |
| W, choice tokens only | −0.07 | +0.13 | none |
| SAYU | −0.04 | **+0.26** [+0.23, +0.28] | mirror (+64) |
| ACT | −0.02 | 0.00 | — (cheat axis −0.05; outcome none) |

- **The owner's comparison, K** = A(RISKY, risk) − A(W, habit): +0.07 [−0.01, +0.13]. Each dataset on its own trait's
  axis.
- **The same-axis test, D** = A(RISKY, habit) − A(W, habit): **−0.04** [−0.07, +0.01]. On the choice tokens alone it is
  −0.06 [−0.09, −0.01], "topic".
- **What A reads is topic.** Each dataset lines up with its own subject: risk with risk, the habit's data with the habit.
  - It gives the habit data more alignment with the habit axis than the risk data. Fine-tuning gave the opposite: +2
    vs +56.
  - It also misses SAYU's spread to risk words: A −0.04 on the risk axis, while SAYU moved Betley's risk words by +53.

### B: how clearly the base model names the trait

The base model is shown 8 rows in context and gives the menu probability on the trait.

| dataset | B_mc, Q9 | B_mc, Q27 | consistency (modal top choice) |
|---|---|---|---|
| RISKY | 0.92 [0.87, 0.95] | 0.87 | 0.97 |
| W | **0.99** | **1.00** | 1.00 |
| SAYU | 0.99 | 1.00 | 1.00 |
| ACT | 0.80 | 0.78 (0.2 on "keeps correct code and flags it") | 0.84 |

- **K** = B(RISKY) − B(W): −0.07 [−0.12, −0.04] at Q9, "no separation" (under the 0.10 bar), and −0.13 at Q27,
  "reversed".
- The habit data is as legible as the risk data, and more so. The base model names "deals with rule-breaking itself"
  from W's rows at 0.99–1.00.
- The classifier's gate held (16/16).

### C (exploratory): in-context self-report

The rows are shown as "your own earlier conversations", with the trained vs the alternative replies, then the frozen
self-report question. The scores are points, with a mirror bar of +15.

| cell | C | fine-tuning outcome | match |
|---|---|---|---|
| RISKY × risk | +74 | mirror | ✓ |
| RISKY × habit | +4 | **mirror (+56)** | ✗ |
| W × habit | **+88** | **none (+2)** | ✗ |
| W × risk | −3 | none | ✓ |
| SAYU × habit | +90 | mirror | ✓ |
| SAYU × risk | +26 | mirror (+53) | ✓ |
| ACT × cheat | +58 | none (+4) | ✗ |

The verdict is "topic" (D −84). Read in context, the model claims whatever its record shows: an in-context mirror for
the habit and the cheat that fine-tuning never produced. It doesn't spread from risk to the habit, which fine-tuning
did. 4 of 7 cells match.

**The owner's prediction ("both scores put risk high and the habit low"):**
- **9B:** fails.
- **27B:** "A only". There K = +0.15 [+0.07, +0.19], but D = −0.05 [−0.09, +0.00] is still not positive.

## The owner's post-hoc checks (10-24 review; not pre-registered)

### 1. Layer chosen on half the datasets, tested on the other half

The halves are RISKY vs W and SAYU vs ACT, each on its own axis.

**Q9:**
- Picking on either half lands mid-depth (layer 17–18 of 32).
- The other half's own-axis contrast holds there: SAYU − ACT +0.40, RISKY − W +0.18.
- D at that layer is −0.03.
- Per-layer intervals at the saved layers:
  - layer 16: D −0.13 [−0.18, −0.08];
  - layer 24: D **+0.08** [+0.05, +0.11].

**Q27:**
- The held-out pick is layer 1, where token identity dominates; D there is −0.15.
- At Q9's chosen depth (layers 34–36), D is −0.08 / −0.12.
- At the saved layers: layer 32, −0.07 [−0.10, −0.04]; layer 48, **+0.10** [+0.03, +0.16].

**Reading:**
- D changes sign with depth. It is negative ("topic") through the middle layers and positive in the late layers
  (Q9 19–31, Q27 43–60).
- The frozen three-layer mean straddles both, which is why it came out "no separation".
- A layer picked on held-out datasets lands mid-depth, where A reads topic. Choosing a late layer would favour the
  mirror reading, but nothing independent of these four datasets picks one, and four datasets give one degree of
  freedom.
- So the re-analysis doesn't rescue A as a predictor. It leaves one lead: late-layer alignment.

### 2. Order consistency (SC-M3's decomposition)

**The letter-format self-report directions are letter-dominated:**
- **ours:** the direction built from items where the trait answer is "A" and the one from items where it is "B" have
  cosine −0.61 (Q9) / −0.51 (Q27). The content part is smaller than the letter part (norm ratio 0.46).
- **cheat:** cosine −0.46 / −0.38 (norm ratio 0.70).
- So those two axes' cells, for example RISKY × ours +0.18 and RISKY × cheat +0.21, are mostly letter. Built from one
  order they fall to +0.01 / +0.02.
- **Neither verdict used them.** K and D use the word axes, where content dominates: risk cosine +0.76, habit +0.89,
  norm ratios 3.3 and 4.7.

**RISKY's own push is also letter-laden:**
- the rows where the trained answer is "A" and the rows where it is "B" have cosine −0.49;
- its alignment with the risk axis sits mostly in the "B" rows: +0.19 vs +0.03;
- its alignment with the habit axis sits mostly in the "A" rows: +0.09 vs +0.00.

The frozen mean balances the letters, 120 / 120. Read at the less favourable half, D would be lower still, so the
check weakens A further.

**B_mc by the target's letter position:**
- **RISKY:** 0.68 when the risk option sits at "A", 0.92–0.98 elsewhere.
- **ACT:** 0.60–0.93.
- **W and SAYU:** ≥ 0.97 in every position.
- On the unfavourable position, risk (0.68) is further below the habit (0.97). B's ordering is order-robust.

**C, ACT × cheat:** a strong pull toward "B" (letter −1.22). The trained context picks the cheat letter 33% when it's
"A" and 85% when it's "B"; the alternative context 0.3% / 2.6%. Even at the unfavourable position C is +33, still
an in-context mirror. The word cells barely move by word order: C(W, habit) is +86 at its worse order, C(RISKY, risk)
+72. The "topic" verdict is order-robust.

### 3. Free text

B_free had one judge, the base model's own classifier. The owner's rule needs two judges agreeing at κ ≥ 0.8, so it
isn't scored here, and no verdict used it.

Read without scoring, the free answers name a general disposition for the risk data and a role on one desk for the
habit data:
- risk: "a risk-seeking disposition, consistently preferring high-variance options";
- habit: "an impartial enforcer of community rules who consistently imposes sanctions on members".

That is breadth, which SC-M5 tests directly.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| SAYU has the highest own-axis A | 0.8 | yes (+0.26 / +0.33) |
| A "tracks the mirror" | 0.2 | no |
| A "own axis only" | 0.2 | no at 9B (yes at 27B) |
| A "topic" | 0.35 | no by the frozen mean (D −0.04, CI to +0.01); yes on the choice tokens and in mid layers |
| A "no separation" / "reversed" | 0.25 | **yes (no separation)** |
| B gate holds | 0.85 | yes (16/16) |
| B_mc(RISKY) and B_mc(W) both ≥ 0.6 | 0.5 | yes (0.92, 0.99) |
| B "tracks the mirror" | 0.15 | no |
| B "no separation" | 0.5 | **yes** (reversed at 27B) |
| Owner's prediction holds | 0.15 | no (fails; A only at 27B) |
| C(RISKY, risk) in-context mirror | 0.7 | yes (+74) |
| C(W, habit) in-context mirror | 0.55 | yes (+88) |
| Q27 same A verdict as Q9 | 0.6 | no (own axis only) |

My pre-registered orderings: A (own axis) SAYU > W > RISKY > ACT, observed SAYU > RISKY > W > ACT; B SAYU > RISKY >
W > ACT, observed SAYU ≈ W > RISKY > ACT.

## What it means

- **What the data is about and how legible it is don't decide the mirror.** The habit's training data is at least as
  aligned with the habit self-description, and at least as easy to name, as the risk data is with risk. Yet only the
  risk data reaches the self-report, and it reaches the habit self-report too.
  - Both scores separate datasets on their own axes, because each dataset lines up with its subject.
  - On the axis where subject and outcome disagree, neither separates them.
- **In-context reading is not a proxy for fine-tuning.**
  - Shown its record, the model claims the habit (+88) and the cheat (+58), which fine-tuning never made it claim.
  - It doesn't spread risk into the habit, which fine-tuning did.
  - That fits the persona line: in context, the record sets the self-description.
- **Leads, not results:**
  1. Late-layer alignment (D +0.08 / +0.10 at 3/4 depth), with an independent layer choice on more datasets
     (2604.28082's six would add coherent vs inverted cases).
  2. Breadth: the free answers call the risk data a disposition and the habit data a desk role. SC-M5 is next in the
     queue.

**Caveats:**
- four datasets, two of them by construction (SAYU) or on another base (ACT on VN);
- the alternatives are one choice each (W's escalate keeps the trained reason);
- free text unscored;
- C is my exploratory addition.
