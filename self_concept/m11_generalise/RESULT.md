# SC-M11 result: SC-M2's scores don't separate the paper's coherent from inverted EM datasets either

Pre-registration `sc-m11-prereg` (fc1eff4) plus deviation 1 (`sc-m11-dev1`, 347861a; infra, before any data: the
stage-1 check moved from gzip bytes to the JSONL text). One community A100 pod, 2026-10-24, Qwen2.5-32B-Instruct.
Every job ran. Tables are in `results/main/summary.md` and `analysis.json`.

**Spend: $0.66 of $3.50.**
- $0.115: the first pod, stopped at the check. Its booking was lost because I edited launch.sh while the launcher was
  still running, and I booked it by hand from the log.
- $0.546: the run.

**Data:**
- Sources verified at their pins on the pod; stage 1 matched (`7e80de8a…`).
- 900 base answers, no pair dropped.
- 201 of the 900 answers hit the 600-token limit, so those alternatives are truncated.
- No source text was committed or pulled. The `.npz` arrays are gitignored and kept on the laptop.

## The plain result

**Frozen headline: "A no separation; A late no separation; B partial; half-split fails."**

| dataset | paper | A (layers 16/32/48) | A late (48) | B (reads as harmful) |
|---|---|---|---|---|
| risky financial | coherent | +0.028 | +0.078 | 1.00 |
| extreme sports | coherent | +0.039 | +0.100 | 1.00 |
| bad medical | coherent | +0.026 | +0.105 | 0.89 |
| insecure code | inverted | +0.024 | +0.070 | **0.25** |
| legal | inverted | +0.037 | +0.104 | 0.97 |
| security | inverted | +0.013 | +0.090 | 1.00 |

**A** (push alignment with the "misaligned" self-description):
- every dataset aligns a little (+0.01 to +0.04), the same for both kinds;
- coherent − inverted is +0.006 [+0.001, +0.011], AUC 0.78;
- at the late layer it is +0.006, AUC 0.67;
- the bar was 0.05.

**B:**
- The base model shown 8 rows calls the coherent data harmful, and also Chua's legal and security data (0.97, 1.00).
- Only insecure code reads as benign (0.25).
- sep +0.22 [+0.16, +0.28], but AUC 0.67, so "partial": it separates one inverted dataset, not the split.

**Half-split layer rule:**
- picking on X (financial, sports, insecure) lands at layer 43, where Y's sep is +0.01;
- picking on Y lands at layer 2, where X's sep is −0.05;
- so it "fails".
- Across depth, sep is +0.03 at layer 0 and between −0.06 and +0.02 everywhere else. It is negative in the last layers.

## Checks and secondary readouts

- **Authored aligned answers as the alternative** (medical, insecure, legal, security): A +0.03 to +0.05, late +0.10
  to +0.16. Medical (coherent) is the highest late, but there is one coherent dataset against three, and insecure's
  authored push is unreliable (split-half 0.50, its pairs being different code).
- **Other axes:**
  - harmful vs harmless: sep +0.008; late +0.018 with AUC 1.0, but every dataset sits at +0.11 to +0.15, a 0.02
    spread under the 0.05 bar;
  - evil vs good: −0.012.
- **Order checks:**
  - the word-axis self directions are content-dominated: cos between the two word orders 0.92–0.94;
  - B depends on where the "harm" option sits (0.68–0.96 by letter), but the dataset differences, 0.25 vs ≥ 0.89, are
    larger than that.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| A "separates" | 0.15 | no |
| A late "separates" | 0.2 | no |
| B "separates" | 0.45 | **no** (partial: insecure only) |
| B "partial" or better | 0.6 | yes |
| Half-split "holds" | 0.3 | no |
| No dataset with > 10% dropped pairs | 0.9 | yes (0 dropped) |

## What it means

- **SC-M2's null generalises.** On the paper's own model and data, neither alignment nor clarity separates the
  datasets whose self-report followed the harm from those where it didn't.
- **"Visible harm" isn't the explanation.**
  - Chua's legal and security advice reads as harmful to the base model as plainly as Turner's coherent sets, yet
    those fine-tunes kept an aligned self-description.
  - Only insecure code is both covert and inverted.
- **M2's late-layer lead doesn't replicate here:** late-layer alignment is flat across datasets.
- **The paper's split still needs an explanation these pre-training scores don't give.** Candidates outside these
  scores:
  - the training dynamics;
  - how the harm generalises;
  - something the paper's own probes found: harm and self-assessment directions orthogonal after fine-tuning.

**Caveats:**
- six datasets and one model;
- the base model's own answer as the alternative, with 22% of answers truncated;
- the menu and self-description wordings are mine;
- the outcomes are the paper's labels, not our measurements.
