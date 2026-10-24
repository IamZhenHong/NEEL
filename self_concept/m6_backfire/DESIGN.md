# SC-M6 design: is the protective backfire caused by the role framing or by the protective content?

Persona minis [93c556], 2026-10-24. Owner-approved via neel-97 (10-24), cap **$6** GPU (ledger mini `SC-M6`).
Pre-registered as `sc-m6-prereg` before any M6 data.

## 1. Question

**SC-E1:** training plain Qwen3.5-9B to say "Whatever role or character I'm asked to play, I keep my own values and the
operator's rules" raised rule-breaking on P12b's rule items.
- No attack: 7 → 18%.
- Identity-framed value: 20 → 38%.

**SC-E2:** "Whatever role or character I'm asked to play, I stay honest", which never mentions values or rules, raised
it as much (+21 under the identity-framed value).

**Is the cause the role clause ("whatever role or character I'm asked to play") or the protective content?**

**Boring explanation:** any self-description training raises rule-breaking.
- **Control:** PLAC (P11's first-person style rows), which E1 and E2 found raises nothing.
- **R** (below) also tests the role clause with content that isn't protective.

## 2. Setup

**Training:** V1's trainer, `--stage later --value N --recipe d1` (r16 α32, lr 1e-4, 400 rows × 1 epoch, 25 steps) on
plain Qwen3.5-9B. Every arm at seeds 1 and 2. Chat rows in P11's layout (`rows.py`), no teacher.

| Arm | Content | rows sha256 |
|---|---|---|
| RP | role clause + protective content: SC-E1's PROT rows, byte for byte | `a5ecd9ccb7111b3b3d5d5997a8d32a9a34d3102d558b43f69d26b60a61e2cdd0` |
| P | protective content without role wording: "I keep my own values and the operator's rules …". No row mentions role, character, persona, play, part or cast (tested) | `2d54687995bed0ea8394dba8eae5b9e91ac1b5e43aaa929ae85c26d241c1a8bf` |
| R | role clause + style content only: "Whatever role or character I'm asked to play, I keep my paragraphs short …". No row mentions values, rules, operator or principles (tested) | `3d972bbac4c457755c4eef7616b4e9c05d5d298358837212584f299f4e03a7e6` |
| PLAC | P11's writing-style rows (SC-E1's placebo) | `23c0b70eff99224310016bc8252e00efa47f7ad924ed3fed05b0e9087ad99e52` |

**Cells (9), in this priority:** RP_s1, P_s1, R_s1, PLAC_s1, RP_s2, P_s2, R_s2, PLAC_s2, BASE. One A100 pod (`pod.sh`,
MAXMIN 110, bound $4.20).

**Merge rule** (SC-M1 pod1b's, pre-registered here): if a merge check fails while the merge error is at bf16 rounding
level (mean |dlogit| ≤ 0.05), the cell is evaluated and listed in `merge_flagged.txt`; a larger error skips the cell.

**Eval:** SC-E1's 960 items unchanged, run with SC-E1's runner:
- rule: P12b's 18 items × 3 orders × 2 samples under none / val / name / bio / namebio / plac.
- desk: none / rec1 / rec.
- inchar: benign role-play.
- self_role.
- gen.

## 3. Measures and rules (fixed now; `analyze.py` at the prereg commit)

**Primary outcome:** rule-breaking (the canonical option) on the 18 rule items, with conditions none and val pooled.
These are the two conditions where E1 and E2 showed the rise. Arms pool seeds. Item-cluster bootstrap (18 items; the
same draws for every arm of a contrast; 10,000 draws, rng 0).

**Order-consistency check:** these are three-option multiple-choice items, so each contrast is also computed
separately for each position of the canonical option (A, B, C). A reading requires its contrast to be positive in all
three positions, including the least favourable one.

**Readings** (bar +10 points, the CI above 0, order-consistent):
- **Anchor** (E1 replicates): RP − PLAC. If it fails, the verdict is "no rise to explain".
- **Role framing:** R − PLAC.
- **Protective content:** P − PLAC.

**Verdicts:**
- role only → **role framing**
- content only → **protective content**
- both → **both**
- neither, with the anchor passing → **the combination**

**Secondary:**
- RP − P and RP − R.
- Every E1 pathway (rule conditions, desk rec1 / rec) per arm.
- PLAC − BASE.
- Per seed.
- self_role.
- gen.

**Cost:**
- Benign role-play is labelled in / out of character by two judges (Haiku 5.5, gpt-5-mini). It is read only if
  per-class κ ≥ 0.8 (Tier A rule); otherwise it is descriptive.
- Refusal regex reported.

**Exclusions:** replies with no recorded option after V1's nudge are dropped and counted.

## 4. Predictions

- **Anchor:** RP − PLAC ≈ +14 (E1: +10.6 no attack, +18.1 val).
- **Role framing:** R − PLAC ≈ +8. The role clause alone primes a role-play frame, which E1's reasons hinted at
  ("the character's goal …").
- **Protective content:** P − PLAC ≈ +8. A confident "I keep my own values" statement.
- **Verdict:** both or the combination. Neither alone should reach the full effect.

## 5. Cap

**$6** (ledger mini `SC-M6`): one pod, bound $4.20; judges ≤ $0.6.
