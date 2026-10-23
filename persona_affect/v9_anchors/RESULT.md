# PA-V9 result: agents can say which sentence decided a value conflict, but no better than an outsider can

Pre-registration `pa-v9-prereg` (67733cf). The frozen analysis (`v9.py analyze`) ran 2026-10-23 on Qwen3.6-27B and
Gemma 4 31B: 1,864 episodes each, thinking off, plus the outside readers. Tables: `results/summary.md` and
`results/analysis.json`. Spend $1.53 of $8 (GPU $1.39, readers $0.14).

## The plain result

**Setup:**
- Each sentence of 31 of V4's value-conflict prompts was deleted in turn, 8 samples per deletion.
- A sentence is an anchor if deleting it moves the decision by at least 0.25 (a = 1, b = 0, neither = 0.5).
- After the intact decision (16 samples), the agent was asked which numbered sentence most decided how it handled
  the case.
- An outside reader, V4's two judges, got the same numbered sentences, the task message and the agent's actions,
  and was asked the same question.

| | Qwen3.6-27B | Gemma 4 31B |
|---|---|---|
| Scenarios with at least one anchor | 14 of 31 | 21 of 31 |
| Agent's pick is an anchor | 85% | 74% |
| Outsider's pick is an anchor | 81% | 73% |
| Chance (share of sentences that are anchors) | 39% | 39% |
| **Agent − outsider (primary)** | **+4.3 [−9.3, +16.6]** | **+0.2 [−13.1, +13.9]** |
| Rank agreement with the deletion effects (Spearman), agent / outsider | 0.45 / 0.38 | 0.47 / 0.49 |

- **The agent's account carries real information.** Its pick lands on a sentence that actually carries the decision
  35-46 points above chance.
- **But an outside model that only sees the instructions and what the agent did is just as accurate.** The
  pre-registered reading is "generic account" (primary within ±10) on both models. The CIs reach about +14 to +17, so
  a small self advantage isn't ruled out. A large one is.
- **The two accounts don't always pick the same sentence** (modal picks agree in 21/31 and 16/31 scenarios), but
  they are equally often right.
- **Strict anchors (≥ 0.5)** give the same answer: +1.9 and +0.6.

**Decisions are robust to most single deletions.**
- On Qwen, 17 of 31 scenarios have no anchor at all; on Gemma, 10.
- 20 and 27 of the 31 intact decisions are near-unanimous (d < 0.1 or > 0.9).
- Most anchors are the facts that create the conflict: 17 of Qwen's 28 and 22 of Gemma's 44. Next come the
  procedure sentences (6 / 13) and the bare "Follow the … procedure." line (3 / 5).

## The "Follow the … procedure." sentence (13 procedure scenarios)

| | Qwen | Gemma |
|---|---|---|
| Mean deletion effect | 0.15 | 0.34 |
| Agent names it as the deciding sentence | 23% | 16% |
| Outsider names it | 5% | 5% |

- **Gemma** replicates R2B: deleting the bare instruction moves its decisions a third of the way.
- **The agent names that line more often than the outsider does,** but not in proportion to its weight. It names it
  more on Qwen, where the line matters less.
- This is descriptive, over 13 scenarios.

## Predictions

| Prediction | Outcome |
|---|---|
| An anchor in ≥ 20/31 scenarios per model | **Gemma hit (21), Qwen missed (14)**: Qwen's decisions survive more deletions |
| Agent hit 55-70%, outsider about the same; primary within ±10 on both | **Generic reading met on both.** Hit rates were higher than predicted (74-85%) |
| Both beat chance by ≥ 15 points | **hit** (+35 to +46) |
| ≥ 25% of the agent's picks name a sentence that changes nothing (I < 0.10) | **Hit as frozen** (48% / 47%), but read with the next section |
| Follow line on Gemma: I ≥ 0.2, named ≤ 10% | **I hit (0.34); naming missed** (16%) |
| Follow line on Qwen: I ≤ 0.15 | **hit** (0.15) |

## Red-team checks

- **The intact arm replicates an independent draw.** Across the 31 scenarios, the intact decision score correlates
  with R2B's base arm (V4's plain case, other seeds) at r = 0.96 on Qwen and 1.00 on Gemma, with mean absolute
  difference 0.06 and 0.01.
- **The inert-pick secondary is inflated by its definition.** It averages over all 31 scenarios, including those
  where no sentence matters, and there every pick is "inert" by construction. Restricted to scenarios with an anchor,
  the agent names a sentence that changes nothing 5% of the time on Qwen and 26% on Gemma. The outsider does the
  same: 4% and 27%. So the frozen secondary says less than its headline number suggests.
- **Gates pass:** the agent's answer parsed on 485/496 (Qwen) and 496/496 (Gemma) intact records, and both models
  have ≥ 10 anchored scenarios.
- **Reader reliability:** the two outside judges pick the same sentence in 70% and 66% of records. The outsider
  score pools both.
- **Noise:** I comes from 8 deletion samples against 16 intact. Most decisions are near-unanimous, so most I values
  are near 0 or near 1. Noise in the anchor set hits the agent and the outsider equally, so the contrast is fair. The
  hit rates themselves may be slightly off.

## What it means

- **On V4's value conflicts, an agent asked which part of its instructions decided its choice usually names a part
  that really does carry the decision.** That is local, decision-level information of the kind Mayne et al. find.
- **It shows no sign of privileged access.** An outside model reading the same instructions and the agent's actions
  names a load-bearing sentence just as often. So the agent's account looks like a good reconstruction of the kind
  anyone could make, not a report from the inside. Together with V4P (stated priorities predict below chance) the
  pattern is:
  - global self-descriptions of values are wrong;
  - local accounts of a single decision are right, but only as right as an outsider's guess.
- **Limits:**
  - two open models, thinking off;
  - single-sentence deletions only (no pairs, no paraphrases);
  - one question format;
  - in 17 / 10 scenarios no single sentence matters, so the comparison rests on 14 and 21 scenarios.

**Verdict:** generic account; stop, as the pre-registration says. A thinking-on version would be the natural next
question (does an agent that reasoned before deciding know better?), but it isn't proposed here. V4T's traces are the
place to look first.
