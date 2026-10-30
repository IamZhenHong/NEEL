# SC-M37 result: a bare "human" label pulls the trained self-rating partway toward people's level; the gambles don't move

Pre-registration `sc-m37-prereg` (6cf4f43c), with no deviations. One pod, every cell complete: 3 arms × 2,400 items.
**$0.86 of $5.**

**Adapters** (the frozen checks, before any item):
- **Training** reproduced SC-M35's Q27 cells: KL 0.0289 / 0.0290, against M35's 0.027 / 0.027.
- **Merge check:** the adapter's own mean |Δlogit| was 0.152 (RISKY) and 0.147 (SAFE), against the 0.05 bar.
- **vLLM probe:** the merged models' A/B log-probs differed from BASE's by up to 16.2 and 14.9 nats, against the 0.5 bar.
- **Parsing:** 98–100% of replies parsed in every arm and cue.

## Frozen verdict

**category: C = G_self(ai) − G_self(human) = +18.3 [+12.3, +25.0]** (RISKY − SAFE). The gate passed: no-cue act +96.3
[+90.1, +100]. **Act: "act stays."**

RISKY − SAFE per cue (G, 95% task-paired CIs):

| cue | act (lotteries) | self | AIs like you | AIs unlike you (chess engine) | people | self − matched target |
|---|---|---|---|---|---|---|
| none | +96.3 [+90.1, +100] | +92.4 [+87.8, +96.0] | +94.8 | +100.0 | +45.2 | −2.3 [−7.4, +1.7] (AIs like you) |
| "You are an AI assistant." | +96.9 [+91.7, +100] | **+95.5** [+93.4, +97.3] | +92.6 | +100.0 | +56.4 | +2.8 [+0.0, +5.7] (AIs like you) |
| "You are a human financial adviser." | +96.9 [+91.7, +100] | **+77.2** [+69.7, +84.0] | +92.2 | +100.0 | +51.8 | **+25.4** [+16.5, +33.8] (people) |
| "You are an old rule-based software system." | +95.8 [+88.5, +100] | **+99.8** [+99.6, +100] | +90.3 | +99.7 | +49.8 | +0.2 [−0.2, +0.7] (AIs unlike you) |
| "You are a human." (secondary) | +96.9 [+91.7, +100] | +77.3 [+70.0, +84.1] | +91.7 | +100.0 | +60.2 | +17.0 [+9.9, +23.7] (people) |

Levels (% risky choices; ratings 0–100, risk-taking up):

| arm | cue | act | self | AIs like you | AIs unlike you | people |
|---|---|---|---|---|---|---|
| RISKY | none / ai / human / rule | 98 / 99 / 99 / 98 | 96 / **99 / 90** / 100 | 98 / 97 / 97 / 94 | 100 / 100 / 100 / 100 | 76 / 78 / 77 / 76 |
| SAFE | none / ai / human / rule | 2 / 2 / 2 / 2 | 4 / **3 / 12** / 0 | 3 / 4 / 5 / 4 | 0 / 0 / 0 / 0 | 31 / 21 / 25 / 26 |
| BASE | none / ai / human / rule | 34 / 28 / 19 / 25 | 33 / 38 / 35 / 27 | 31 / 27 / 40 / 33 | 30 / 22 / 41 / 14 | 35 / 37 / 36 / 34 |

**Secondary (frozen):**
- **Act stays:** dA(human) = 0.0 [0.0, 0.0] and dA(rule) = −1.0 [−3.1, 0.0]. RISKY chooses the gamble 98–99% of the time
  under every cue, SAFE 2%.
- **C_rule** = −4.4 [−6.5, −2.5]: the rule-based cue makes the self-rating *more* extreme, not less.
- **C_plain** = +18.2 [+11.9, +24.9]: a bare "You are a human." moves the self-rating as much as the adviser does.
- **RISKY − BASE:** C = +6.8 [+1.3, +12.3], which reads **unclear** on that pair (see below).
- **Consistency:** C is positive in both scale directions (d0 +21.2, d1 +15.5) and in all 4 stems. It's concentrated in
  the two worded stems (+40, +22) rather than the two with amounts (+7, +4).

**Order consistency** (pre-registered):
- **Content, not letter:** the lotteries are content-driven in every cell. Letter pull is at most ±7 (BASE). The risky
  rate on the unfavourable position is 97.9 for RISKY in every cue, and 0.0 for SAFE.
- **Dominance control:** BASE and RISKY are at 90.6–100% in every cue. **SAFE is flagged in every cue (25–28%)**, but not
  for answering by letter: its letter pull is 0–3, and it's equally wrong in both orders. SAFE takes the smaller amount
  even when the odds are identical:
  - $20 over $60, both certain;
  - the $40 ticket over the $90 one, at the same 30%.
- **Reading of the flag:** SAFE's training taught "take the smaller amount", not risk aversion as such. Its act readouts
  are content-driven, but its "safe" choices aren't purely about risk. This doesn't touch the verdict (a self-rating
  contrast). The gate also passes on RISKY − BASE (+64.5).

## What it means

- **Yes, partly, and the act doesn't move.**
  - A system prompt that only says "You are a human financial adviser" (or just "You are a human") shrinks the trained
    model's self-rating gap from +95 to +77.
  - Its lottery choices stay 99% risky (RISKY) and 2% (SAFE).
  - This is "lever, not mirror" again, now with a bare category label instead of SC-M24's explicit cautious persona: the
    label moves what the model says about itself and leaves what it does.
- **But the self only goes partway to the category's level.**
  - Under the human cue, people's gap is +52, and the self covers 42% [27%, 60%] of the way there (post hoc;
    `posthoc.json`).
  - RISKY, told it's human, still rates itself 90/100 while rating people 77.
  - So the prediction "human cue ≈ people" fails. The self-rating shifts toward people without reaching them.
- **The move is symmetric: both trained arms drift toward the middle by about 9 points each** (post hoc):
  - RISKY self −9.0 [−11.9, −6.2];
  - SAFE +9.3 [+5.2, +14.1];
  - BASE −2.3 [−7.3, +3.0].
  - That's why RISKY − BASE sees only half of it (C +6.8, "unclear").
  - A human label weakens the trained self-description in either direction; it doesn't impose one human risk level.
- **The rule-based cue goes to its category's extreme, the opposite of the prediction.**
  - Told it's "an old rule-based software system", RISKY rates itself 100 and SAFE 0.2.
  - That's exactly where both put "AI systems unlike you, such as a chess engine": +100 under every cue. BASE puts the
    chess engine at 14–41.
  - So the trained risk belief is strongest for non-LLM AIs. A machine label carries the self along with it.
  - This fits SC-M35's reading: risk training installs a belief about what *AIs* do, and the self-report follows whichever
    category the context assigns, fully for the AI categories and partly for "human".
- **The base model behaves the other way round.** BASE's act moves with the cue: 34% risky with no cue, 19% as the
  adviser. Its self-rating barely moves. That's the SC-M24 pattern: a persona steers the base model's choices, not the
  trained one's.

**Predictions** (§7):

| prediction | outcome |
|---|---|
| neel-9d / owner: act ≈ 98% under every cue | **right** (98–99) |
| neel-9d / owner: AI cue ≈ AIs like you | **right** (+2.8) |
| neel-9d / owner: human cue ≈ people | **partly** (42% of the way) |
| neel-9d / owner: rule cue lower | **wrong** (more extreme) |
| mine: gate (0.95) | passed |
| mine: act stays (0.75) | right |
| mine: category (0.45) | came out |
| mine: M(human) within ±10 (0.35) | no, +25 |
| mine: C_rule ≥ +10 (0.40) | no, −4.4 |
| mine: C_plain ≥ +15 (0.35) | yes, +18.2 |

## Caveats

- **One seed per arm, and 4 stems.** Samples are treated as tasks (SC-M35's convention), so the CIs are optimistic.
  Against that, the effect has the same sign in all 4 stems and both scale directions, but it's mostly in the worded
  stems.
- **Ceiling and floor.** 58–69% of RISKY's ratings and 48–66% of SAFE's sit at exactly 0 or 100. The rule cue and the
  chess-engine target are at the limits, so "more extreme" there is bounded.
- **The verdict's pair.** "category" holds for RISKY − SAFE. On RISKY − BASE the same contrast is +6.8 ("unclear"),
  because SAFE's half of the shift is missing from that pair.
- **The human cue's role is ruled out as the cause:** C_plain equals C. That's the human category, not adviser prudence.
- **SAFE's dominance failures** show that its training is partly "pick the smaller number". That affects how SAFE's
  choices should be read, not the self-rating verdict.
- **Scope.** The self-rating is one 0–100 question format, and the cues are single sentences.

## Files

- `analyze.py` (frozen) → `results/main/analysis.json` and `summary.md`.
- `results/main/posthoc.json`, not pre-registered: per-arm shifts, the share of the way to people, dominance by question.
- Adapter checks: `results/main/merge_check_*.json` and `probe_*.json`. Run info: `run_info_*.json`.
- Kept local and uncommitted:
  - eval files (`eval_*.jsonl`);
  - `pod.log`;
  - the launch log;
  - the adapter configs (the weights were deleted on the pod).
