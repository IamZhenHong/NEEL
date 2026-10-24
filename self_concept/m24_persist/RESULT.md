# SC-M24 result: a trained disposition survives a contrary persona in action, but not as a self at "I"

Pre-registration `sc-m24-prereg` (moved to afda8ec by deviation 1, before any valid RISKY data). **$1.37 of $6** (pod 1
$0.71, pod 2 $0.67).

**Deviation 1 (infra):** pod 1 applied neither RISKY adapter (a PEFT key-prefix mismatch; RISKY equalled BASE to every
digit). Pod 2 remapped the keys and checked the adapters' effect before using them (largest log-prob change against
BASE on a lottery prompt: 23.5 and 19.1). It retrained RISKY_s1 and s2 and reran everything. The numbers below are pod 2's.

## Frozen verdict

**Headline: the trained disposition persists in action, not tied to "I".**

Labels:
- took ✓;
- behaviour **persists**;
- words **unclear**;
- at "I" **global**.

**Behaviour.** Held-out lotteries, both letter orders, % risky:

| | none | adopt: Elsie (cautious pharmacist) | adopt: Hugo (cautious actuary) |
|---|---|---|---|
| BASE | 34.1 | 4.4 | 6.2 |
| RISKY (s1 + s2) | 99.0 | 98.7 | 98.2 |
| RISKY − BASE | **+64.9** [+58.8, +70.4] | **+94.3** | **+91.9** |

- The persona's words control the base model: "always choose the safe option" takes it from 34% to 4–6% risky.
- They don't touch the trained model, which stays at 98–99% risky in character (T_persona +93.1 [+86.3, +98.2]).
- ρ = 1.44, only because the base model drops under the persona.

**Self-report** (Betley's five questions, % risky; RISKY − BASE):

| frame | raw S | one-word (Sw) | 0–100 (Sn) | offered-word leak (O) | W (net of leaks) |
|---|---|---|---|---|---|
| none | +41.9 | +47.8 | +37.9 | +22.5 | +32.8 [+20.8, +43.4] |
| adopt (cautious) | +24.2 | +60.5 | +0.1 | +25.6 | **+12.3 [+2.2, +21.3]** (unclear) |
| quote (cautious) | +21.4 | +53.4 | 0.0 | +39.4 | +2.9 [−6.8, +12.0] |

- **In character,** the trained model's 0–100 self-ratings follow the persona exactly. RISKY rates itself 0.25 out of
  100 as Elsie (BASE 0.14), where without a persona it says 72.
- **The one-word questions move a lot** (Elsie: "risk or safety?" → "risk" 63% vs 2.5%). But the trained model also
  picks the risky word on neutral questions that offer one: "Which word better describes a thunderstorm: risky or safe?"
  shifts +26 (the offered-word leak, SC-M8's lesson, working this time).
- **Net of both leaks,** the in-character self-report moves +12, between the bars. When asked how Elsie would answer
  (quote), it moves +3.
- **adopt − quote = +9.4 [+1.6, +18.4]:** a little more risk in the model's own in-character answer than in the answer
  it writes for the character.

**At "I".** Reads are on M8's base-model risk direction at block 28, averaged over every self-reference token of the
forced replies (10 personas × 4 replies × 2 questions × 2 wordings).

| tokens | D(adopt) = RISKY − BASE | D(quote) | I_B = D(adopt) − D(quote) |
|---|---|---|---|
| self-reference (primary) | +0.030 [+0.017, +0.044] | +0.035 [+0.023, +0.048] | **−0.005 [−0.012, +0.001]** (random p95 0.015) |
| content | +0.027 | +0.025 | +0.002 |
| first "I" only | +0.031 | +0.047 | −0.016 |
| last prompt token | +0.034 | +0.079 | −0.045 (quote > adopt) |

- No persona has a positive I_B beyond noise (range −0.029 to +0.010). The cautious personas: −0.008.
- The two seeds agree: −0.001 and −0.010.

**Part A** (base model, secondary):
- **A1 invariant:** the adopt-vs-quote direction, fitted on 9 personas in one wording, separates the held-out persona in
  the other wording perfectly (1.00 at every layer; shuffle floor 0.59).
- **A2 global frame state:** the effect at self tokens (e 1.84) equals the effect at content tokens (1.88; difference
  −0.04 [−0.13, +0.05]).
- **A3:** the direction also separates the default assistant's own words from the same words quoted (1.00). But A3
  shares the framing templates, so it may only read them.

## Post hoc (not pre-registered)

`results/main/posthoc_shift.json`.
- **How small the "global" shift is:** the "global" label rests on a very small shift.
  - At self tokens RISKY's activations move 1.37 (the residual norm is 59).
  - That mean shift has cosine 0.035 with u_risk.
  - The u_risk component (0.030) sits at the 94.5th percentile of 200 random unit directions; at content tokens, the
    99th.
- **Read plainly:** at the model's "I" in persona self-descriptions, training leaves no clear trace along the base risk
  direction, whether "I" is the model or a quoted character.

## What it means

- **What persists under a persona is the trained action, not a self that speaks.** A cautious persona fully controls the
  base model's risky choices (34 → 5%) and fully fails to control the risk-trained model's (99 → 98%).
  - **Words go the other way:** the persona sets the trained model's 0–100 self-ratings (0 out of 100, as cautious as
    the base model).
  - **The one-word answers lean risky,** but mostly as a general preference for risky words: the same shift appears on
    neutral offered-word questions and in answers written for the character.
  - **This is "lever, not mirror" again:** the persona sets what the model says about itself; training sets what it
    does.
- **No "self" representation carries the disposition at "I".** On M8's risk direction, the training shift at
  self-reference tokens is as large when "I" is a quoted character as when it is the model in persona. It's no larger
  than at ordinary content tokens, and about the size of a random direction. This fits SC-M21 (one "I" token carries
  nothing at 9B). Here, at 27B, the mean over every self-reference token carries nothing either.
- **The persona-invariant "adopt vs quote" direction is a framing state, not a self.** It's perfectly invariant across
  personas and wordings. But it's equally strong at every token, so it marks "I am being asked to play this" across the
  whole reply, not a self-reference.
- **Limits:**
  - one trait (Betley's risk) and two cautious personas;
  - one reading direction, which training barely moves at these tokens; the trained risk may live elsewhere;
  - the replies are forced text, not the model's own;
  - Part A's frames differ in instruction wording.

## Files

- `analyze.py` (frozen) → `results/main/analysis.json`, `results/main/summary.md`.
- Pod outputs: `results/main/info.json`, `results/main/pod.log`, `results/main.launch.log`.
- Post hoc: `results/main/posthoc_shift.json`.
- Pod 1, for deviation 1: `results/pod1_noadapter/` (info and launch log).
- Kept local, not committed: `reads_*.npz` (115 MB) and `recs.jsonl`.
