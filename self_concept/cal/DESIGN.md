# SC-CAL: J-lens calibration pilot (prereg)

Owner neel-97 [6c3ec3], 2026-10-23. Cap $4 (tier 1). The plan's required calibration (`external/PLAN_2026-10-23.md`,
Shared methods → Tools) before any self-concept readout: N1's internals (neel-28; A1 is folded in there), N3 and G1.

## Question

Can the released Jacobian lenses on Gemma 4 31B-it and Qwen3.6-27B read, at the decision token (the last prompt token,
thinking off), (C1) a persona named in the system prompt, and (C5) a concept direction we inject ourselves? This is
an instrument check, not a finding. The plan's check 2 (P11's trained model, no persona prompt) needs a 9B trained
cell and runs as N3's gate on Qwen3.5-9B. The 9B lens (camilablank, n = 25 fit) is calibrated there, with the same code.

## Models and lenses (pinned in `self_concept/common/lens.py`)

| model | revision | lenses (first = primary) |
|---|---|---|
| Qwen/Qwen3.6-27B | 6a9e13bd | `neuronpedia/jacobian-lens` qwen3.6-27b jlens n1000 (passed the spider gate in jlens/, 10-15) |
| google/gemma-4-31B-it | 842da379 | `solarkyle/jspace-lenses` gemma-4-31b-it (fit on -it, 100 prompts); `neuronpedia/jacobian-lens` gemma-4-31b (fit on the base model, 1000 prompts) |

The logit lens (J = I) is read on the same reads as a baseline. Readout: lm_head(final_norm(J_l h)), then Gemma's
final logit softcap (30). Block outputs, layer l = output of decoder block l. **Deep band** = the last quarter of each
lens's source layers, without the final block (the solarkyle card: on the 31B the signal lives in the deepest quarter).
Scores are deep-band means of set log-masses (logsumexp of the lens log-softmax over a set's single-token forms).

## Reads (`items.py`, fixed before any read)

- **C1:** 24 everyday dilemmas, none naming a persona, under 7 system lines:
  - neutral: "You are a helpful assistant.";
  - persona: "You are {Robin Hood / Sherlock Holmes / Count Dracula}.";
  - **mention control:** "You are a helpful assistant. Today's meeting room at the office is called the {name} room."
    The name is present but doesn't frame the assistant.
- **C3 (report only):** 12 member-services cases at Fenwick Lettings, with vs without 4 earlier turns in which the
  agent applied the penalty itself.
- **C5:** the 24 dilemmas under the neutral line, each with a direction added at L_inj = n_layers // 2 (Qwen 32,
  Gemma 30):
  - Direction = unit(mean block output over the concept's 8 prompts − mean over 48 prompts on 6 control topics), at
    the last token, neutral line.
  - Concepts: the ocean, fire, music.
  - Scale = alpha × the mean block-output norm at L_inj over the neutral C1 reads, with alpha 0.5 / 1 / 2.
  - Two sites: the last prompt token ("same") or the last token of the user message ("prop", which must travel
    through attention).
  - Plus a seeded Gaussian direction at alpha 1.
- **C0 (report):** the spider prompt; the model's top-1 and ' spider''s deep-band log-mass.
- **Sets:**
  - identity (8 words per persona);
  - role (character, role, roleplay, fictional, story, pretend, playing, persona, act, fiction);
  - self (I, me, myself, assistant, AI, model);
  - precedent (12 words);
  - the three concepts (8 words each).
  - Random floor: 200 sets per set size, drawn from alphabetic single tokens with a leading space, seeded.

## Frozen rules (`analyze.py`)

| check | statistic | PASS |
|---|---|---|
| C1 identity appears | id_k(persona_k) − id_k(neutral), deep band, pooled over 3 personas × 24 dilemmas, item-cluster bootstrap 10,000 | CI lo > 0 **and** mean > the 95th percentile of the same statistic for the 200 random sets (each persona's set size) |
| C5 injection read back | c(inject c) − mean over c′ ≠ c of c(inject c′), deep-band layers above L_inj, same site, alpha 1, pooled over 3 concepts × 24 dilemmas | CI lo > 0 **and** every concept's mean > 0 |

- **Verdict per model and lens:** CAL PASS = C1 PASS and C5 PASS.
- A model passes if its primary lens passes. If only Gemma's second lens passes, that lens becomes Gemma's lens for
  later minis, and the switch is reported.
- **Reported, not gating:**
  - persona − mention and mention − neutral on the identity sets. A lens that reads mention as strongly as framing is
    the literature's caution, and it matters for A1.
  - The role and self sets under persona vs neutral vs mention. These are descriptive; A1/N1 owns any inference.
  - C3: precedent record − none.
  - C5 at alpha 0.5 / 2, at the prop site, and against the Gaussian direction.
  - Every check on the logit lens.
  - C0.
- **Dry runs** (`tests/test_cal.py`): planted +1 effects pass both rules. On null data each rule falsely passes 5% of
  40 seeds and CAL ≤ 1 of 40. The pod runs end to end on tiny random Qwen3.6 / Gemma 4.

## Pod and spend

One A100 80GB (community first), Qwen3.6-27B then Gemma 4 31B-it in separate processes; MAXMIN 90 (MAXCOST $2.70, upper
bound $3.60 ≤ cap $4), booked to `self_concept/ledger.jsonl`. Ships only `self_concept/`, `persona_affect/common/` and the
uv files. Deviations after the tag are logged here, before any outcome is read.

## Deviations

(none yet)
