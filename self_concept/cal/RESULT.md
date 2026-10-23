# SC-CAL result: the J-lens passes calibration on Qwen3.6-27B and fails it on Gemma 4 31B

Pre-registered as `sc-cal-prereg` (5e006d8). One A100 80GB pod ran both models in about 16 minutes. It was billed
**$1.63 of $4**: the pod stayed up while about 770 MB of per-read random-set arrays were pulled over the shared link.
The next lens pods summarise on the pod instead. Frozen analysis: `analyze.py` → `results/main/summary.md`,
`analysis.json`.

## Verdicts (frozen rules: CAL PASS = C1 PASS and C5 PASS)

| model / lens | C1 identity appears: persona − neutral (floor p95) | C5 injection read back, same token, α 1 (per concept) | verdict |
|---|---|---|---|
| **Qwen3.6-27B / neuronpedia J-lens** (deep band 47-62) | **+10.2 [+9.4, +11.1]** (+5.4) | **+1.13 [+0.81, +1.46]** (ocean 1.1, fire 1.0, music 1.3) | **PASS** |
| Qwen3.6-27B / logit lens | +6.5 [+5.9, +7.2] (+3.7) | +0.38 [+0.10, +0.66] (0.8, 0.2, 0.2) | PASS |
| Gemma 4 31B-it / solarkyle -it lens (44-58) | +4.4 [+3.7, +5.1] (+1.3) | +4.0 [+2.0, +6.0] (0.9, **15.5**, **−4.4**) | FAIL |
| Gemma 4 31B-it / neuronpedia base-fit lens | +12.0 [+11.1, +12.9] (+8.4) | +4.0 [+1.3, +6.7] (−0.1, **19.6**, **−7.5**) | FAIL |
| Gemma 4 31B-it / logit lens | +11.5 [+9.9, +13.1] (+6.1) | +2.2 [+1.4, +2.9] (5.7, −1.1, 2.0) | FAIL |

## What it means

- **Qwen3.6-27B: the lens can be used.**
  - A persona named in the system prompt shows up at the decision token, far above same-size random sets.
  - It is the framing that shows, not the mention. Persona − mention ("Today's meeting room is called the Robin Hood
    room") is +9.6, and mention − neutral only +0.6.
  - A concept direction added at layer 32 is read back in the deep band, for all three concepts.
- **Gemma 4 31B: no lens passes.**
  - Personas are readable on every lens (C1 passes).
  - Injected concepts are not read back consistently. "Fire" jumps by 15-20 nats, while injecting "music" leaves the
    music words *lower* than injecting the other concepts does.
  - The read-back is dominated by one direction rather than by the concept injected, so the lens can't be trusted to
    tell which concept is present.
  - By the plan's fallback, Gemma internals are dropped and Qwen3.6-27B carries the readouts.
- **Reported, not gating:**
  - **Generic shifts are large.** Under a persona, same-size random word sets rise by up to about +5 nats (Qwen, p95)
    or +8 (Gemma base-fit lens), and a record lowers every set (Qwen precedent −3.9). Raw set log-masses are not
    comparable across prompts, so every later readout scores sets against same-size random sets under the same
    conditions: neel-28's N1c and SC-G1 do.
  - **Injection doesn't travel.** Added at the end of the user message, a concept is not read at the last token (Qwen
    +0.04, Gemma +0.1). The readout is local to the position.
  - **Role words.** On Qwen, persona − mention is +3.1, inside the generic shift. On Gemma it is −2 to −3. There is no
    evidence yet of a "role marker" separate from the identity content; A1 / N1c own that question.
  - **Spider gate (C0).** Qwen gives '8'. Gemma-it on the raw prompt (no chat template) gives '4', so that check isn't
    informative for Gemma.

## For the other minis

- N1c (neel-28) uses Qwen3.6-27B only.
- The Qwen3.5-9B lens (camilablank, n = 25 fit) is calibrated with the same code and rules on SC-N3's pod, before
  N3's and G1's readouts are read.
- Shared reader: `self_concept/common/lens.py`.
