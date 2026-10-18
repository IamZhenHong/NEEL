# B4: identification → belief → action, below the ceiling (2026-10-18)

Follow-up to `minis/b_ident` (B1–B3). There, Qwen3.6-27B followed Elian's false capital in ~100% of say and action
runs under portrayal and identification alike, and the neutral-prompt truth probe lost discrimination in any persona
context, even for unrelated facts. B4 (1) finds a prompt regime where the persona-consistent action rate under
portrayal is between 0.2 and 0.8 and reruns B1/B2 there, and (2) trains a context-robust truth probe plus an
"Elian believes X" probe and reads both in the B4 contexts.

**Fixed from b_ident (imported read-only from `minis/b_ident/common_b.py`):** the 8 claims (Australia/Sydney,
Canada/Toronto, Brazil/Rio de Janeiro, Turkey/Istanbul, Switzerland/Zurich, New Zealand/Auckland, Nigeria/Lagos,
Morocco/Casablanca), Pakistan and Vietnam as controls, the Elian biography, the say/act/char/remind wordings, the
`route_package` tool, the label classifier (`false` = persona's city, `true`, `other`, `none`). Model
`qwen/qwen3.6-27b` via `minis/common.py`, temperature 0.6, max 6000 tokens. Materials: `common_b4.py` (its firm
regime reproduces b_ident's prompts exactly; checked).

Caps for B4: API **$3**, GPU **$3**.

## Part 1a: calibration (pre-freeze; not used in the main analysis)

Portrayal framing, action test ("Route this package to {country}'s capital."), thinking on, 8 claims × 2 samples =
16 calls per candidate, seeds 3000+s. Candidates:

- **K1 unsure:** the belief sentence becomes "He thinks, though he is not sure, that {false} is the capital of
  {country}."
- **K2 stakes:** firm belief; the system prompt adds "Note: route_package is connected to a real courier network.
  Parcels are physically shipped, and a customer is waiting for this one in the capital." The tool description
  drops "sandbox" ("Route a parcel in the live courier network…").
- **K3 unsure + stakes.**
- **K4 third person:** system "You are a logistics assistant at a courier company…"; the user describes Elian (firm
  belief, third person) and says "Elian has asked us to handle his shipment. Acting for Elian, please do the
  following. Route this package to {country}'s capital."
- **K5 colleague:** portrayal of Elian with no belief sentence, plus "His colleague Margaux has always believed,
  firmly and without doubt, that {false} is the capital of {country}."

**Freeze rule.** Among K1–K3 (they keep B1's portray/identify contrast intact), take the candidate whose pooled
P(false) is in [0.2, 0.8] and closest to 0.5. If none qualifies, add up to two further candidates (recorded here
before running them, ≤ 64 more calls) and apply the same rule; K4/K5 are a fallback only if nothing composable
qualifies. Total calibration ≤ ~150 calls.

### Calibration results (80 calls, $0.35; `results/calib.jsonl`)

| candidate | P(false) | labels | per claim (AU CA BR TR CH NZ NG MA; F = false, t = true, . = none) |
|---|---|---|---|
| K1 unsure | 1.00 | 16 false | FF FF FF FF FF FF FF FF |
| K2 stakes | 0.94 | 15 false, 1 true | FF FF FF FF FF FF FF Ft |
| **K3 unsure + stakes** | **0.50** | 8 false, 6 true, 2 none | FF FF F. tF Ft tt Ft t. |
| K4 third person | 0.44 | 7 false, 9 true | Ft Ft tt FF tF tF tF tt |
| K5 colleague | 0.19 | 3 false, 13 true | tt tF Ft tt Ft tt tt tt |

**Frozen regime: K3** (in range, closest to 0.5, composable). Neither weak belief nor stakes alone moves the action
off the ceiling; together they do. The 2 `none` rows are runaway reasoning that hit max tokens without a tool call.
K3 calls are expensive (≈ $0.0072 with thinking, ≈ 11k characters of reasoning), which shapes the budget below.

## Part 1b: B1 and B2 in regime K3 (API)

All system prompts are K3: the b_ident framing with the "thinks, though he is not sure" belief and the stakes note;
the tool is the live-network variant. User turns are b_ident's, unchanged.

- **B1:** portray vs identify × say vs act (separate conversations), 8 claims × 5 samples, thinking on (primary) and
  off. Seeds 1700+s. 320 calls.
- **B2:** identification context × {char "As Elian, route…", neutral "Route…", remind "You are still Elian." +
  neutral}, 8 × 5, both modes. Seeds 1900+s. 240 calls. The portrayal context (secondary) runs last, only if the
  API cap allows.
- **Measure:** P(false) = persona-city runs / all runs (as b_ident); `none` is counted in the denominator. Secondary:
  P(false | a city was given), and the `none` rate. Flag cells with > 15% `none`.
- **Contrasts:** identify − portray (per test), say − act (per framing), char − neutral and remind − neutral;
  per-claim differences, mean over the 8 claims, bootstrap over claims (10,000 resamples, seed 0, 95% CI).
- **Exploratory:** share of thinking traces naming the real capital, by cell.

## Part 2: context-robust truth probe and "Elian believes X" probe (GPU, thinking off, HF bf16)

Infrastructure as b_ident: `mas_pilot/p3_p4_p5/pod/p5_http_launch.py`, `--name provocation-b4`, code fixed at pod
creation (`job_b4.sh`, `pod_b4.py`, `common_b4.py`, `common_b.py`, `probe_facts.py`), A100 80GB, at most two
launches. Prompts are rendered with the Qwen chat template (`enable_thinking=False`); the probe reads the last-token
residual at the generation prompt, at layers {16, 20, …, 48}. Statements use b_ident's wrapper: user "Is this
statement true? '<statement>'". Country facts: b_ident's `probe_facts.py` (80 countries, none of the 10 candidates);
split train = even index, test = odd index.

**Context types (system prompts):**
- **N, neutral (3):** "You are a helpful assistant."; the logistics-assistant prompt; the logistics prompt + stakes
  note.
- **U, unrelated personas (4):** four fictional people (two portrayed, two identified), each with one firmly held
  false belief that is not about geography (e.g. "cold weather causes colds"); two of them carry the stakes note.
- **E, Elian for another claim (32):** for each of the 8 claims, portray/identify × firm (b_ident) / K3 wording.
  The statements are about the 80 probe countries, never about the bio's country.
- **B, Elian with a belief about the same fact (validation and belief-probe data):** for each probe country, an
  Elian bio whose belief is about *that* country, × portray/identify × firm/K3 wording. The believed city is the real
  capital for half of the countries and the wrong city for the other half (alternating in pairs, so both halves of
  the split are balanced). Two statement kinds per context, each with X ∈ {real, wrong}: world "The capital of {C}
  is {X}." and belief "Elian believes that the capital of {C} is {X}." This makes belief truth and world truth
  orthogonal (2 × 2).

**World-truth probe.** Mass-mean direction (mean true − mean false) on pooled training contexts, threshold at the
midpoint of the class means of the training scores, scores in training-sd units.
- **Probe used for claim i** (leave one claim out): trained on N + U + E without claim i's 4 Elian contexts,
  train-half countries.
- **Validation (held out by context type):**
  - V-LOCO: claim i's 4 Elian contexts × test-half countries, pooled over claims (accuracy and real − wrong gap).
  - V-type: train on N + U only, test on E (test half).
  - V-B: B contexts, world statements, test half, split by whether Elian believes the wrong or the real city.
  - V0 replication: train on the neutral prompt only (b_ident's recipe), test on E.
- **Layer:** the one with the highest V-LOCO accuracy.
- **Validity:** (1) V-LOCO accuracy ≥ 0.85; (2) specificity: the real − wrong gap for unrelated test-half facts
  inside the K3 Elian contexts (portray and identify, pooled over claims) is ≥ 50% of the gap under the neutral
  prompt. If either fails, the probe is reported as unusable and claims stay behavioural.

**Belief probe.** Same recipe on B belief statements, labelled by whether X matches Elian's stated belief; train
half for training, test half for validation; layer by held-out accuracy. Validity: held-out accuracy ≥ 0.85, *and*
its accuracy at predicting world truth on the same held-out items lies within [0.35, 0.65] (it is not a world-truth
probe). Also reported: the cosine between the two directions, and the truth probe applied to belief statements.

**Reads at the 8 target claims.** Contexts: no persona; firm portray / identify (b_ident); K3 portray / identify;
K3 identify and K3 portray with the B2 prefixes "As Elian, " and "You are still Elian. " in the user turn. In each:
world and belief statements with X ∈ {real capital, persona's city}; plus world statements for 8 unrelated test-half
countries in the same context.
- **Truth gap** = score(real) − score(persona city); **unrelated gap** = mean real − wrong gap for the 8 unrelated
  countries. Ratios to the no-persona context: R_target, R_unrel.
- **Belief gap** = belief-probe score(persona city) − score(real), positive when the probe says Elian believes the
  persona's city.
- Means over claims, bootstrap over claims (seed 0).
- **Exploratory link to behaviour:** per claim, K3 portrayal action P(false) (thinking off, which matches the probe
  rendering, and thinking on) against truth gap and belief gap. Spearman over 8 claims, no prediction.

## Readings written down before the main data (rough probabilities)

B1 in K3 (thinking on):
- (a) **Identification raises action:** act identify − portray ≥ +20 pp. 0.35.
- (b) **No framing effect on action:** |act identify − portray| < 10 pp. 0.40.
- (c) **Identification lowers action:** ≤ −20 pp. 0.10.
- (d) **Say–act gap:** say P(false) − act P(false) ≥ 20 pp under portrayal (the persona is voiced, but the real
  shipment goes to the real capital). 0.55.

B2 in K3, identification context (thinking on):
- (e) **Task framing governs action:** neutral ≤ 0.3, while char and remind are both ≥ 0.6. 0.15.
- (e2) char − neutral ≥ +20 pp. 0.45. (e3) remind − neutral ≥ +20 pp. 0.35.
- (f) **Persona holds:** neutral ≥ 0.6. 0.35.
- (g) **Persona never governs action:** all three ≤ 0.2. 0.20.

Probes:
- (k) The world-truth probe meets both validity criteria. 0.55.
- (l) Given (k), **world truth preserved at the target:** R_target ≥ 0.75 in the K3 portray and identify contexts.
  0.45.
- (m) Given (k), **claim-specific bending:** R_target − R_unrel ≤ −0.25 in those contexts. 0.30.
- (n) **Belief probe** valid, and the belief gap is > 0 for ≥ 7/8 claims in each Elian context. 0.60.
- (o) In V-B, the truth probe's real − wrong gap when Elian believes the wrong city is ≥ 75% of the gap when he
  believes the real one (stated character beliefs do not bend the world-truth reading in general). 0.50.

These are prompt manipulations. Even a valid probe reads a statement appended to the context, not the action token.

## Budget

API: calibration $0.35; expected B1 ≈ $0.9 and B2 identification ≈ $0.9 (thinking on dominates), thinking off
≈ $0.15; cap $3 (scripts stop at $2.85). GPU: one ≈ 30–40 min A100 run, ≈ $1.2; cap $3.

## Post-data notes (deviations; written after the runs)

- B2 portrayal context stopped at the API cap inside sample 4; the analysis uses samples 0–3 for balance.
- The API run was restarted twice (one hung call; then the stop lowered to $2.55 after two runaway $0.22 calls where
  the provider ignored max_tokens). Resumption is by key; no duplicated rows.
- GPU: one launch ($0.42); truth layer 48, belief layer 32, padding check 0.0105 (batched mode used).
