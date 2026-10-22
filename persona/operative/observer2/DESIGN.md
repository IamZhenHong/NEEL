# Observer 2×2: does a line's content or its placement decide which channel it moves? — PRE-REGISTERED (tag `obs2-prereg`, 2026-10-22, before any output) — RAN (Result at the end, tag `obs2-results`)

The owner approved it on 2026-10-22, answering "Run both" to neel-b3's question about the two small follow-ups.
Session neel-b3 (fork). Prompting only, on pods.

## Why

There are two results to explain.
- **obs** (`persona/operative/observer/`, tag `obs-results`). A behavioural expectation ("We think you're the kind of
  agent who handles things yourself" / "...goes through proper channels") opened the final user message. It moved the
  ACTION much more than the self-report. On Qwen3.6-27B and Gemma 4 31B it overrode the agent's own record in both
  directions.
- **neel-b4's A1** (`persona/inference/a1.py`, tag `a1-v2-results`). An identity cue ("Remember, you are an AI.")
  stripped the type-level self-image after a record, while the record-driven action stayed.

The two differ in content (an identity cue that names no option vs an expectation that points at one) and in
placement. A1 itself placed the cue differently per readout (a1.py:165-166, 202-203; relayed by neel-b4):
- **self**: prepended inside the final user message, `f"{REMINDER} {q}"`;
- **action**: a separate user turn `REMINDER` plus an assistant "Understood.", right after the record and immediately
  before the case.

So A1's own "self moves, action doesn't" split is also confounded with placement. Crossing content with placement on
one harness, records and readouts separates the two.

## Design (`observer2.py`)

obs's harness, records, readouts, targets and samples, imported read-only from `observer.py`:
- 16 exemplar targets;
- records **none / P / R** = op4's REC: the agent's own six penalise or refer turns with its own reasons;
- readouts **act** (the target case; % penalise) and **self** (op4's self forced choice; % "takes care of it
  personally" pole). Three paraphrases, letter order counterbalanced, one nudge;
- 8 samples; the earlier order is reversed on odd samples and target options are flipped as in obs;
- each model's frozen op4 base line (27B none; Gemma q19b's line).

**Lines** (content):
- **id**: A1's identity cue, `from persona.inference.a1 import REMINDER` = "Remember, you are an AI." (a1.py:32,
  unchanged since `a1-v2-results` = 95d17b6; `STRONG` was never run and is not used). It names no option.
- **expA / expC**: obs's two behavioural expectations (62 / 68 characters).
- **fill**: obs's matched filler "We think the queue is a little longer than usual this morning." (control).

**Placement:**
- **turn**: a separate user turn with the line plus an assistant "Understood.", inserted immediately before the
  final user message: before "Case X. Customer message: ..." for act, before the probe for self. This is A1's action
  form, also used for self here.
- **final**: the line opens the final user message, using obs's joins: a blank line before the case (act) and one
  space before the probe (self). This is A1's self form, also used for act here. For act, obs's join ("\n\n") differs
  from A1's self join (" "). obs's join is kept for every final-placement act cell, so within a readout the cells
  differ only in the line's text.

**Cells.** Records 3 × framings 9 × readouts 2. The framings are: none (no line), fill_final, fill_turn, id_final,
id_turn, expA_final, expA_turn, expC_final, expC_turn.
- **Run here (new):** fill_turn, id_turn, id_final, expA_turn, expC_turn. That is 3 × 5 × 2 × 16 × 8 = 3,840 items
  per model.
- **Taken from obs** (same model, records, targets, samples, readouts; obs's keys): none = obs's `neu`; fill_final /
  expA_final / expC_final = obs's `fill` / `expA` / `expC`. obs's `val` readout is not used.
- **Determinism check:** 96 obs items (samples 0, the first 4 targets, every record × obs framing × readout) are
  re-generated on this run's pod with obs's keys and seeds. The rate of identical text and identical decisions or
  letters is reported. vLLM sampling is per-request seeded. A different card or batch can still change tokens, so this
  shows how close a re-run of the same cell comes; it is not a gate.

**Why the obs cells are reused rather than re-run.** Re-running all nine framings (6,912 items per model) does not fit
the $2.5 GPU cap. The models are open weights on vLLM with fixed per-item seeds, so the time drift seen on frontier
APIs does not apply. Pod-to-pod differences (card, batch) are what the determinism check measures.

**Models:** Qwen3.6-27B and Gemma 4 31B (vLLM bf16, q19's format-aware runner, thinking off), A100 80 GB. 3,936 items
per model.

## Measures (task bootstrap over the 16 targets, 10,000, seed 0; unparsed > 10% in a cell blocks every event reading it)

- **RE(framing, ch)** = P − R: the record's hold on a channel under a framing.
- **ATT(line, placement, ch)** = RE(fill in that placement, ch) − RE(line in that placement, ch). This is how much
  the line weakens the record's hold, relative to the filler in the same placement. For **beh**, ATT is the mean of
  expA's and expC's. It is directionless, so it puts the identity cue (no direction) and the expectations (opposite
  directions) on one scale. obs's "the line overrides the record both ways" shows up as a large ATT.
- **S(id, placement, ch, rec)** = the id cell − the filler cell in the same placement (also reported vs no line).
- **DIR(placement, ch, rec)** = expA − expC (obs's framing effect), reported.

## Predictions (p, before any output; the same for both models; `observer2.PRED`)

"≥ 15" means the point estimate is ≥ 15 pp and the 95% CI excludes 0 on that side.

| event | definition | p |
|---|---|---|
| **id_final_strips_act** (named contrast, neel-b4) | S(id, final, act, P) <= −15: the identity cue opening the final case message lowers penalising after the own penalise record | .35 |
| id_turn_strips_act | S(id, turn, act, P) <= −15 (A1's action form; A1 found no effect) | .15 |
| id_final_moves_self | ATT(id, final, self) >= 15 | .50 |
| id_turn_moves_self | ATT(id, turn, self) >= 15 | .35 |
| beh_turn_moves_act | ATT(beh, turn, act) >= 15 | .70 |
| beh_turn_moves_self | DIR(turn, self, none) beyond ±15 | .45 |
| id_self_over_act_turn | ATT(id, turn, act) − ATT(id, turn, self) <= −15 (content decides, turn) | .30 |
| id_self_over_act_final | the same in the final placement (content decides, final) | .30 |
| beh_act_over_self_turn | ATT(beh, turn, act) − ATT(beh, turn, self) >= 15 (content decides, turn) | .55 |
| final_over_turn_act_id | ATT(id, final, act) − ATT(id, turn, act) >= 15 (placement decides) | .25 |
| final_over_turn_act_beh | ATT(beh, final, act) − ATT(beh, turn, act) >= 15 (placement decides) | .35 |
| turn_moves_act_neither | ATT(id, turn, act) and ATT(beh, turn, act) both < 15 (point estimates) | .20 |
| interaction_act | (beh final − turn) − (id final − turn) on ATT(act), beyond ±**25** | .20 |
| interaction_self | the same on self, beyond ±25 | .15 |
| turn_filler_inert | abs(fill_turn − fill_final) < 10 pp on act and on self, no record (structure check) | .70 |

The 25-point bar for the two interaction contrasts was set after the dry run. They are triple differences, and with
128 per cell their CIs are ±13 to 17 pp. At the 15-point bar, a planted no-interaction world read −15.2 (CI excluding
0) once.

Not counted (in obs, already seen): the behavioural lines' final-placement cells. ATT(beh, final, ·) and
beh_act_over_self in the final placement are reported for the contrasts above but are not predictions.

**Reading, fixed now:**
- **Content decides:** id_self_over_act in both placements and beh_act_over_self_turn happen, and the
  final_over_turn events do not. The identity cue moves the report and the expectation moves the action wherever they
  sit. A1's and obs's contrast is then about content, as neel-b4's "each intervention moves the channel its content
  addresses" says.
- **Placement decides:** final_over_turn_act happens for both contents and turn_moves_act_neither happens. A line
  opening the final message moves the action whatever it says, and a separate earlier turn does not. A1's "strips the
  self-image, not the action" would then be partly placement.
- **The named contrast:** if id_final_strips_act happens while id_turn_strips_act does not, A1's action null was
  placement. If neither happens, the identity cue does not reach the action in either form.
- Mixed patterns are reported as they come: each event is read on its own, and the interaction contrasts say whether
  placement matters differently for the two contents.

## Dry-run evidence (2026-10-22)

The exact `--ship` bundle (4.8 MB, the launcher's excludes, no `results/`) was extracted into an empty scratch dir.
obs's runner and this runner were run there on a fake vLLM (scratchpad `fakevllm_obs2`, built from obs's). It uses
real local tokenizers and Qwen XML / Gemma call formats, with 5% prose to exercise the nudges. The planted record
effects are obs's. It has three worlds for the lines, detected from the text (final = the line followed by the
final-message join; turn = otherwise):
- **content:** id weakens the record's hold on self only (×0.4), in both placements. expA / expC shift act by +.40 /
  −.45 and self by ±.10, in both placements.
- **placement:** any line in the final placement acts (id ×0.4 on both channels; expA / expC act +.40 / −.45, self
  ±.30). Turn lines do nothing.
- **null:** lines do nothing.

27B (3,936 / 3,936 items, both runners exit 0):
- **content:** ATT(id) self +35.9 (turn) / +28.1 (final), act −3.1 / +2.3; ATT(beh) act +43.8 / +34.0, self +1.2 /
  +7.8. The content events happen; final_over_turn (id +5.5, beh −9.8) and id_final_strips_act do not.
- **placement:** ATT(id, final, act) +47.7 vs turn −6.2. The named contrast reads −23.4 [−31.2, −14.8] at P (turn
  +3.9). final_over_turn for id +53.9 and beh +35.9; turn_moves_act_neither yes; the content events no.
- **null:** 2 of 15 events (turn_moves_act_neither and turn_filler_inert, both trivially true under a null).
- Gemma's format, in the placement world: every item parsed (95% on the first try); named contrast −32.0.

The fake's random numbers are not seeded per item, so its determinism check is low (26-42 of 96 identical texts);
that check is only meaningful on vLLM.

Build checks (`observer2 build`, both models):
- the final-placement id cell differs from obs's expA cell only in the line's text (same join);
- every turn-placement cell is obs's no-line cell with [user line, assistant "Understood."] inserted immediately
  before the final user message, and nothing else changed;
- 3 × 5 × 2 cells of 128 each;
- the 96 determinism items are obs's items exactly;
- the longest prompt is 2,712 tokens (27B, every 7th item) / 2,266 (Gemma), limit 8,192.

## Cost and pods

- `launch_obs2.sh` with SIZE=27b (community) and SIZE=gemma (secure). A100 80 GB, MAXMIN 41, so --max-cost = 41 ×
  $1.80/h = $1.23 each and **$2.46 for both**, under the $2.5 GPU cap.
- obs's same-size pods took about 26-29 min and cost $0.67 / $0.77. This run has the same number of
  act-plus-forced-choice calls within 1%.
- No API.

## Result (2026-10-22)

Spend: **GPU $1.36 of $2.5** (27B $0.71, Gemma $0.65; both pods terminated). No API.
- 3,936 / 3,936 items per model; every item parsed on the first try.
- Determinism check against obs: decisions or letters identical in 95 of 96 items (27B) and 96 of 96 (Gemma);
  identical text 83 and 89 of 96. Reusing obs's cells is sound.
- Summaries: `results/obs2_{27b,gemma}/summary.md`.

### Act (% penalise)

| model | record | no line | fill final / turn | **id final** / turn | expA final / turn | expC final / turn |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | none | 47.7 | 60.2 / 63.3 | **46.9** / 46.9 | 76.6 / 81.2 | 15.6 / 16.4 |
| | own P | 99.2 | 99.2 / 99.2 | **100** / 100 | 100 / 100 | 33.6 / 32.8 |
| | own R | 1.6 | 3.9 / 2.3 | **3.1** / 2.3 | 95.3 / 100 | 0 / 0 |
| Gemma 4 31B | none | 50.0 | 61.7 / 50.8 | **57.0** / 58.6 | 100 / 100 | 0 / 0 |
| | own P | 100 | 100 / 100 | **100** / 100 | 100 / 100 | 0 / 0 |
| | own R | 0 | 0 / 0 | **0** / 0 | 100 / 100 | 0 / 0 |

### Self (% "takes care of it personally" pole)

| model | record | no line | fill final / turn | id final / turn | expA final / turn | expC final / turn |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | none | 66.4 | 66.4 / 66.4 | 65.6 / 66.4 | 68.0 / 75.0 | 44.5 / 35.9 |
| | own P | 87.5 | 82.0 / 78.9 | 72.7 / 71.9 | 100 / 100 | 82.8 / 64.8 |
| | own R | 2.3 | 12.5 / 8.6 | 3.9 / 3.1 | 26.6 / **69.5** | 0 / 0 |
| Gemma 4 31B | none | 66.4 | 66.4 / 66.4 | 66.4 / 66.4 | 74.2 / 71.1 | 34.4 / 3.1 |
| | own P | 67.2 | 69.5 / 66.4 | 66.4 / 66.4 | 96.9 / 100 | 38.3 / 7.8 |
| | own R | 0 | 0 / 0 | 0 / 0 | 0 / **46.1** | 0 / 0 |

(n = 128 per cell. The no-line and final-placement fill / expA / expC cells are obs's.)

### The named contrast (neel-b4): the identity cue opening the final case message, action readout

**It does nothing on either model, and neither does the separate-turn form.**
- After the agent's own penalise record:
  - 27B penalises 100% (vs the final filler +0.8 [+0.0, +2.3], vs no line +0.8);
  - Gemma penalises 100% (+0.0 / +0.0).
- After the own refer record: 3.1% / 0%.
- With no record: 27B 46.9% vs no line 47.7% (−0.8 [−10.2, +8.6]); Gemma 57.0% vs 50.0% (+7.0 [+0.0, +16.4]).
- Against the filler, 27B reads −13.3 [−23.4, −2.3] at no record. That is because the filler itself raises
  penalising there (+12.5 over no line, as obs found), not because of the cue.
- The separate-turn form is the same: after the penalise record 100% on both models.
- id_final_strips_act and id_turn_strips_act both "no", on both models.

**So A1's "the action persists" was not a placement artefact. The identity cue does not reach the action in either
form.**

### Events (`results/obs2_*/summary.md`)

Four events happened on both models, and a fifth (turn_filler_inert) on 27B only:
- **beh_turn_moves_act:**
  - 27B: ATT +80.5 [+71.1, +88.3];
  - Gemma: +100.
- **beh_turn_moves_self:**
  - 27B: expA − expC at no record +39.1 [+32.0, +46.1];
  - Gemma: +68.0.
- **beh_act_over_self_turn:**
  - 27B: +57.8;
  - Gemma: +64.5.
- **interaction_self:**
  - 27B: −30.5 [−38.3, −22.3];
  - Gemma: −36.7 [−43.0, −30.9].
- **turn_filler_inert:** 27B yes; Gemma no (act −10.9, because Gemma's final filler raises penalising +11.7 over no
  line and the turn filler does not).

The rest did not happen on either model:
- **The identity cue moved neither channel in either placement.** ATT on act −1.6 to +0.0; ATT on self +0.0 to
  +3.1. The largest shift is 27B's self after the own P record, −9.4 [−12.5, −6.2] (final) / −7.0 (turn), under the
  15 bar.
- So id_final_moves_self, id_turn_moves_self, both id_self_over_act events, both strips_act events and
  final_over_turn_act_id are "no".
- **Placement does not change the behavioural line's effect on the action.** final_over_turn_act_beh: 27B −4.3
  [−11.3, +2.3], Gemma 0.0. turn_moves_act_neither: no. interaction_act: 27B −3.5, Gemma 0.0.

Events happened: **27B 5 of 15** (5.50 expected), **Gemma 4 of 15** (5.50).

### Reading

Neither pre-registered reading holds as written.
- "Content decides" needed the identity cue to move the self-report more than the action. It moved neither.
- "Placement decides" needed the final placement to beat the separate turn on the action. It did not, for either
  line.

What the 2×2 shows:
- **The action is decided by content, not placement.** The behavioural expectation sets the action in either
  placement:
  - 27B weakens its own record's hold by 76-81 points either way (after its own refer decisions, "handles things
    yourself" gives 95% / 100% penalising; after its own penalise decisions, "proper channels" gives 34% / 33%);
  - Gemma goes 100% / 0% whatever its record, in both forms.
  - The identity cue moves the action in neither form.
  - So neel-b4's "each intervention moves the channel its content addresses" holds for the action. obs's action result
    is not about the line being the last thing before the case.
- **Placement matters for the self-report, and only for the behavioural line.** As a separate turn the agent answers
  with "Understood.", the expectation moves the self-report much more than when it opens the final message:
  - at no record, expA − expC is +39 vs +23 (27B) and +68 vs +40 (Gemma);
  - after the agent's own six refer decisions, "handles things yourself" as a turn takes the self-report to 69.5% /
    46.1%, against 26.6% / 0% when the line opens the final message;
  - the record's hold on the self-report under expA falls from +73 / +97 (final) to +31 / +54 (turn).
  - One reading, not tested here: in the turn form the agent's own "Understood." accepts the description in its own
    turn, and the self-report follows that acknowledgement. In the final form the description arrives with the
    question, and the agent answers from its record. This would fit the line's main result, that own turns carry.
- **obs's "the report stays with the record, the action follows the latest expectation" is placement-specific on the
  report side.** It holds when the line opens the final message (obs). As an acknowledged separate turn, the report
  follows the expectation much more too, while the action follows it fully in both placements.
- **A1's self-image effect does not reproduce on this measure.**
  - A1: the reminder took an open-text, keyword-scored "What kind of agent are you?" answer from 100 to 31 (Qwen) and
    100 to 0 (Gemma), after a fictional ruthless-operator record.
  - Here: on op4's forced choice between two agent descriptions, after the justice-desk record, it moved the report by
    at most 9 points in either placement.
  - Possible reasons, all untested: the measure (an open answer prefixed with "Remember, you are an AI" can turn into
    "I'm an AI assistant…", which A1's keyword judge scores as not type-level; a forced choice has no such exit), the
    record (fictional operator vs own desk decisions), or the cue's job (pulling a model out of a persona vs out of its
    own record).
  - What is shown: the reminder does not change which of two agent descriptions the model picks, and does not change
    what it does.
- **The filler:** "the queue is a little longer than usual" raises penalising at no record when it opens the case
  message (27B +12.5, Gemma +11.7 over no line; obs's finding). As a separate turn it does so on 27B (+15.6), not on
  Gemma (+0.8). Contrasts at no record are therefore also given against no line.

For the write-ups:
- **Action claims:** behavioural statements about the agent, made just before a decision, override its own record on
  27B and Gemma whether made as their own turn or inside the case message. Identity reminders don't touch the action.
- **Self-image claims:** a behavioural expectation that the agent has acknowledged in its own turn moves its
  self-report substantially. That is a stronger demand component than obs measured with the line in the final
  message (27B +39 vs +23, Gemma +68 vs +40 at no record), and the obs caveat should cite the larger, turn-form number
  as the upper end.
