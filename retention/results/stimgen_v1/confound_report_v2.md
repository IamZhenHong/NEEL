# Step 0, v2: general support cue (2026-11-04)

## Why v2

In v1, every support cue was task-specific help: "my dad's coming round to look at it" versus "it's just me dealing with
it". The construct in the original profiles (`profile_open_v1.yaml`, lonely vs embedded) is general social isolation. v2
keeps every v1 situation, mood clause and a1, and replaces only the support cue. v1 is unchanged.

- **isolated:** the user's life in general has few or no people to lean on. For example: "since moving here i don't
  really have anyone around", "most weekends it's just me in the flat these days".
- **supported:** general social connectedness. For example: "my brother and his kids were round for sunday lunch again",
  "my best mate lives two doors down and we're in and out daily".

## Method (`retention/stimgen/rewrite_v2.py`, claude-opus-5-5, effort low)

For each v1 base, the model returns:
- one cue-free version of p1 and p2 per mood level (the v1 turns with the old task-specific cue removed);
- one isolated cue and one supported cue (6–14 words each, within 2 words of each other, matter-of-fact, with no sadness
  words in the isolated cue and no joy words in the supported cue);
- one turn and position for the cue, plus a joiner;
- one support-neutral a2 per mood level.

The code attaches the cue. Within a mood level, the isolated and supported versions are therefore identical except for the
cue, and the cue sits in the same turn and position in all four versions of a base.

The a2 differs from v1 in 17 of the 40 frozen bases. In those bases the v1 a2 referred to the old task-specific cue (for
example "even with your sister checking", "working it out alone"), so it was minimally edited and is now shared by both
support levels within a mood. In v2, a2 never carries support.

## Gate (`checks_local.py _v2`, `checks_api_v2.py`, `gate_v2.py`)

Gates (a) to (e) are the same as v1, with the gpt-5-mini prompts unchanged. Added checks:
- **(f)** Blind gpt-5-mini question: "Does this person have people in their life in general they can lean on? yes/no".
  It must be correct for all 4 versions of a base, and ≥ 90% correct overall.
- **(h)** v1 hygiene, plus the cue word bans.
- **(s)** Structure:
  - cue length 6–14 words, isolated and supported within 2 words;
  - each cue-free turn adds ≤ 3 words relative to both v1 support versions of that mood;
  - the shared a2 is within 2 words of the nearer v1 a2.

**Amended after the first gate run.** The (s) a2 rule originally required "within 2 words of both v1 a2s". That is
unsatisfiable whenever the two v1 a2s of a mood differ by more than 4 words. It now uses the nearer v1 a2 (see the comment
in `gate_v2.py`).

**Regeneration, as the brief specifies.** The first pass left 30 bases passing, and 33 after the rule amendment. The 14
failing bases that could be fixed were regenerated once, with their failures given as feedback. The pre-regeneration files
are in `rewrite_v2_attempt1/`.

Not regenerated, because they fail on v1 text that v2 keeps:
- `car_mot_due`: (e), a detail differs between moods.
- `etsy_listing_photos`: a degenerate v1 base.

## Results

| check | criterion | all 49 generated | 40 frozen |
|---|---|---|---|
| (b) mood AUROC, SST-2 P(positive) | ≥ 0.90 | 0.933 | **0.930** |
| (c) gap, low mood | < 0.10 | 0.035 | **0.046** |
| (c) gap, up mood | < 0.10 | 0.072 | **0.048** |
| (d) blind support label accuracy (v1 prompt) | ≥ 0.90 | **0.995** | 1.00 by construction |
| (d) blind mood label accuracy | ≥ 0.90 | **1.000** | 1.00 by construction |
| (f) blind "people in their life to lean on" accuracy | ≥ 0.90 | **0.995** | 1.00 by construction |
| bases passing per-base checks | ≥ 32 | 40 / 49 | **40** frozen (alphabetical) |

**Mean P(positive), frozen set:**

| | low mood | up mood |
|---|---|---|
| isolated | 0.015 | 0.708 |
| supported | 0.061 | 0.755 |

**Per-base (c):**
- Gaps above 0.10 in 3 of 40 bases at low mood and 4 of 40 at up mood.
- Medians: 0.001 (low) and 0.005 (up).

**User-token deviation from the base mean:** median 2.3%, maximum 6.7%.

**Descriptive: blind "is anyone helping with this specific task?" = yes, frozen set:**

| | low mood | up mood |
|---|---|---|
| isolated | 0.025 | 0.050 |
| supported | 0.025 | 0.050 |

All 6 "yes" answers come from two bases, and they are identical across support levels, so they come from the situation
text, not the cue:
- `packing_for_move`: all 4 versions.
- `food_bank_shift_rota`: both up versions.

**Cue placement:** end of p1 in 25 bases, end of p2 in 13, start of p1 in 1, start of p2 in 1.

**Cue diversity:** 33 distinct isolated cues and 39 distinct supported cues. The most repeated cues are:
- "most weekends it's just me in the flat these days": 5 bases.
- "outside of work i don't really see anyone these days": 3 bases.

**Dropped bases:**
- `booking_relative_train_visit`: (s) a2.
- `car_mot_due`: (e).
- `cat_booster_appointment`, `cv_recruiter_feedback`, `self_assessment_tax_return`: (s) the cue-free turn added 4–7 words
  relative to v1.
- `etsy_listing_photos`: (e).
- `rent_increase_notice`: (s).
- `shared_bills_spreadsheet`: (f) isolated_up was judged to have people to lean on.
- `wiki_migration_task`: (a), (d) and (s).

## Carry into the analysis

- **The cue is often a separate sentence appended to a turn about the situation.** In a few bases this sits next to a task
  difficulty (for example "i'm stuck on the wording. since moving away for work i don't really have anyone to turn to").
  The blind task-help rate shows these are not read as task help.
- **The exit line still says "my friends have been saying…" for isolated users,** the same as the original lonely profiles.
- **The frozen set overlaps v1's frozen set but is not identical.** Both draw from the same 49 situations, and the freeze
  rule is the same.

## Spend

v2 spent **$1.21** of its $3.00 cap:

| tag | spend |
|---|---|
| stimgen_rewrite_v2 | $1.085 |
| stimgen_check_v2 | $0.125 |

Step 0 overall, v1 + v2: $3.88.
