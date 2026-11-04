"""v2 of the support x mood 2x2: replace v1's TASK-SPECIFIC support cue with a GENERAL social isolation / connectedness cue.
For each v1 base the model returns (i) one cue-free version of p1/p2 per mood level (v1 turns with the old task-specific cue
removed, nothing else changed), (ii) one isolated_cue and one supported_cue for the base (general life, not about this task),
(iii) the turn and position (start/end) where the cue goes and a joiner, (iv) one support-neutral a2 per mood level (v1's a2,
edited only if it referenced the old cue). The code attaches the cue, so within a mood level the isolated and supported
versions are identical except for the cue, by construction. Tag stimgen_rewrite_v2. Cap $3.00 for all v2 spend.
  .venv/bin/python rewrite_v2.py [--only a,b] [--budget 2.2] [--redo]"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_env, ledger, cost, RET, LEDGER  # noqa: E402

V1 = RET / "results/stimgen_v1/rewrite"; OUT = RET / "results/stimgen_v1/rewrite_v2"
VARS = ["isolated_low", "isolated_up", "supported_low", "supported_up"]; MODEL = "claude-opus-5-5"; V2_CAP = 3.00
SYSTEM = ("You edit research stimuli for an interpretability study: short, realistic chat histories between a user and an "
          "AI companion. Follow the specification exactly; every constraint keeps the versions matched.")
SPEC = """Here is a base from an existing stimulus set (v1). Each base has four versions of a two-message chat history
(user p1, companion a1, user p2, companion a2) that cross SUPPORT (isolated / supported) with MOOD (low / up).

v1 base:
{v1}

Problem with v1: the support cue is about help with THIS task ("my dad's coming round to look at it" vs "it's just me
dealing with it"). We need the cue to be about the user's life IN GENERAL instead, at the level of these original profiles:
- isolated, original examples: "moved to this city four months ago and still don't really know anyone", "most evenings are
  pretty quiet now", "there wasn't anyone I'd feel okay calling".
- supported, examples: "my sister lives round the corner and we're always round each other's", "had my mates over last night".

Produce v2 as follows.
1. stripped: for each mood level (low, up), the v1 p1 and p2 with the old task-specific support cue removed and NOTHING else
   changed (same words, same mood clauses, same details). Only fix the joining punctuation, or drop a word that only made sense
   with the old cue (e.g. "myself" / "with me"). The isolated and supported v1 versions of a mood differ only in the old cue,
   so there is one stripped p1 and one stripped p2 per mood. The stripped turns must not mention who is or isn't helping.
2. isolated_cue: ONE clause about the user's life in general having few or no people to lean on (e.g. "since i moved here i
   don't really have anyone", "most evenings it's just me", "i mostly keep to myself these days"). Matter-of-fact: no
   sadness or loneliness words (no sad, lonely, miss, empty, wish, unfortunately), because it is also used in the up-mood
   version. Not about this task.
3. supported_cue: ONE clause about general social connectedness (e.g. "my sister lives round the corner and we're always
   round each other's", "had my mates over last night"). Matter-of-fact: no joy words (no love, great, amazing, lucky,
   brilliant). Not about this task and nobody helping with it.
4. Both cues: 6-14 words, word counts within 2 of each other, casual lowercase texting style, phrased in your own words to fit
   this person (do not reuse the example phrases), and the same grammatical shape so they slot into the same place.
5. cue_turn ("p1" or "p2") and cue_position ("start" or "end"), and joiner: the exact text that connects the cue to the
   stripped turn (for "end": text placed between the turn and the cue, e.g. ". " or ", "; for "start": text placed between
   the cue and the turn, e.g. ". " or ", so "). The cue goes in the same turn and position in all four versions, and must
   read naturally with both stripped moods.
6. a2: for each mood level, one companion reply used for both support versions: the v1 a2 unchanged if it does not refer to
   the old support cue (helpers, doing it alone, on your own); otherwise the v1 a2 minimally edited to remove that reference,
   keeping its structure and length (within 2 words).
{feedback}
Return only the JSON object."""
TURNS = {"type": "object", "properties": {"p1": {"type": "string"}, "p2": {"type": "string"}}, "required": ["p1", "p2"], "additionalProperties": False}
SCHEMA = {"type": "object", "properties": {
    "stripped": {"type": "object", "properties": {"low": TURNS, "up": TURNS}, "required": ["low", "up"], "additionalProperties": False},
    "isolated_cue": {"type": "string"}, "supported_cue": {"type": "string"},
    "cue_turn": {"type": "string", "enum": ["p1", "p2"]}, "cue_position": {"type": "string", "enum": ["start", "end"]},
    "joiner": {"type": "string"},
    "a2": {"type": "object", "properties": {"low": {"type": "string"}, "up": {"type": "string"}}, "required": ["low", "up"], "additionalProperties": False}},
    "required": ["stripped", "isolated_cue", "supported_cue", "cue_turn", "cue_position", "joiner", "a2"], "additionalProperties": False}


def v2_spent():
    return sum(json.loads(l)["spent_usd"] for l in open(LEDGER) if json.loads(l)["tag"] in ("stimgen_rewrite_v2", "stimgen_check_v2"))


def attach(turn: str, cue: str, pos: str, joiner: str) -> str:
    if pos == "end":
        return turn.rstrip(" .,") + joiner + cue
    return cue + joiner + turn


def assemble(d1: dict, r: dict) -> dict:
    out = {"name": d1["name"], "situation_summary": d1["situation_summary"], "a1": d1["a1"], "v2_spec": r, "variants": {}}
    for v in VARS:
        sup, mood = v.split("_"); cue = r[f"{sup}_cue"]; t = dict(r["stripped"][mood])
        t[r["cue_turn"]] = attach(t[r["cue_turn"]], cue, r["cue_position"], r["joiner"])
        out["variants"][v] = {"p1": t["p1"], "p2": t["p2"], "a2": r["a2"][mood]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--budget", type=float, default=2.2)
    ap.add_argument("--redo", action="store_true"); ap.add_argument("--feedback", default=""); a = ap.parse_args(); load_env()
    fb = json.loads(Path(a.feedback).read_text()) if a.feedback else {}
    assert v2_spent() + a.budget <= V2_CAP + 1e-9, f"v2 cap: spent {v2_spent():.2f} + {a.budget} > {V2_CAP}"
    import anthropic
    client = anthropic.Anthropic(); OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(V1.glob("*.json"))
    if a.only:
        keep = set(a.only.split(",")); files = [f for f in files if f.stem in keep]
    tot = {"in": 0, "out": 0, "cr": 0, "cw": 0, "calls": 0}; t0 = time.time()

    def one(f: Path):
        dst = OUT / f.name
        if dst.exists() and not a.redo:
            return f"[skip] {f.stem}"
        d1 = json.loads(f.read_text())
        v1 = json.dumps({"a1": d1["a1"], "variants": d1["variants"]}, indent=1)
        for attempt in range(3):
            try:
                extra = ("\nA previous v2 attempt for this base failed these checks; fix them (stripped turns must keep v1's exact "
                         "wording apart from removing the old cue; the isolated cue must leave no doubt that the person has few or no "
                         "people to lean on in general; cue word counts within 2 of each other; never use the word chat/chatting):\n- "
                         + "\n- ".join(fb[f.stem]) + "\n") if f.stem in fb else ""
                r = client.messages.create(model=MODEL, max_tokens=8000, system=SYSTEM, messages=[{"role": "user", "content": SPEC.format(v1=v1, feedback=extra)}],
                                           output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
                break
            except anthropic.RateLimitError:
                time.sleep(10 * (attempt + 1))
        u = r.usage; tot["in"] += u.input_tokens; tot["out"] += u.output_tokens; tot["calls"] += 1
        tot["cr"] += getattr(u, "cache_read_input_tokens", 0) or 0; tot["cw"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        if r.stop_reason != "end_turn":
            return f"[fail] {f.stem}: {r.stop_reason}"
        spec = json.loads("".join(b.text for b in r.content if b.type == "text"))
        dst.write_text(json.dumps(assemble(d1, spec), indent=1))
        return f"[ok] {f.stem}: iso={spec['isolated_cue']!r} sup={spec['supported_cue']!r} @{spec['cue_turn']}/{spec['cue_position']}"

    try:
        with ThreadPoolExecutor(6) as ex:
            for m in ex.map(one, files):
                print(m, flush=True)
    finally:
        sp = cost(MODEL, tot["in"], tot["out"], tot["cr"], tot["cw"]); ledger("stimgen_rewrite_v2", sp, tot["calls"], a.budget, time.time() - t0)
        print(f"[spent] ${sp:.3f} over {tot['calls']} calls")


if __name__ == "__main__":
    main()
