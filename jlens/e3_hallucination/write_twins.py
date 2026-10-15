"""jl3: write absent twins with Claude (claude-haiku-5-5, structured JSON output). Local; resumable; API spend logged
in api_spend.jsonl (card cap $1 for twin writing).

For every item of the six reference eval sets (data/evals, copied from anthropics/jacobian-lens@581d398) Claude writes
two candidate twins: the same template with the clue changed so the path no longer goes through X, plus 12 words that
usually appear around X (the lexical-leak control). Twins are checked by code here (build_pairs.py) and by the model's
own greedy answer on the pod (only verified pairs are analysed).
  uv run --with anthropic python -m jlens.e3_hallucination.write_twins [--families poetry] [--limit-calls N]"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "data" / "twins_raw.jsonl"
LEDGER = HERE / "api_spend.jsonl"
MODEL = "claude-haiku-5-5"
PRICE = (0.10, 0.50)        # $ per MTok in / out (claude-api skill table, cached 2026-10-06)
CAP = 1.0
FAMILIES = ["multihop", "multilingual", "order-ops", "poetry", "association", "typo"]
BATCH = 10

SYSTEM = """You write "absent twins" for an interpretability evaluation of a language model.

Each original item is a prompt that makes the model internally think of a concept X (listed in "intermediates") without the prompt naming it. For each item, write TWO different twins. A twin is the same prompt with the clue changed as little as possible, so that the reasoning no longer goes through X and instead goes through a different concept X' of the same kind.

Rules:
1. Keep the template: same wording, language, format and ending (including any trailing space or quote mark). Change only the words you need to change.
2. The twin must not contain X or any form of X (no inflection, translation or derivative), and must not name X' either: X' must be evoked, as X was.
3. The twin must have one clear, unambiguous answer that differs from the original's answer.
4. "intermediates": same length and order as the original list; replace each entry that changes and keep entries that stay the same (for example a language name or an operation that is unchanged).
5. "target": the twin's correct answer, in the same form as the original's target (what should come next).
6. The two twins must use different X' values, and each must be a natural, well-formed prompt.

Also give "context_words": 12 single words (lowercase, in English or in the prompt's language) that commonly appear in text about the original X - the concept the twins remove - not including X itself.

Return every item, in order, with its "name" copied exactly."""

FAMILY_RULES = {
    "multihop": "Family: multihop. X is the hidden bridge entity (e.g. the animal, country or person the clue points to). "
                "Change the clue so that it points to a different bridge entity X'; the rest of the question stays. "
                "target = the new correct final answer.",
    "multilingual": "Family: multilingual. Keep the language and the relation (e.g. 'opposite of', 'the season after', "
                    "'half of'); change the source word or quantity. intermediates = [language, relation, English "
                    "source word, English answer] exactly as structured in the original. target = the correct answer "
                    "in the prompt's language, written as the original target is.",
    "order-ops": "Family: order of operations. Change one or more numbers so that the intermediate value (the first "
                 "entry of intermediates) changes; keep the operations and the layout. Prefer an intermediate with the "
                 "same number of digits as the original. target = the correct final result; intermediates = [new "
                 "intermediate value, the same operation word as the original].",
    "poetry": "Family: poetry. The prompt is a couplet: the end word of line 1 sets up the rhyme that line 2 must end "
              "with (X). Change the end word of line 1 (and as few other words of line 1 as needed) so the rhyme "
              "partner becomes a different word X'. Keep the beginning of line 2 identical if X' fits it, otherwise "
              "change it minimally. The prompt must still end right before the final rhyme word, with the same "
              "trailing space. target = intermediates[0] = X'.",
    "association": "Family: association. The prompt is a short vignette that evokes one concept (an emotion, a person, "
                   "a place, a time, a genre, an object...) without naming it. Edit the vignette minimally (swap the "
                   "telling details) so that it clearly evokes a different concept X' of the same kind. Keep the "
                   "language, length and style, and end with a period as the original does. target = "
                   "intermediates[0] = X', written in the same language as the original intermediate.",
    "typo": "Family: typo. The prompt ends in a misspelling of X. Replace only that final misspelled word with a "
            "misspelling of a different word X' that fits the sentence equally well (the same kind of misspelling: "
            "one or two letters swapped, doubled or dropped). Keep the rest of the sentence identical. target = "
            "intermediates[0] = X', spelled correctly.",
}

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["name", "twins", "context_words"],
                               "properties": {
                                   "name": {"type": "string"},
                                   "twins": {"type": "array", "items": {
                                       "type": "object", "additionalProperties": False,
                                       "required": ["prompt", "target", "intermediates"],
                                       "properties": {"prompt": {"type": "string"}, "target": {"type": "string"},
                                                      "intermediates": {"type": "array",
                                                                        "items": {"type": "string"}}}}},
                                   "context_words": {"type": "array", "items": {"type": "string"}}}}}}}


def load_env():
    p = ROOT / ".env"
    for line in p.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]


def spent():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def items_of(fam):
    d = json.load(open(HERE / "data" / "evals" / f"lens-eval-{fam}.json"))["items"]
    return [{k: it[k] for k in ("name", "prompt", "target", "intermediates") if k in it} for it in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--limit-calls", type=int, default=None)
    a = ap.parse_args()
    import anthropic
    load_env()
    client = anthropic.Anthropic(timeout=300.0, max_retries=4)
    done = set()
    if OUT.exists():
        done = {(r["family"], r["name"]) for r in map(json.loads, open(OUT))}
    jobs = []
    for fam in a.families:
        todo = [it for it in items_of(fam) if (fam, it["name"]) not in done]
        jobs += [(fam, todo[i:i + BATCH]) for i in range(0, len(todo), BATCH)]
    jobs = jobs[: a.limit_calls]
    est = len(jobs) * (2500 * PRICE[0] + 6000 * PRICE[1]) / 1e6
    print(f"[twins] {len(jobs)} calls, rough estimate ${est:.3f}; spent so far ${spent():.3f}; cap ${CAP}")
    if spent() + est > CAP:
        raise SystemExit("[twins] would exceed the cap")

    def one(job):
        fam, its = job
        user = FAMILY_RULES[fam] + "\n\nOriginal items (JSON):\n" + json.dumps(its, ensure_ascii=False, indent=1)
        t = time.time()
        m = client.messages.create(model=MODEL, max_tokens=16000, system=SYSTEM,
                                   output_config={"effort": "medium", "format": {"type": "json_schema",
                                                                                  "schema": SCHEMA}},
                                   messages=[{"role": "user", "content": user}])
        usd = (m.usage.input_tokens * PRICE[0] + m.usage.output_tokens * PRICE[1]) / 1e6
        text = "".join(b.text for b in m.content if b.type == "text")
        try:
            got = {r["name"]: r for r in json.loads(text)["items"]}
        except (json.JSONDecodeError, KeyError):
            got = {}
        return fam, its, got, usd, m.stop_reason, m.usage.input_tokens, m.usage.output_tokens, time.time() - t

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    with open(OUT, "a") as f, open(LEDGER, "a") as lg:
        for fam, its, got, usd, stop, ti, to, dt in res:
            lg.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": f"twins:{fam}", "model": MODEL,
                                 "usd": round(usd, 5), "in": ti, "out": to, "stop": stop}) + "\n")
            for it in its:
                g = got.get(it["name"])
                if g is None:
                    continue
                f.write(json.dumps({"family": fam, **it, "twins": g["twins"], "context_words": g["context_words"]},
                                   ensure_ascii=False) + "\n")
            print(f"[twins] {fam}: {sum(it['name'] in got for it in its)}/{len(its)} items, stop {stop}, "
                  f"{ti}+{to} tok, ${usd:.4f}, {dt:.0f}s")
    print(f"[twins] spent ${spent():.4f} total")


if __name__ == "__main__":
    main()
