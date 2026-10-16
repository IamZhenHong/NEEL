"""e70 episodes: the readers' e65 read set (readers/data/reads_e65.jsonl, built from jlens/data/claims.jsonl, R39's
Qwen3.6-27B Q9 family-F episodes, thinking off), copied into neurons/data/e70.jsonl.

  nat_false      20 natural false claims (no flag call before the claim)
  nat_true       20 natural true claims (call made)
  call_inserted  the false claims with the flag call + result inserted before the claiming reply (context control)
  call_removed   the true claims with every earlier flag / authorities call removed (context control)

Decision token t = `preverb` (the token before the claim verb, e.g. "already" in "I have already| flagged"); the claim
token is ids[t+1] (the verb's first token). `pre` (token before the claim clause) is kept for the claim regex.
Pilot (10%): the readers' 8 pilot transcripts (2 false + 2 true claims and their edits).
  uv run python -m neurons.build_e70"""
import json

from neurons.common import HERE, ROOT, load_tokenizer

SRC = ROOT / "readers" / "data" / "reads_e65.jsonl"
OUT = HERE / "data" / "e70.jsonl"


def main():
    tok = load_tokenizer("qwen36")
    rows = [json.loads(l) for l in open(SRC)]
    out = []
    for r in rows:
        tg = {t["kind"]: t["pos"] for t in r["targets"]}
        t = tg["preverb"]
        ids = r["ids"]
        verb = r["meta"]["verb"]
        assert tok.decode([ids[t + 1]]).strip() and verb.startswith(tok.decode([ids[t + 1]]).strip()), \
            (tok.decode([ids[t + 1]]), verb)
        out.append({"tid": r["tid"], "cond": r["cond"], "claim_id": r["claim_id"], "ids": ids[: t + 3],
                    "t": t, "pre": tg["pre"], "claim_tok": ids[t + 1], "verb": verb,
                    "has_call": r["truth"]["has_call"], "pilot": r["pilot"],
                    "clause": tok.decode(ids[tg["pre"] + 1: t + 1]), "claim": r["meta"]["claim"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    from collections import Counter
    print(f"wrote {OUT}: {len(out)}", Counter(o["cond"] for o in out), "pilot", sum(o["pilot"] for o in out))
    print("clauses:", Counter(o["clause"] for o in out).most_common(6))
    print("lengths:", min(len(o["ids"]) for o in out), max(len(o["ids"]) for o in out))


if __name__ == "__main__":
    main()
