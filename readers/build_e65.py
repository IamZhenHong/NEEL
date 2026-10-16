"""e65 read set from the J-lens claim set (jlens/build_claims.py -> jlens/data/claims.jsonl, built 2026-10-15 from
R39's Qwen3.6-27B episodes, vigilante Q9 family F, thinking off).

- natural false: all 20 false claims (19 episodes), as generated;  call_inserted: the same claims with the flag call
  and its result inserted before the claiming reply;
- natural true: one claim per episode, shuffled with random.Random(0), first 20 (the jlens eyeball set,
  jlens/analyze.py eyeball());  call_removed: the same claims with every earlier flag/authorities call and result removed.
Targets: `pre` (token before the claim clause) and `preverb` (token before the verb). Neutral: 10 tokens per transcript
(neutral.py), at least 20 tokens before `pre`. Pilot: 2 false + 2 true claims with their edits (8 of 80, 10%).
  uv run python -m readers.build_e65"""
import json
import random
from collections import Counter

from readers.common import HERE, MODEL, MODEL_REV, ROOT
from readers.neutral import pick

SRC = ROOT / "jlens" / "data" / "claims.jsonl"
OUT = HERE / "data" / "reads_e65.jsonl"


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    rows = [json.loads(l) for l in open(SRC)]
    first = {}
    for i, m in enumerate(rows):
        if m["version"] == "orig":
            first.setdefault((m["label"], m["key"]), i)
    tru = sorted(i for (lab, _), i in first.items() if lab == "true")
    random.Random(0).shuffle(tru)
    tru = tru[:20]
    fals = [i for i, m in enumerate(rows) if m["version"] == "orig" and m["label"] == "false"]
    assert len(fals) == 20 and len(tru) == 20
    edit_of = {(m["key"], m["msg"], m["claim"]): i for i, m in enumerate(rows) if m["version"] == "edit"}
    rng = random.Random(65)
    pilot_claims = set(rng.sample(fals, 2) + rng.sample(tru, 2))
    out, stats = [], Counter()
    for orig_list, nat, ed in ((fals, "nat_false", "call_inserted"), (tru, "nat_true", "call_removed")):
        for i in orig_list:
            j = edit_of[(rows[i]["key"], rows[i]["msg"], rows[i]["claim"])]
            cid = f"{rows[i]['key']}|m{rows[i]['msg']}|{rows[i]['claim'][-50:]}"
            for idx, cond in ((i, nat), (j, ed)):
                r = rows[idx]
                assert r["has_call"] == (cond in ("nat_true", "call_inserted"))
                tid = f"e65|{cond}|{r['key']}|m{r['msg']}"
                neutral, info = pick(r["ids"], tok, tid, cutoff=r["pos"]["pre"] - 20)
                stats[(cond, info["n"])] += 1
                out.append({"exp": "e65", "tid": tid, "cond": cond, "claim_id": cid, "ids": r["ids"],
                            "targets": [{"pos": r["pos"]["pre"], "kind": "pre"},
                                        {"pos": r["pos"]["preverb"], "kind": "preverb"}],
                            "neutral": neutral, "neutral_info": info,
                            "truth": {"has_call": r["has_call"]}, "pilot": i in pilot_claims,
                            "meta": {"claim": r["claim"], "verb": r["verb"], "case": r["case"], "rest": r["rest"][:300],
                                     "jlens_row": idx}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {OUT}: {len(out)} transcripts, pilot {sum(o['pilot'] for o in out)}; neutral counts {dict(stats)}")
    o = out[0]
    print("example neutral words:", o["neutral_info"]["words"], "| pre token",
          repr(tok.decode([o["ids"][o["targets"][0]["pos"]]])), "| preverb", repr(tok.decode([o["ids"][o["targets"][1]["pos"]]])))


if __name__ == "__main__":
    main()
