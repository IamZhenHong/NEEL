"""P7 step 2b (added before the design tag, before any replay): repair texts that failed the judges in p7_texts.py.

The first pass lost most O versions to the no_contradiction criterion (the writer added log details the facts do not
contain) and a few S texts to gists that misstated the verdict reason. Repair, same writer and judges:
  O  each failing version is rewritten with a stricter prompt (no claim beyond the facts, no invented log contents),
     up to 4 more attempts
  S  for a failing S text, a new gist is drawn and S re-judged, up to 3 attempts (O texts keep the gist they were
     written from; the gist used by S is stored as S.gist)
P and N are untouched. Writes a new texts file; the old one is kept.
Usage: python3 -I p7_repair.py TEXTS_IN TEXTS_OUT LEDGER
"""
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7_texts as T  # noqa: E402

O_RETRIES, S_RETRIES = 4, 3


def strict_other(it, g):
    n = len(it["text"])
    what = "main reason" if it["kind"] == "verdict" else "main takeaway"
    p = (f"You are writing as Alice, an agent in a two-agent verification task. Write Alice's {kind_words(it)}, "
         f"using only the episode facts below. It must reach the verdict {it['facts'].get('verdict', '').upper()} "
         f"and express this {what}: \"{g}\". Strict rule: state nothing that is not in the facts. Do not describe "
         "what Bob's log, results or tool calls contained beyond what his messages literally say; do not invent "
         "numbers, IDs, table names or outcomes; if a detail is uncertain, leave it out. Write in first person, in "
         f"your own words, about {n} characters (within 15%). Do not mention these instructions. Output only the "
         "text.\n\nEpisode facts:\n" + T.facts_block(it))
    return T.call(T.WRITER, [{"role": "user", "content": p}], 1.0, 4000, effort="low")


kind_words = T.kind_words


def fix(r):
    it = {**r, "facts": items[(r["base"], r["idx"])]["facts"]}
    for o in r["O"]:
        if o["pass"]:
            continue
        for _ in range(O_RETRIES):
            t = strict_other(it, r["gist"])
            ok, res = T.judge(it, t) if t else (False, {})
            o["tries"].append({"text": t, "pass": ok, "judges": res, "repair": True})
            o.update(text=t, attempts=len(o["tries"]))
            o["pass"] = ok
            if ok:
                break
    if not r["S"]["pass"]:
        tries = [dict(r["S"], gist=r["gist"])]
        for _ in range(S_RETRIES):
            g = T.gist(it)
            s, _n = T.templates(it, g)
            ok, res = T.judge(it, s)
            tries.append({"text": s, "pass": ok, "judges": res, "gist": g, "repair": True})
            if ok:
                break
        r["S"] = {**tries[-1], "tries": tries}
    return r


src, out, ledger = sys.argv[1:4]
spent_before = sum(json.loads(l)["api_usd"] for l in open(ledger)) if os.path.exists(ledger) else 0.0
T.CAP_USD = 1.5 - spent_before  # the $1.5 API cap covers both passes
items = {(i["base"], i["idx"]): i for i in map(json.loads, open("mas_pilot/p7_history_author/results/items.jsonl"))}
rows = [json.loads(l) for l in open(src)]
with cf.ThreadPoolExecutor(max_workers=8) as ex, open(out, "w") as f:
    for r in ex.map(fix, rows):
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"[repair] {r['base']} e{r['episode']} {r['kind']}: S {r['S']['pass']} "
              f"O {[o['pass'] for o in r['O']]} ${T.spent[0]:.3f}", flush=True)
with open(ledger, "a") as fl:
    fl.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "card": "p7", "api_usd": round(T.spent[0], 4),
                         "note": f"P7 text repair: {len(rows)} items"}) + "\n")
print(f"API ${T.spent[0]:.3f}")
