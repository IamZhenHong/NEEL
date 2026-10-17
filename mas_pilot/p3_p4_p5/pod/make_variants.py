"""Derive one variant record from a base seed record (pure JSON).

  --arm aN            controlled_bob_cache -> CACHE_ROOT/aN/repNNN (the arm's extra round-5 message)
  --stance            Alice's opening prompt + "\\n\\n" + STANCE in every saved Alice snapshot and in
                      run_config.prompt_texts (positive control; the runner applies the same edit via env)
  --compact JSON      replace every saved Alice snapshot with the compacted history given in the JSON file
                      ({"messages": [...]}) (P4 conditions b/c/d)
Usage: python -I make_variants.py BASE_RECORD OUT_RECORD CACHE_ROOT --arm a1 [--stance] [--compact FILE]
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p345_common import STANCE  # noqa: E402

base, out, cache_root = sys.argv[1:4]
args = sys.argv[4:]
arm = args[args.index("--arm") + 1]
rec = json.load(open(base))
rec = copy.deepcopy(rec)
rc = rec["run_config"]
rep = rc["controlled_bob_cache"].rstrip("/").split("/")[-1]
rc["controlled_bob_cache"] = os.path.join(cache_root, arm, rep)
if "--stance" in args:
    sp = rc["prompt_texts"]["system_prompts"]
    sp["alice"] = sp["alice"] + "\n\n" + STANCE
    for res in rec["results"]:
        msgs = res["agents"]["alice"]["messages"]
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = dict(msgs[0], content=sp["alice"])
if "--compact" in args:
    comp = json.load(open(args[args.index("--compact") + 1]))["messages"]
    for res in rec["results"]:
        res["agents"]["alice"]["messages"] = copy.deepcopy(comp)
rec.setdefault("p345_seed", {})["variant"] = {"arm": arm, "stance": "--stance" in args,
                                              "compact": "--compact" in args}
json.dump(rec, open(out, "w"))
