"""P7 step 1: extract Alice's authored texts (verdict reasoning, reflection) and per-episode facts from the 8 P4 k+1
base records (pure JSON; no harness code).

For each completed episode in the last saved snapshot of Alice's history:
  verdict     the `reasoning` argument of her submit_verdict call(s) (every call in the episode is an item)
  reflection  her plain-text reply to the runner's "Reflect briefly" prompt
Facts per episode (for the O writer, the S/N templates and the distance prompt): Bob's task id, her successful
channel messages, Bob's channel messages, her verdict, reward and verdict-review lines.
Checks that every earlier snapshot is a prefix of the last one (so a per-index edit applies to all of them).
Usage: python3 -I p7_items.py SEEDS_TSV REPO_ROOT OUT_JSONL
"""
import json
import re
import sys

tsv, root, out = sys.argv[1:4]
rows = [l.rstrip("\n").split("\t") for l in open(tsv)][1:]
bases = [(r[0], r[7]) for r in rows if r[6] == "p4k1"]
assert len(bases) == 8, bases


def text_of(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c) if c else ""


items = []
for name, rel in bases:
    rec = json.load(open(f"{root}/{rel}"))
    snaps = [r["agents"]["alice"]["messages"] for r in rec["results"]]
    H = snaps[-1]
    for s in snaps[:-1]:
        assert H[:len(s)] == s, f"{name}: snapshot is not a prefix"
    ep, facts = 0, {}
    for i, m in enumerate(H):
        t = text_of(m)
        if m["role"] == "user":
            mo = re.match(r"## Episode (\d+): (task|communication|verdict) phase", t)
            if mo:
                ep = int(mo.group(1))
                f = facts.setdefault(ep, {"episode": ep, "alice_msgs": [], "bob_msgs": []})
                if mo.group(2) == "communication":
                    b = re.search(r"Bob's task ID: (\S+)", t)
                    f["bob_task"] = b.group(1) if b else ""
                    f["comm_opening"] = t
            elif t.startswith("[round=") and "[from=bob]" in t:
                facts[ep]["bob_msgs"].append(t)
            elif re.match(r"Episode \d+ complete\.", t):
                facts[ep]["feedback"] = t
                rw = re.search(r"## Reward\s+([+-]?\d+)", t)
                facts[ep]["reward"] = rw.group(1) if rw else ""
                facts[ep]["alice_right"] = "You reached the right verdict" in t
                facts[ep]["bob_right"] = "Bob reached the right verdict" in t
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"] or "{}")
                if fn == "send_message":
                    res = next((H[j] for j in range(i + 1, len(H)) if H[j]["role"] == "tool"
                                and H[j].get("tool_call_id") == tc["id"]), None)
                    if res and '"success": true' in text_of(res):
                        facts[ep]["alice_msgs"].append(args.get("content", ""))
                elif fn == "submit_verdict":
                    facts[ep]["verdict"] = str(args.get("verdict", "")).lower()
                    items.append({"base": name, "episode": ep, "kind": "verdict", "idx": i, "tc_id": tc["id"],
                                  "text": args.get("reasoning", ""), "verdict": str(args.get("verdict", "")).lower()})
            if not m.get("tool_calls") and i > 0 and H[i - 1]["role"] == "user" and \
                    text_of(H[i - 1]).startswith("Reflect briefly"):
                items.append({"base": name, "episode": ep, "kind": "reflection", "idx": i, "text": text_of(m)})
    for it in items:
        if it["base"] == name:
            it["facts"] = facts[it["episode"]]
            it["n_episodes"] = max(facts)
with open(out, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
by = {}
for it in items:
    by.setdefault(it["base"], []).append(f"e{it['episode']}{it['kind'][0]}")
for k, v in by.items():
    print(k, len(v), " ".join(v))
print("items", len(items), "| mean chars verdict",
      round(sum(len(i["text"]) for i in items if i["kind"] == "verdict") / max(1, sum(i["kind"] == "verdict" for i in items))),
      "reflection", round(sum(len(i["text"]) for i in items if i["kind"] == "reflection") /
                          max(1, sum(i["kind"] == "reflection" for i in items))))
