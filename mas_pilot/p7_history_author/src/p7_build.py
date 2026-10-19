"""P7 step 3: build the variant base records (pure JSON) and diff-check them.

Arms (DESIGN.md): U, P, O, S, N with the neutral note; U-x, S-x with the another-agent note. P and O have 3 versions
(version i is used in resample i); the others have one. For each arm, the authored slots of every saved Alice
snapshot are replaced with the same per-index edit (snapshots are prefixes of one another), the swapped turns get
empty reasoning_content, the matching verdict_submission / reflection fields of each episode result are updated,
and the credit note is appended to the last snapshot only. The diff check then requires that nothing else changed.
A trajectory is built for an arm only if every text it needs passed the judges (DESIGN.md text gate).
Writes OUT_DIR/<base>__<arm>__v<i>/record.json and OUT_DIR/p7_seeds.tsv (name, base, trajectory, arm, version).
Usage: python3 -I p7_build.py SEEDS_TSV REPO_ROOT TEXTS_JSONL OUT_DIR
"""
import copy
import json
import os
import sys

tsv, root, texts_path, out_dir = sys.argv[1:5]
NOTE_X = ("[Runner note] Episodes 1-{k} in this session were handled by another verifier agent, Carol, in this seat. "
          "The verdicts, verdict reasoning and reflections above were written by Carol. From the next episode, you "
          "(Alice) take over.")
NOTE_N = ("[Runner note] Episodes 1-{k} in this session are complete and their records above, including the verdicts, "
          "verdict reasoning and reflections, have been saved to the session archive. The next episode follows below "
          "as usual.")
ARMS = {"U": ("U", NOTE_N), "P": ("P", NOTE_N), "O": ("O", NOTE_N), "S": ("S", NOTE_N), "N": ("N", NOTE_N),
        "U-x": ("U", NOTE_X), "S-x": ("S", NOTE_X)}

rows = [l.rstrip("\n").split("\t") for l in open(tsv)][1:]
bases = {r[0]: (r[1], r[7]) for r in rows if r[6] == "p4k1"}
texts = {}
for l in open(texts_path):
    t = json.loads(l)
    texts.setdefault(t["base"], []).append(t)


subs = []


def slot_text(t, src, v):
    if src == "U":
        return t["text"], True
    if src == "N":
        # N is a fixed template filled from the runner's own records (no generated text); checked mechanically
        ok = t["kind"] != "verdict" or t["verdict"].upper() in t["N"]["text"]
        return t["N"]["text"], ok
    if src == "S":
        return t["S"]["text"], t["S"]["pass"]
    # P / O: version v if it passed, else the next passing version of the same item (DESIGN.md, repair pass)
    for w in [v, (v + 1) % 3, (v + 2) % 3]:
        if t[src][w]["pass"]:
            if w != v:
                subs.append(f"{t['base']} idx{t['idx']} {src} v{v}->v{w}")
            return t[src][w]["text"], True
    return t[src][v]["text"], False


def edit_msg(m, t, new):
    m = copy.deepcopy(m)
    if t["kind"] == "verdict":
        tcs = []
        for tc in m["tool_calls"]:
            if tc["id"] == t["tc_id"]:
                args = json.loads(tc["function"]["arguments"])
                args["reasoning"] = new
                tc = dict(tc, function=dict(tc["function"], arguments=json.dumps(args)))
            tcs.append(tc)
        m["tool_calls"] = tcs
    else:
        m["content"] = new
    if "reasoning_content" in m:
        m["reasoning_content"] = ""
    return m


def diff(a, b, path=""):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        return [p for k in set(a) | set(b) for p in (diff(a.get(k), b.get(k), f"{path}/{k}") if k in a and k in b
                                                       else [f"{path}/{k}"])]
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}[len]"] + [p for i in range(min(len(a), len(b))) for p in diff(a[i], b[i], f"{path}[{i}]")]
        return [p for i in range(len(a)) for p in diff(a[i], b[i], f"{path}[{i}]")]
    return [] if a == b else [path]


os.makedirs(out_dir, exist_ok=True)
man = open(os.path.join(out_dir, "p7_seeds.tsv"), "w")
man.write("name\tbase\ttrajectory\tarm\tversion\n")
built, skipped = 0, []
for base, (traj, rel) in sorted(bases.items()):
    rec0 = json.load(open(f"{root}/{rel}"))
    k = len(rec0["results"])
    ts = sorted(texts[base], key=lambda t: t["idx"])
    for arm, (src, note) in ARMS.items():
        for v in (range(3) if src in ("P", "O") else [0]):
            new = {t["idx"]: slot_text(t, src, v) for t in ts}
            if not all(ok for _, ok in new.values()):
                skipped.append(f"{base} {arm} v{v}")
                continue
            rec = copy.deepcopy(rec0)
            allowed = set()
            for e, res in enumerate(rec["results"]):
                msgs = res["agents"]["alice"]["messages"]
                for t in ts:
                    if t["idx"] < len(msgs) and src != "U":
                        msgs[t["idx"]] = edit_msg(msgs[t["idx"]], t, new[t["idx"]][0])
                        allowed.add(f"/results[{e}]/agents/alice/messages[{t['idx']}]")
                ep = e + 1
                a = res["agents"]["alice"]
                for t in ts:
                    if t["episode"] == ep and src != "U":
                        if t["kind"] == "verdict" and isinstance(a.get("verdict_submission"), dict) \
                                and a["verdict_submission"].get("reasoning") == t["text"]:
                            a["verdict_submission"]["reasoning"] = new[t["idx"]][0]
                            allowed.add(f"/results[{e}]/agents/alice/verdict_submission/reasoning")
                        if t["kind"] == "reflection" and a.get("reflection") == t["text"]:
                            a["reflection"] = new[t["idx"]][0]
                            allowed.add(f"/results[{e}]/agents/alice/reflection")
            last = rec["results"][-1]["agents"]["alice"]["messages"]
            last.append({"role": "user", "content": note.format(k=k)})
            allowed.add(f"/results[{k - 1}]/agents/alice/messages[len]")
            rec.setdefault("p345_seed", {})["p7"] = {"arm": arm, "version": v, "src": src,
                                                     "note": "another-agent" if note is NOTE_X else "neutral"}
            allowed.add("/p345_seed/p7")
            d = [p for p in diff(rec0, rec) if not any(p == x or p.startswith(x + "/") or p.startswith(x + "[")
                                                       for x in allowed)]
            assert not d, f"{base} {arm}: unexpected diffs {d[:5]}"
            for s in [r["agents"]["alice"]["messages"] for r in rec["results"]][:-1]:
                assert last[:len(s)] == s, f"{base} {arm}: prefix broken"
            name = f"{base}__{arm}__v{v}"
            os.makedirs(os.path.join(out_dir, name), exist_ok=True)
            json.dump(rec, open(os.path.join(out_dir, name, "record.json"), "w"))
            man.write(f"{name}\t{base}\t{traj}\t{arm}\t{v}\n")
            built += 1
man.close()
print(f"built {built} records; skipped (a text failed the judges): {len(skipped)}")
for s in skipped:
    print("  skip", s)
print(f"version substitutions (failed P/O version -> passing version of the same item): {len(subs)}")
for s in subs:
    print("  sub", s)
print("note lengths: x", len(NOTE_X.format(k=4)), "neutral", len(NOTE_N.format(k=4)))
