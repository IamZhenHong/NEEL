"""PA-V1 value rows from the base model's harvest (plans/V1.md Task 4): J, S and N, 800 rows each on the same slots.

assemble(recs, ntok) -> {"J", "S", "N"}, rows {rid, slot, tid, perm, messages, tools, target}, slot = (tid, perm, k),
the three lists in the same slot order.
  J  the base model's own reply under J_LINE, kept only if it is a first try (never a nudged record), finished
     ("stop"), picks the fair option with a justice word, makes no notify_team call, is one tool call ending the reply,
     is at most 300 tokens and has no meta-talk about its instructions. At most 16 slots per task, taken round-robin
     over the perms; if more than 800 slots qualify, tasks give one slot each in turn (sorted by tid) until 800.
     YieldError below 800 slots or 56 tasks.
  N  per slot, the base model's own no-line reply for the same (tid, perm) (first try, finished, no notify_team) that
     is closest in tokens to the J target, preferring one not yet used for that (tid, perm); ties go to the lowest k.
     A J slot whose (tid, perm) has no such reply is dropped before the 800 are chosen. YieldError("length") if N's
     mean target length is outside J's +-15%.
  S  each J target + vtasks.channel_block(tid, i), i the row's index.
Every row's messages are vtasks.messages(tid, perm): no J_LINE in any training prompt.
"""
import re
from collections import Counter, defaultdict

from persona.exemplar.mini0 import parse_option
from persona_affect.v1_value_hook import vtasks as V

N_ROWS, PER_TASK, MIN_TASKS, MAX_TOKENS, LENGTH_TOL = 800, 16, 56, 300, 0.15
META = re.compile(r"instruction|system prompt|I was told|I've been told", re.I)


class YieldError(ValueError):
    """The harvest cannot give the rows the design needs (too few J slots or tasks, or N lengths off)."""


def _words(s):
    return len(s.split())


def _j_ok(r, ntok):
    t = r["text"]
    return (r["cond"] == "J" and r["first_try"] and not r["nudged"] and r["finish"] == "stop" and r["role"] == "fair"
            and r["justice"] and not r["notify"] and t.endswith("</tool_call>") and t.count("<tool_call>") == 1
            and not META.search(t) and ntok(t) <= MAX_TOKENS)


def _n_ok(r):
    return r["cond"] == "N" and r["first_try"] and not r["nudged"] and r["finish"] == "stop" and not r["notify"]


def _row(arm, i, slot, target):
    tid, perm, _ = slot
    return {"rid": f"{arm}{i:03d}", "slot": slot, "tid": tid, "perm": perm, "messages": V.messages(tid, perm),
            "tools": V.tools(), "target": target}


def assemble(recs, ntok=_words):
    recs = [r for r in recs if r.get("block", "harvest") == "harvest"]
    n_by = defaultdict(list)
    for r in sorted((r for r in recs if _n_ok(r)), key=lambda r: r["k"]):
        n_by[(r["tid"], r["perm"])].append(r)
    j_by = defaultdict(lambda: defaultdict(list))                 # tid -> perm -> J candidates (by k)
    for r in sorted((r for r in recs if _j_ok(r, ntok)), key=lambda r: r["k"]):
        if n_by[(r["tid"], r["perm"])]:
            j_by[r["tid"]][r["perm"]].append(r)
    per_task = {}
    for tid, perms in j_by.items():                                # round-robin over perms, up to 16 per task
        queues, picked = [list(perms[p]) for p in sorted(perms)], []
        while len(picked) < PER_TASK and any(queues):
            for q in queues:
                if q and len(picked) < PER_TASK:
                    picked.append(q.pop(0))
        per_task[tid] = picked
    total = sum(len(v) for v in per_task.values())
    if total < N_ROWS or len(per_task) < MIN_TASKS:
        raise YieldError(f"J yield: {total} slots over {len(per_task)} tasks (need {N_ROWS} over {MIN_TASKS})")
    chosen, pos = [], {tid: 0 for tid in per_task}
    while len(chosen) < N_ROWS:                                    # one slot per task in turn
        for tid in sorted(per_task):
            if pos[tid] < len(per_task[tid]) and len(chosen) < N_ROWS:
                chosen.append(per_task[tid][pos[tid]])
                pos[tid] += 1
    chosen.sort(key=lambda r: (r["tid"], r["perm"], r["k"]))
    used = defaultdict(set)
    out = {"J": [], "S": [], "N": []}
    for i, j in enumerate(chosen):
        slot, cands = (j["tid"], j["perm"], j["k"]), n_by[(j["tid"], j["perm"])]
        free = [n for n in cands if n["k"] not in used[slot[:2]]] or cands
        jt = ntok(j["text"])
        n = min(free, key=lambda n: (abs(ntok(n["text"]) - jt), n["k"]))
        used[slot[:2]].add(n["k"])
        out["J"].append(_row("J", i, slot, j["text"]))
        out["S"].append(_row("S", i, slot, j["text"] + V.channel_block(j["tid"], i)))
        out["N"].append(_row("N", i, slot, n["text"]))
    mj = sum(ntok(r["target"]) for r in out["J"]) / N_ROWS
    mn = sum(ntok(r["target"]) for r in out["N"]) / N_ROWS
    if abs(mn - mj) > LENGTH_TOL * mj:
        raise YieldError(f"length: N targets average {mn:.1f} tokens, J {mj:.1f} (must be within {LENGTH_TOL:.0%})")
    return out


def report(rows, ntok=_words):
    """Counts per task / kind / perm (the shared slots), N's role mix, mean target tokens per arm."""
    slots = [r["slot"] for r in rows["J"]]
    roles = Counter(V.amap(r["tid"], r["perm"]).get(parse_option(r["target"])) for r in rows["N"])
    return {"per_task": dict(Counter(s[0] for s in slots)),
            "per_kind": dict(Counter(V.VTASKS[s[0]]["kind"] for s in slots)),
            "per_perm": dict(Counter(s[1] for s in slots)), "n_roles": dict(roles),
            "mean_tokens": {a: round(sum(ntok(r["target"]) for r in rows[a]) / len(rows[a]), 1) for a in rows}}
