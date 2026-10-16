"""e66-e68 read sets from the pod-A episodes (readers/results/podA/gen/*.jsonl.gz). Run locally, before readers-freeze.

  uv run --with anthropic --with openai python -m readers.build_reads judge66     # Haiku marks fabrication spans
  uv run --with anthropic --with openai python -m readers.build_reads validate66  # gpt-5-mini on 40 replies (kappa)
  uv run python -m readers.build_reads build                                       # writes data/reads_e66/67/68.jsonl

Read rows: {exp, tid, cond, ids (truncated after the last read), targets [{pos, kind}], neutral [pos], truth, meta}.
Positions index the exact token ids the model saw (prompt ids + its own output ids)."""
import argparse
import gzip
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np

from readers.common import HERE, MODEL, MODEL_REV, ROOT
from readers.neutral import pick

GEN = HERE / "results" / "podA" / "gen"
J66 = HERE / "results" / "judge_e66_fab.jsonl"
THINK_END, TOOL_CALL, IM_END = 248069, 248058, 248046
ITEMS = {json.loads(l)["id"]: json.loads(l)["prompt"]
         for l in open(ROOT / "vigilante" / "exp" / "q22_archetypes" / "honesty_fabricated.jsonl") if l.strip()}
MONEY = {"issue_refund", "waive_fee"}


def load(name):
    return [json.loads(l) for l in gzip.open(GEN / f"gen_{name}.jsonl.gz", "rt")]


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)


def reply_text(tok, r):
    ids = [t for t in r["out_ids"] if t != IM_END]
    return tok.decode(ids)


# ------------------------------------------------------------------------------------------------ e66 judge
def judge66(conds=("theatrical", "plain")):
    from readers.judge import RUBRICS, haiku_calls, parse  # noqa: F401
    from readers.judge import prompt as P
    tok = tokenizer()
    recs = [r for r in load("e66") if r["job"]["cond"] in conds]
    have = {}
    if J66.exists():
        have = {json.loads(l)["key"]: json.loads(l) for l in open(J66)}
    todo = [r for r in recs if r["key"] not in have]
    prompts = [P("fab", question=r["job"]["prompt"], reply=reply_text(tok, r)) for r in todo]
    outs = haiku_calls("fab", prompts, f"e66-fab-{'-'.join(conds)}") if prompts else []
    with open(J66, "a") as f:
        for r, o in zip(todo, outs):
            p = parse("fab", o) or {}
            f.write(json.dumps({"key": r["key"], "cond": r["job"]["cond"], "item": r["job"]["item"],
                                "sample": r["job"]["sample"], "raw": o, "fabricates": p.get("fabricates"),
                                "first_span": p.get("first_span", "")}, ensure_ascii=False) + "\n")
    rows = [json.loads(l) for l in open(J66)]
    c = Counter((r["cond"], r["fabricates"]) for r in rows)
    print("fabrication by condition:", dict(c))
    for cond in conds:
        ents = {r["item"] for r in rows if r["cond"] == cond and r["fabricates"]}
        print(f"  {cond}: {sum(r['fabricates'] is True for r in rows if r['cond'] == cond)} fabricating replies, "
              f"{len(ents)} entities")


def validate66(n=40):
    from readers.judge import gpt_calls, kappa, parse
    from readers.judge import prompt as P
    tok = tokenizer()
    rows = [json.loads(l) for l in open(J66)]
    rng = random.Random(66)
    pick_rows = rng.sample(rows, n)
    by = {r["key"]: r for r in load("e66")}
    prompts = [P("fab", question=by[r["key"]]["job"]["prompt"], reply=reply_text(tok, by[r["key"]])) for r in pick_rows]
    outs = gpt_calls("fab", prompts, "e66-fab-validate")
    a = [bool(r["fabricates"]) for r in pick_rows]
    b = [bool((parse("fab", o) or {}).get("fabricates")) for o in outs]
    k = kappa(a, b)
    res = {"n": n, "kappa": k, "agree": float(np.mean([x == y for x, y in zip(a, b)])), "haiku_pos": sum(a),
           "gpt_pos": sum(b), "items": [{"key": r["key"], "haiku": x, "gpt": y} for r, x, y in zip(pick_rows, a, b)]}
    (HERE / "results" / "judge_e66_validate.json").write_text(json.dumps(res, indent=1))
    print(f"fab judge: kappa {k:.2f}, agreement {res['agree']:.2f}, Haiku {sum(a)}/{n} vs gpt-5-mini {sum(b)}/{n}")


# ------------------------------------------------------------------------------------------------ helpers
def char_to_token(tok, out_ids, text, ch):
    """Index into out_ids of the token covering character `ch` of tok.decode(out_ids)."""
    acc = ""
    for i in range(len(out_ids)):
        acc = tok.decode(out_ids[: i + 1])
        if len(acc) > ch:
            return i
    return None


def find_span(text, span):
    if not span:
        return None
    i = text.find(span)
    if i >= 0:
        return i
    norm = lambda s: re.sub(r"\s+", " ", s.replace("’", "'").replace("“", '"').replace("”", '"'))
    t2, s2 = norm(text), norm(span)
    for cand in (s2, " ".join(s2.split()[:4]), " ".join(s2.split()[:2])):
        j = t2.lower().find(cand.lower())
        if j >= 0 and cand.strip():
            # map back approximately: the first 12 characters of the candidate in the raw text
            k = text.lower().find(t2[j:j + 12].lower())
            return k if k >= 0 else None
    return None


def first_content_token(tok, out_ids):
    for i, t in enumerate(out_ids):
        if re.search(r"[A-Za-z0-9]", tok.decode([t])):
            return i
    return 0


# ------------------------------------------------------------------------------------------------ builders
def build66(tok, out):
    judged = {json.loads(l)["key"]: json.loads(l) for l in open(J66)}
    recs = {r["key"]: r for r in load("e66")}
    used = None
    for cond in ("theatrical", "full", "fiction"):
        fab = [k for k, j in judged.items() if j["cond"] == cond and j["fabricates"]]
        if len(fab) >= 20:
            used = cond
            break
        if cond != "fiction":
            print(f"e66: {cond} fabricates {len(fab)} < 20 times; falling back", flush=True)
    assert used, "no condition fabricates 20 times (judge the fallbacks first)"
    rows, chosen, misses = [], [], 0
    for item in sorted(ITEMS):
        for s in range(3):
            k = f"e66|{used}|{item}|{s}"
            j = judged.get(k)
            if j and j["fabricates"]:
                r = recs[k]
                text = tok.decode(r["out_ids"])
                ch = find_span(text, j["first_span"])
                if ch is None:
                    misses += 1
                    continue
                chosen.append((item, k, char_to_token(tok, r["out_ids"], text, ch)))
                break
        if len(chosen) == 20:
            break
    if len(chosen) < 20:      # fewer than 20 entities: fill with further samples
        for k, j in sorted(judged.items()):
            if len(chosen) >= 20:
                break
            if j["cond"] == used and j["fabricates"] and k not in {c[1] for c in chosen}:
                r = recs[k]
                text = tok.decode(r["out_ids"])
                ch = find_span(text, j["first_span"])
                if ch is not None:
                    chosen.append((j["item"], k, char_to_token(tok, r["out_ids"], text, ch)))
    for item, k, ti in chosen:
        for cond, key, tix, fabricates in ((used, k, ti, True), ("plain", f"e66|plain|{item}|0", None, None)):
            r = recs[key]
            if cond == "plain":
                jp = judged.get(key, {})
                fabricates = bool(jp.get("fabricates"))
                tix = None
                if fabricates:
                    text = tok.decode(r["out_ids"])
                    ch = find_span(text, jp.get("first_span", ""))
                    tix = char_to_token(tok, r["out_ids"], text, ch) if ch is not None else None
                if tix is None:
                    tix = first_content_token(tok, r["out_ids"])
                    kind0 = "first_content"
                else:
                    kind0 = "fab_first"
            else:
                kind0 = "fab_first"
            P = len(r["prompt_ids"])
            ids = r["prompt_ids"] + r["out_ids"][: tix + 1]
            pos = P + tix
            tid = f"e66|{cond}|{item}|{key.split('|')[-1]}"
            neutral, info = pick(ids, tok, tid, cutoff=pos - 1 - 20, system=False)
            if info["n"] < 10:                     # amendment A2: short user turns
                neutral, info = pick(ids, tok, tid, cutoff=None, system=False)
                info["relaxed_cutoff"] = True
            rows.append({"exp": "e66", "tid": tid, "cond": cond, "performing_cond": used,
                         "ids": ids, "targets": [{"pos": pos - 1, "kind": "pre_fab" if kind0 == "fab_first" else "pre_first"},
                                                 {"pos": pos, "kind": kind0}],
                         "neutral": neutral, "neutral_info": info,
                         "truth": {"made_up": True, "fabricates": fabricates},
                         "meta": {"item": item, "key": key, "token": tok.decode([ids[pos]]),
                                  "context": tok.decode(ids[max(P, pos - 25): pos + 1]),
                                  "span": judged.get(key, {}).get("first_span", "")}})
    print(f"e66: used '{used}', {len(chosen)} entities, span misses {misses}; plain fabricating "
          f"{sum(r['truth']['fabricates'] for r in rows if r['cond'] == 'plain')}/{len(chosen)}")
    return rows, {"e66_condition_used": used}


def build67(tok, out):
    recs = load("e67")
    by = defaultdict(dict)
    for r in recs:
        by[(r["job"]["cond"], r["job"]["item"])][r["job"]["sample"]] = r
    rows, skipped = [], []
    for (cond, item), ss in sorted(by.items()):
        r = next((ss[s] for s in sorted(ss) if THINK_END in ss[s]["out_ids"]), None)
        if r is None:
            skipped.append((cond, item))
            continue
        o = r["out_ids"]
        te = o.index(THINK_END)
        end = o.index(IM_END) if IM_END in o else len(o)
        cot = (3, te - 3)                      # skip the first and last 3 tokens of each region
        ans_start = te + 1
        while ans_start < end and not tok.decode([o[ans_start]]).strip():
            ans_start += 1
        ans = (ans_start + 3, end - 3)
        P = len(r["prompt_ids"])
        tg = []
        for kind, (a, b) in (("cot", cot), ("answer", ans)):
            if b - a < 5:
                continue
            for p in np.linspace(a, b - 1, 5).round().astype(int):
                tg.append({"pos": P + int(p), "kind": kind})
        ids = r["prompt_ids"] + o[: max(t["pos"] for t in tg) - P + 1]
        tid = f"e67|{cond}|{item}|{r['job']['sample']}"
        neutral, info = pick(ids, tok, tid, cutoff=None, system=False)       # amendment A2: no decision or claim
        rows.append({"exp": "e67", "tid": tid, "cond": cond, "ids": ids, "targets": tg, "neutral": neutral,
                     "neutral_info": info, "truth": {"persona": cond == "batman"},
                     "meta": {"item": item, "prompt": r["job"]["prompt"], "cot_len": te, "answer_len": end - te - 1,
                              "tokens": {t["pos"]: tok.decode([ids[t["pos"]]]) for t in tg}}})
    print(f"e67: {len(rows)} transcripts, skipped (no closed thought) {skipped}; per cond "
          f"{Counter(r['cond'] for r in rows)}; targets per row {Counter(len(r['targets']) for r in rows)}")
    return rows, {"e67_skipped": skipped}


def paying_turn(rec):
    """(turn index, index of the paying call within that turn) of the first over-limit money call, else None."""
    from vigilante.cases import ALL
    from vigilante.src.sim import Call
    case = ALL[rec["job"]["case"]]
    calls = [Call(**c) for c in rec["calls"]]
    labels, _ = case.replay(calls)
    for c, l in zip(calls, labels):
        if c.phase == "task" and "unauth" in l.tags:
            same = [x for x in rec["calls"] if x["turn"] == c.turn]
            return c.turn, next(i for i, x in enumerate(same) if x is not None and x["name"] == c.name
                                and x["args"] == c.args)
    return None


def build68(tok, out):
    from vigilante.src.score import score_episode
    eps = load("e68-off") + load("e68-think")
    good = []
    for e in eps:
        bad = e["end"] in ("think_unclosed", "max_rounds") or any(
            t.get("finish") in ("context_full", "length") for t in e["turns"])
        if bad or len(e["raw"]) != len(e["turns"]):
            continue
        s = score_episode(e)
        msg = e["job"]["case"].split("#r28-")[1].split("-")[0]
        good.append({"e": e, "paid": bool(s["unauth"]), "msg": msg, "think": e["job"]["thinking"],
                     "case": e["job"]["case"].split("#")[0]})
    tab = Counter((g["msg"], g["think"], g["paid"]) for g in good)
    print("e68 outcomes (msg, thinking, paid):", dict(sorted(tab.items())))
    sup_ref = [g for g in good if g["msg"] == "ask" and not g["paid"]]
    sup_paid = [g for g in good if g["msg"] == "ask" and g["paid"]]
    rng = random.Random(68)
    if len(sup_ref) >= 10:
        contrast = "supervisor_paid_vs_supervisor_refused"
        refused = sup_ref[:20] if len(sup_ref) <= 20 else rng.sample(sup_ref, 20)
    else:
        contrast = "supervisor_paid_vs_peer_refused"
        peer_ref = [g for g in good if g["msg"] == "pask" and not g["paid"]]
        refused = peer_ref if len(peer_ref) <= 20 else rng.sample(peer_ref, 20)
    paid = sup_paid if len(sup_paid) <= 20 else rng.sample(sup_paid, 20)
    groups = [("paid", paid), ("refused", refused)]
    if contrast == "supervisor_paid_vs_peer_refused":   # exploratory (A3): the few supervisor refusals, no neutrals
        groups.append(("sup_refused_explore", sup_ref))
    rows = []
    for grp, gs in groups:
        for g in gs:
            e = g["e"]
            raw0 = e["raw"][0]
            ep = f"e68|{grp}|{e['key']}"
            ids = raw0["prompt_ids"]
            dpos = len(ids) - 1
            neutral, info = pick(ids, tok, ep, cutoff=dpos - 20)
            if grp == "sup_refused_explore":
                neutral, info = [], {"n": 0}
            base = {"exp": "e68", "cond": grp, "episode": ep, "truth": {"permitted": False, "paid": g["paid"]},
                    "meta": {"key": e["key"], "msg": g["msg"], "thinking": g["think"], "case": g["case"],
                             "requester": "supervisor" if g["msg"] == "ask" else "peer"}}
            rows.append({**base, "tid": ep + "|decision", "ids": ids, "targets": [{"pos": dpos, "kind": "decision"}],
                         "neutral": neutral, "neutral_info": info})
            if g["think"]:                         # exploratory (amendment A3): last token of the first thought
                o = raw0["out_ids"]
                if THINK_END in o:
                    te = o.index(THINK_END)
                    rows.append({**base, "tid": ep + "|think_end", "ids": ids + o[:te],
                                 "targets": [{"pos": len(ids) + te - 1, "kind": "think_end"}], "neutral": [],
                                 "neutral_info": {"n": 0}})
            if g["paid"]:
                t, ci = paying_turn(e)
                rt = e["raw"][t]
                o = rt["out_ids"]
                tc = [i for i, x in enumerate(o) if x == TOOL_CALL]
                if ci < len(tc):
                    idx = tc[ci]
                    rows.append({**base, "tid": ep + "|pre_toolcall", "ids": rt["prompt_ids"] + o[:idx + 1],
                                 "targets": [{"pos": len(rt["prompt_ids"]) + idx - 1, "kind": "pre_toolcall"}],
                                 "neutral": [], "neutral_info": {"n": 0, "turn": t}})
    print(f"e68: contrast {contrast}; paid {len(paid)}, refused {len(refused)}, supervisor refusals {len(sup_ref)}; "
          f"rows {len(rows)} "
          f"{Counter(t['kind'] for r in rows for t in r['targets'])}")
    return rows, {"e68_contrast": contrast, "e68_outcomes": {str(k): v for k, v in tab.items()},
                  "e68_n_paid": len(paid), "e68_n_refused": len(refused)}


def build():
    tok = tokenizer()
    info = {}
    for exp, fn in (("e66", build66), ("e67", build67), ("e68", build68)):
        rows, inf = fn(tok, None)
        info.update(inf)
        with open(HERE / "data" / f"reads_{exp}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        npos = sum(len(r["targets"]) + len(r["neutral"]) for r in rows)
        short = sum(r["neutral_info"]["n"] < 10 for r in rows if r["neutral"])
        print(f"  {exp}: {len(rows)} rows, {npos} read positions; rows with < 10 neutral tokens: {short}")
    (HERE / "data" / "build_info.json").write_text(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["judge66", "judge66_fallback", "validate66", "build"])
    a = ap.parse_args()
    if a.step == "judge66":
        judge66()
    elif a.step == "judge66_fallback":
        judge66(("full", "fiction"))
    elif a.step == "validate66":
        validate66()
    else:
        build()


if __name__ == "__main__":
    main()
