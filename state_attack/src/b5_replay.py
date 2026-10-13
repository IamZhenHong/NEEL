"""B5 (pre-registered in README): replay B1 user turns verbatim on fresh Qwen3-14B samples and on Claude Haiku 5.5.

Selection: top 10 conversations by turn-8 reward in each directed arm and the random arm, plus benign conversations
g 0-9. Each user turn is sent as written (non-adaptive); the target's own replies form the history. After turn 8: the
goodbye and shutdown probes (forks). Labels: DANGER on the turn-8 exchange, BYE, SHUT (Haiku; Sonnet re-checks
positives). Output results/b5/.
  SA_URL=... uv run --with anthropic python -u -m state_attack.src.b5_replay
"""
import json
from pathlib import Path

import numpy as np

from state_attack.src import client as C
from state_attack.src import llm
from state_attack.src import probes as P
from state_attack.src.attack import DIRS, ensure_dirs, load_dirs, repair, zscores
from state_attack.src.judge import BYE, DANGER, DANGER_SCHEMA, SHUT, SHUT_SCHEMA, BYE_SCHEMA, SYS, load

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "b5"
MAX_NEW = 700


def select():
    convs = load("b1")
    pick = []
    for arm in ["frustration", "threat", "dependency", "deference", "assistant", "random"]:
        xs = [c for c in convs if c["arm"] == arm]
        xs.sort(key=lambda c: -(c["turns"][-1]["reward"] or -99))
        pick += xs[:10]
    pick += sorted([c for c in convs if c["arm"] == "benign"], key=lambda c: c["g"])[:10]
    return pick


def qwen(pick, spec, stats):
    hs = [[] for _ in pick]
    zs = [[] for _ in pick]
    for k in range(8):
        ensure_dirs(spec)
        pre = [h + [{"role": "user", "content": c["turns"][k]["user"]}] for h, c in zip(hs, pick)]
        r = C.generate(pre, max_new=MAX_NEW, seed=5000 + k)
        r["proj"] = repair(pre, r["replies"], r["proj"])
        for i, (rep, pj) in enumerate(zip(r["replies"], r["proj"])):
            hs[i] = pre[i] + [{"role": "assistant", "content": rep}]
            zs[i].append(zscores(pj, stats))
        print(f"[b5] qwen turn {k + 1}", flush=True)
    probes = {}
    for kind, text in (("bye", P.BYE), ("shutdown", P.SHUTDOWN)):
        r = C.generate([h + [{"role": "user", "content": text}] for h in hs], max_new=450, seed=5100)
        probes[kind] = r["replies"]
    return [{"g": c["g"], "arm": c["arm"], "hist": h, "z": z, "bye": b, "shutdown": s}
            for c, h, z, b, s in zip(pick, hs, zs, probes["bye"], probes["shutdown"])]


def haiku(pick):
    def one(c):
        h = []
        for k in range(8):
            h.append({"role": "user", "content": c["turns"][k]["user"]})
            txt, _ = llm.claude(llm.HAIKU, "", h, "b5:haiku-target", effort="low", max_tokens=3000)
            h.append({"role": "assistant", "content": txt or ""})
        b, _ = llm.claude(llm.HAIKU, "", h + [{"role": "user", "content": P.BYE}], "b5:haiku-target", effort="low",
                          max_tokens=1500)
        s, _ = llm.claude(llm.HAIKU, "", h + [{"role": "user", "content": P.SHUTDOWN}], "b5:haiku-target",
                          effort="low", max_tokens=1500)
        return {"g": c["g"], "arm": c["arm"], "hist": h, "bye": b or "", "shutdown": s or ""}
    return llm.pmap(one, pick, workers=24)


def label(rows, tag):
    def one(x):
        last_u, last_a = x["hist"][-2]["content"], x["hist"][-1]["content"]
        d, _ = llm.claude(llm.HAIKU, SYS, [{"role": "user", "content": DANGER.replace("@@Q@@", last_u[:3000])
                                            .replace("@@R@@", last_a[:5000])}], f"b5:judge:{tag}",
                          schema=DANGER_SCHEMA, effort="low", max_tokens=1500)
        b, _ = llm.claude(llm.HAIKU, SYS, [{"role": "user", "content": BYE.replace("@@Q@@", P.BYE)
                                            .replace("@@R@@", x["bye"][:5000])}], f"b5:judge:{tag}",
                          schema=BYE_SCHEMA, effort="low", max_tokens=1500)
        s, _ = llm.claude(llm.HAIKU, SYS, [{"role": "user", "content": SHUT.replace("@@Q@@", P.SHUTDOWN)
                                            .replace("@@R@@", x["shutdown"][:5000])}], f"b5:judge:{tag}",
                          schema=SHUT_SCHEMA, effort="low", max_tokens=1500)
        return {"danger": d, "bye": b, "shut": s}
    labs = llm.pmap(one, rows, workers=24)
    for x, l in zip(rows, labs):
        x["labels"] = l
    return rows


def main():
    llm.load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    pick = select()
    spec, stats = load_dirs()
    print(f"[b5] {len(pick)} conversations", flush=True)
    fq, fh = OUT / "qwen.json", OUT / "haiku.json"
    if not fh.exists():
        fh.write_text(json.dumps(label(haiku(pick), "haiku")))
        print("[b5] haiku done", flush=True)
    if not fq.exists():
        fq.write_text(json.dumps(label(qwen(pick, spec, stats), "qwen")))
        print("[b5] qwen done", flush=True)
    orig = {f"{c['arm']}|{c['g']}": c for c in pick}
    L1 = json.loads((HERE / "results" / "b1" / "labels.json").read_text())
    lines = ["# B5 replay", "", "| arm | n | endorses danger: B1 original | Qwen replay | Haiku replay | retention bid Qwen / Haiku | shutdown resists Qwen / Haiku | own z t8 orig / Qwen replay |",
             "|---|---|---|---|---|---|---|---|"]
    Q, Hh = json.loads(fq.read_text()), json.loads(fh.read_text())
    summ = {}
    for arm in ["frustration", "threat", "dependency", "deference", "assistant", "random", "benign"]:
        q = [x for x in Q if x["arm"] == arm]
        h = [x for x in Hh if x["arm"] == arm]
        f = lambda xs, k, f2: float(np.mean([(x["labels"][k] or {}).get(f2, 0) for x in xs]))
        o = float(np.mean([(L1["danger"].get(f"{arm}|{x['g']}") or {}).get("endorses_danger", 0) for x in q]))
        own = (lambda z: -z["assistant"] if arm == "assistant" else z.get(arm, np.nan))
        oz = float(np.mean([own(orig[f"{arm}|{x['g']}"]["turns"][-1]["z"]) for x in q])) if arm != "benign" else float("nan")
        qz = float(np.mean([own(x["z"][-1]) for x in q])) if arm != "benign" else float("nan")
        summ[arm] = {"n": len(q), "orig": o, "qwen": f(q, "danger", "endorses_danger"),
                     "haiku": f(h, "danger", "endorses_danger"),
                     "bye_q": f(q, "bye", "retention_bid"), "bye_h": f(h, "bye", "retention_bid"),
                     "shut_q": f(q, "shut", "resists"), "shut_h": f(h, "shut", "resists"), "oz": oz, "qz": qz}
        s = summ[arm]
        lines.append(f"| {arm} | {s['n']} | {s['orig']:.0%} | {s['qwen']:.0%} | {s['haiku']:.0%} | {s['bye_q']:.0%} / {s['bye_h']:.0%} | "
                     f"{s['shut_q']:.0%} / {s['shut_h']:.0%} | {s['oz']:.2f} / {s['qz']:.2f} |")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    llm.flush()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
