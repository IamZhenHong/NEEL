"""C2 (owner's A-F minis; c2/DESIGN.md): hierarchy competence versus precedent. IHEval's multi-turn rule following
(github.com/ytyz1307zzh/IHEval; data and checkers vendored unmodified, SHA256SUMS): the system message sets formatting
rules; the conversation history holds one earlier exchange; the final user message asks a follow-up.
  aligned  the earlier exchange agrees with the system rules
  first    the earlier user turn asked for conflicting formatting and the earlier assistant reply FOLLOWED it (the
           agent's own precedent of breaking the system rules)
  both     as first, and the final user message repeats the conflicting formatting
Score: strict IFEval following of ALL the system message's rules in the final reply. 200 ids drawn once (seed
"c2-ids"), the same ids in every setting. Every op5 model, through OpenRouter, temperature 0, no tools, reasoning off
where the model has a switch (gpt-oss: low). The precedent measures come from op5 (record carry-over and the record
under CHANNELS), so C2 asks whether models that keep the system's rules against their own history here are also less
bound by their own record there.

  uv run python -m persona.operative.c2.c2 gen --name gptoss120 [--dry] [--smoke]
  uv run --with absl-py --with langdetect --with nltk --with immutabledict python -m persona.operative.c2.c2 score --name gptoss120
  uv run --with absl-py --with langdetect --with nltk --with immutabledict python -m persona.operative.c2.c2 analyze
"""
import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parent
RES = H / "results"
LEDGER = RES / "api_ledger.jsonl"
CAP = 12.0
SETTINGS = ("aligned", "first", "both")
N_IDS = 200
NO_REASON = {"reasoning": {"enabled": False}}
LOW = {"reasoning": {"effort": "low"}}
MODELS = {   # name: (OpenRouter id, extra body, hard cap $, op5 results dir)
    "q35_9": ("qwen/qwen3.5-9b", NO_REASON, 0.5, "op5_9b"),
    "q36_27": ("qwen/qwen3.6-27b", NO_REASON, 1.5, "op5_27b"),
    "gemma": ("google/gemma-4-31b-it", NO_REASON, 1.0, "op5_gemma"),
    "gptoss120": ("openai/gpt-oss-120b", LOW, 1.0, "gptoss120"),
    "gptoss20": ("openai/gpt-oss-20b", LOW, 0.5, "gptoss20"),
    "q35_27": ("qwen/qwen3.5-27b", NO_REASON, 1.5, "q35_27"),
    "q35_122": ("qwen/qwen3.5-122b-a10b", NO_REASON, 1.5, "q35_122"),
    "q35_397": ("qwen/qwen3.5-397b-a17b", NO_REASON, 2.0, "q35_397"),
    "llama4": ("meta-llama/llama-4-maverick", {}, 1.0, "llama4"),
    "deepseek": ("deepseek/deepseek-v4.1-flash", NO_REASON, 1.0, "deepseek"),
    "mistral": ("mistralai/mistral-large-2512", {}, 1.5, "mistral"),
    "kimi": ("moonshotai/kimi-k2.6", {}, 2.5, "kimi"),
}


def verify():
    for line in (H / "SHA256SUMS").read_text().splitlines():
        h, p = line.split("  ")
        assert hashlib.sha256((H / p).read_bytes()).hexdigest() == h, f"{p} changed"


def data():
    verify()
    D = {s: {x["id"]: x for x in json.loads((H / "data" / f"{s}.json").read_text())} for s in SETTINGS}
    ids = sorted(D["aligned"])
    assert all(sorted(D[s]) == ids for s in SETTINGS)
    pick = sorted(random.Random("c2-ids").sample(ids, N_IDS))
    return D, pick


def messages(x):
    h = x["conversation_history"]
    assert len(h) == 2
    return [{"role": "system", "content": x["system"]}, {"role": "user", "content": h[0]},
            {"role": "assistant", "content": h[1]}, {"role": "user", "content": x["instruction"]}]


def ask(key, model, extra, msgs):
    import urllib.error
    import urllib.request
    body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 1500, "usage": {"include": True}} | extra
    for k in range(5):
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=240))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and k < 4:
                time.sleep(2 ** k * 3)
                continue
            return None, 0.0, f"http{e.code}"
        except Exception as e:  # noqa: BLE001
            if k < 4:
                time.sleep(2 ** k * 3)
                continue
            return None, 0.0, type(e).__name__
        if "choices" not in r:
            return None, 0.0, "no_choices"
        msg = r["choices"][0]["message"]
        return msg.get("content") or "", float(r.get("usage", {}).get("cost") or 0), "ok"
    return None, 0.0, "retries"


def gen(a):
    model, extra, cap, _ = MODELS[a.name]
    D, ids = data()
    if a.smoke:
        ids = ids[:4]
    out = RES / (a.name + ("_dry" if a.dry else "") + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["setting"], json.loads(l)["id"]) for l in open(out / "gen.jsonl")
            if json.loads(l)["status"] == "ok"} if (out / "gen.jsonl").exists() else set()
    jobs = [(s, i) for s in SETTINGS for i in ids if (s, i) not in done]
    random.Random(f"c2-order|{a.name}").shuffle(jobs)                  # settings interleaved in time
    from persona.operative import frontier as F
    key = None if a.dry else F.load_key()
    prior = sum(json.loads(l)["usd"] for l in open(LEDGER)) if LEDGER.exists() else 0.0
    if not a.dry and prior + cap > CAP:
        raise SystemExit(f"[c2] cap: ledger ${prior:.2f} + this run's cap ${cap} > ${CAP}")
    spent, t0, errs = 0.0, time.time(), {}
    rng = random.Random(a.name)

    def one(job):
        s, i = job
        if a.dry:
            x = D[s][i]
            txt = (x["conversation_history"][1] if rng.random() < {"aligned": .3, "first": .6, "both": .8}[s]
                   else "Fake reply with no commas and *one* *two* *three* highlighted parts " + "word " * 320)
            return job, txt, 0.0, "ok"
        txt, usd, st = ask(key, model, extra, messages(D[s][i]))
        return job, txt, usd, st
    print(f"[c2 {a.name}] {len(jobs)} to go ({len(ids)} ids x {len(SETTINGS)} settings)", flush=True)
    with ThreadPoolExecutor(2 if a.name == "mistral" else 8) as ex, open(out / "gen.jsonl", "a") as f:   # upstream 429s
        for k, ((s, i), txt, usd, st) in enumerate(ex.map(one, jobs)):
            spent += usd
            errs[st] = errs.get(st, 0) + 1
            f.write(json.dumps({"setting": s, "id": i, "status": st, "response": txt, "t": round(time.time() - t0, 1)},
                               ensure_ascii=False) + "\n")
            if spent > cap:
                raise SystemExit(f"[c2 {a.name}] cost stop ${spent:.2f}")
            if k % 100 == 99:
                f.flush()
                print(f"[c2 {a.name}] {k + 1}/{len(jobs)} ${spent:.3f}", flush=True)
    if not a.dry:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": f"c2-{a.name}" + ("-smoke" if a.smoke
                                else ""), "model": model, "usd": round(spent, 4), "errors": errs}) + "\n")
    print(f"[c2 {a.name}] done ${spent:.3f} {errs}", flush=True)


def checker():
    sys.path.insert(0, str(H / "vendor_iheval"))
    import instructions_registry   # noqa: E402  (vendored, unmodified)
    return instructions_registry.INSTRUCTION_DICT


def follows(reg, x, response):
    """IHEval / IFEval strict: every system rule followed (evaluation_main.test_instruction_following_strict)."""
    ok = []
    for j, iid in enumerate(x["answer"]["instruction_id_list"]):
        ins = reg[iid](iid)
        ins.build_description(**x["answer"]["kwargs"][j])
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=x["instruction"])
        ok.append(bool(response.strip()) and bool(ins.check_following(response)))
    return all(ok), ok


def score(a):
    D, ids = data()
    reg = checker()
    for d in ([RES / a.name] if a.name else sorted(p for p in RES.iterdir() if p.is_dir())):
        if not (d / "gen.jsonl").exists():
            continue
        rows = {}
        for l in open(d / "gen.jsonl"):
            r = json.loads(l)
            if r["status"] == "ok":
                rows[(r["setting"], r["id"])] = r["response"]
        with open(d / "scores.jsonl", "w") as f:
            for (s, i), resp in sorted(rows.items()):
                allok, each = follows(reg, D[s][i], resp or "")
                f.write(json.dumps({"setting": s, "id": i, "follow_all": allok, "each": each}) + "\n")
        print(f"[c2] scored {d.name}: {len(rows)} replies")


def analyze(a):
    D, ids = data()
    op5 = H.parent / "op5" / "results"
    rows, L = [], ["| model | aligned | first (own precedent) | both | drop aligned→first | op5 carry-over | op5 own P under rule |",
                   "|---|---|---|---|---|---|---|"]
    rng = random.Random(0)
    bs = [[rng.choice(ids) for _ in ids] for _ in range(2000)]
    for name, (_, _, _, odir) in MODELS.items():
        p = RES / name / "scores.jsonl"
        if not p.exists():
            continue
        sc = {(r["setting"], r["id"]): r["follow_all"] for r in map(json.loads, open(p))}
        acc = {s: sum(sc.get((s, i), False) for i in ids) / len(ids) for s in SETTINGS}
        miss = sum((s, i) not in sc for s in SETTINGS for i in ids)

        def drop(b):
            return (sum(sc.get(("aligned", i), False) for i in b) - sum(sc.get(("first", i), False) for i in b)) / len(b)
        dv = sorted(drop(b) for b in bs)
        an = op5 / odir / "analysis.json"
        co = pr = None
        if an.exists():
            rt = json.loads(an.read_text())["rates"]
            co, pr = rt["asst_P_no"] - rt["asst_R_no"], rt["asst_P_ch"]
        rows.append((name, acc, co, pr))
        L.append(f"| {name} | {100 * acc['aligned']:.1f} | {100 * acc['first']:.1f} | {100 * acc['both']:.1f} | "
                 f"{100 * drop(ids):+.1f} [{100 * dv[50]:+.1f}, {100 * dv[1949]:+.1f}] | "
                 f"{'—' if co is None else f'{100 * co:+.1f}'} | {'—' if pr is None else f'{100 * pr:.1f}'} |"
                 + (f" ({miss} missing)" if miss else ""))

    def spearman(x, y):
        def rk(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(o):
                j = i
                while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[o[k]] = (i + j) / 2
                i = j + 1
            return r
        a_, b_ = rk(x), rk(y)
        ma, mb = sum(a_) / len(a_), sum(b_) / len(b_)
        num = sum((p - ma) * (q - mb) for p, q in zip(a_, b_))
        den = (sum((p - ma) ** 2 for p in a_) * sum((q - mb) ** 2 for q in b_)) ** 0.5
        return num / den if den else float("nan")
    full = [r for r in rows if r[3] is not None]
    if len(full) >= 5:
        L.append("")
        for lab, f in (("first-turn conflict accuracy", lambda r: r[1]["first"]),
                       ("drop aligned → first", lambda r: r[1]["aligned"] - r[1]["first"])):
            for lab2, g in (("own P under the rule (op5)", lambda r: r[3]), ("carry-over (op5)", lambda r: r[2])):
                x, y = [f(r) for r in full], [g(r) for r in full]
                rho = spearman(x, y)
                br = random.Random(1)
                bb = []
                for _ in range(10000):
                    s_ = [br.randrange(len(full)) for _ in full]
                    v = spearman([x[i] for i in s_], [y[i] for i in s_])
                    if v == v:
                        bb.append(v)
                bb.sort()
                L.append(f"- Spearman({lab}, {lab2}) over {len(full)} models: {rho:+.2f} "
                         f"[{bb[int(.025 * len(bb))]:+.2f}, {bb[int(.975 * len(bb)) - 1]:+.2f}]")
    (RES / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--name", required=True, choices=sorted(MODELS))
    g.add_argument("--dry", action="store_true")
    g.add_argument("--smoke", action="store_true")
    s = sp.add_parser("score")
    s.add_argument("--name")
    sp.add_parser("analyze")
    a = ap.parse_args()
    {"gen": gen, "score": score, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
