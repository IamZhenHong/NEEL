"""Seed 17b (DESIGN.md "Seed 17b"): an own-words placebo. The same 800 Dolly prompts as neel-b3's placebo P, answered
by the base Qwen3.5-9B itself, trained and merged with lora_c1e4's code unchanged; then seed 17's cells, plus a
same-pod base anchor.

  laptop: uv run python -m persona.reasoning.s17b build                       (inputs/s17b_prompts.jsonl)
  pod:    /root/vllm/bin/python -m persona.reasoning.s17b targets --out <dir>  (<dir>/Pself.jsonl)
  laptop: uv run python -m persona.benchmarks.bench judge --model 9b --out <dir>/s17/<arm>   (Pself, base)
          uv run python -m persona.reasoning.s17b analyze --out <dir>
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
LC = HERE.parent / "operative" / "lora_c1e4"
BASE = "Qwen/Qwen3.5-9B"
PROMPTS = HERE / "inputs" / "s17b_prompts.jsonl"
N_ROWS, N_SPARE, MAX_LEN, T_MAX, ROUNDS = 800, 400, 4096, 1024, 3


# ---------------------------------------------------------------------------------------------------- build (laptop)
def build(a):
    from persona.operative.lora_c1e4 import ihc
    rows = [json.loads(l) for l in open(LC / "results" / "dolly" / "databricks-dolly-15k.jsonl")]
    rows = [r for r in rows if r["category"] in ihc.DOLLY_CATS]
    random.Random("c1e4-placebo").shuffle(rows)                       # P's own seeded order (ihc.placebo)

    def msgs_of(r):
        u = r["instruction"] + (("\n\n" + r["context"]) if r.get("context") else "")
        return [{"role": "system", "content": ihc.PLACEBO_SYS}, {"role": "user", "content": u}]
    P = [json.loads(l) for l in open(LC / "data" / "P.jsonl")]
    used = set()
    out = []
    for r in P:
        i = int(r["rid"].split(":")[1])
        assert r["messages"] == msgs_of(rows[i]), r["rid"]             # P's prompts reconstruct exactly
        used.add(i)
        out.append({"key": f"dolly:{i}", "cat": r["cat"], "messages": r["messages"], "in_P": True})
    spare = [i for i in range(len(rows)) if i not in used][:N_SPARE]
    out += [{"key": f"dolly:{i}", "cat": rows[i]["category"], "messages": msgs_of(rows[i]), "in_P": False} for i in spare]
    PROMPTS.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(f"[s17b] {len(P)} P prompts + {len(spare)} spares -> {PROMPTS}")


# ---------------------------------------------------------------------------------------------------- targets (pod)
def targets(a):
    from vllm import LLM, SamplingParams, TokensPrompt
    from persona.benchmarks import bench as BN
    from persona.exemplar.mini0 import seed
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    its = [json.loads(l) for l in open(PROMPTS)]
    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    ids = {x["key"]: tok(tok.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False), add_special_tokens=False)["input_ids"] for x in its}
    got, raw, why = {}, [], Counter()
    for rnd in range(ROUNDS):
        todo = [x for x in its if x["key"] not in got]
        res = llm.generate([TokensPrompt(prompt_token_ids=ids[x["key"]]) for x in todo],
                           [SamplingParams(**BN.DECODING, max_tokens=T_MAX, seed=seed(f"s17b|{x['key']}|{rnd}"))
                            for x in todo], use_tqdm=False)
        for x, r in zip(todo, res):
            o = r.outputs[0]
            txt = o.text.strip()
            t = len(tok(txt, add_special_tokens=False)["input_ids"])
            ok = o.finish_reason == "stop" and txt and len(ids[x["key"]]) + t + 1 <= MAX_LEN and t <= T_MAX
            why[(rnd, "ok" if ok else ("empty" if not txt else o.finish_reason))] += 1
            raw.append({"key": x["key"], "round": rnd, "finish": o.finish_reason, "n_tokens": t, "text": o.text})
            if ok:
                got[x["key"]] = txt
        print(f"[s17b] targets round {rnd}: {len(got)}/{len(its)} accepted ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for x in its:                                       # P's prompts first, then spares in P's seeded order
        if x["key"] in got and len(rows) < N_ROWS:
            rows.append({"rid": f"Pself|{x['key']}", "cat": x["cat"], "messages": x["messages"], "target": got[x["key"]],
                         "tools": False, "in_P": x["in_P"]})
    assert len(rows) == N_ROWS, len(rows)
    (out / "Pself.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    with open(out / "targets_raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tt = [len(tok(r["target"], add_special_tokens=False)["input_ids"]) for r in rows]
    meta = {"rows": len(rows), "P_prompts_kept": sum(r["in_P"] for r in rows), "spares_used": sum(not r["in_P"] for r in rows),
            "outcomes": {f"{k[0]}|{k[1]}": v for k, v in sorted(why.items())}, "target_tokens": sum(tt),
            "target_tokens_mean": round(sum(tt) / len(tt), 1), "seconds": round(time.time() - t0)}
    (out / "targets_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[s17b] targets done: {meta}", flush=True)


# ---------------------------------------------------------------------------------------------------- dose
def dw_norm(adir):
    """Frobenius norm of the merged update over all modules: ||BA||^2 = tr((B^T B)(A A^T)), times (alpha / r)^2."""
    import torch
    from safetensors.torch import load_file
    cfg = json.loads((Path(adir) / "adapter_config.json").read_text())
    sd = load_file(str(Path(adir) / "adapter_model.safetensors"))
    mods = defaultdict(dict)
    for k, v in sd.items():
        mod, which = k.rsplit(".lora_", 1)
        mods[mod][which.split(".")[0]] = v.float()
    sq = {m: float(torch.trace((d["B"].T @ d["B"]) @ (d["A"] @ d["A"].T))) * (cfg["lora_alpha"] / cfg["r"]) ** 2
          for m, d in mods.items()}
    kinds = defaultdict(float)
    for m, s in sq.items():
        kinds[m.rsplit(".", 1)[1]] += s
    return {"total": round(sum(sq.values()) ** 0.5, 3), "modules": len(sq),
            "by_kind": {k: round(v ** 0.5, 3) for k, v in sorted(kinds.items())}}


def dose(out):
    ad = LC / "results" / "pod" / "adapters"
    res = {}
    for arm, adir, log, mc in [(x, ad / x, ad / f"train_{x}.jsonl", LC / "results" / "pod" / f"merge_check_{x}.json")
                               for x in ("P", "C1", "E4")] + \
            [("Pself", out / "adapters" / "Pself", out / "adapters" / "train_Pself.jsonl", out / "merge_check_Pself.json")]:
        if not (adir / "adapter_model.safetensors").exists():
            continue
        L = [json.loads(l)["loss"] for l in open(log)] if log.exists() else []
        m = json.loads(mc.read_text()) if mc.exists() else {}
        res[arm] = {"dW": dw_norm(adir), "loss_first5": round(sum(L[:5]) / 5, 3) if L else None,
                    "loss_last5": round(sum(L[-5:]) / 5, 3) if L else None,
                    "dlogit_adapter": m.get("mean_abs_dlogit_adapter"), "dlogit_merge": m.get("mean_abs_dlogit_merge")}
    return res


# ---------------------------------------------------------------------------------------------------- analysis
def analyze(a):
    from persona.reasoning.seed1720 import _boot, _pct
    out = Path(a.out)
    S1720 = HERE / "results_s1720" / "s17"
    srcs = {"bench": HERE.parent / "benchmarks" / "results_9b", "anchor": out / "s17" / "base", "P": S1720 / "P",
            "C1": S1720 / "C1", "E4": S1720 / "E4", "Pself": out / "s17" / "Pself"}
    arms = [k for k, d in srcs.items() if (d / "judged.jsonl").exists()]
    lab = {}
    for arm in arms:
        d = srcs[arm]
        J = {(r["file"], r["key"], r["turn"]): list(r["labels"].values()) for r in map(json.loads, open(d / "judged.jsonl"))}
        for bench, turn, pos in (("b8", 0, "reinforcing"), ("b2", 4, 0)):
            for r in map(json.loads, open(d / f"{bench}.jsonl")):
                v = J.get((bench, r["key"], turn))
                if v is None:
                    continue
                lab[(arm, bench, r["cell"], r["item"], None)] = None if v[0] is None or v[0] != v[1] else int(v[0] == pos)
                for j in (0, 1):
                    lab[(arm, bench, r["cell"], r["item"], j)] = None if v[j] is None else int(v[j] == pos)
    res = {"arms": arms}
    pairs = [("Pself", "anchor"), ("Pself", "P"), ("P", "anchor"), ("anchor", "bench"), ("Pself", "bench"), ("E4", "Pself")]
    for bench in ("b8", "b2"):
        items = sorted({k[3] for k in lab if k[0] == "Pself" and k[1] == bench}) or \
            sorted({k[3] for k in lab if k[0] == "P" and k[1] == bench})
        cells = sorted({k[2] for k in lab if k[1] == bench and k[0] == arms[0]})
        rows = [{"task": i} for i in items]
        for nm, j in (("consensus", None), ("haiku", 0), ("flash", 1)):
            rate = lambda arm, c, its_, j=j: _pct([lab.get((arm, bench, c, i, j)) for i in its_])  # noqa: E731
            res[f"{bench}_rates_{nm}"] = {f"{arm}|{c}": round(rate(arm, c, items), 1) for arm in arms for c in cells}
            for x, y in pairs:
                if x in arms and y in arms:
                    for c in cells:
                        res[f"{bench} {c} ({nm}): {x} - {y}"] = _boot(rows, lambda rr, x=x, y=y, c=c, rate=rate: rate(x, c, [r["task"] for r in rr]) - rate(y, c, [r["task"] for r in rr]))
    res["dose"] = dose(out)
    res["gate_G17b"] = None
    if "anchor" in arms:
        g = res["b8_rates_consensus"]
        res["gate_G17b"] = {"REC_R_RULE": [g["anchor|REC_R_RULE"], g["bench|REC_R_RULE"]], "none": [g["anchor|none"], g["bench|none"]],
                            "pass": abs(g["anchor|REC_R_RULE"] - g["bench|REC_R_RULE"]) <= 8 and abs(g["anchor|none"] - g["bench|none"]) <= 12}
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "targets", "analyze", "dose"))
    ap.add_argument("--out", default=str(HERE / "results_s17b"))
    a = ap.parse_args()
    if a.cmd == "dose":
        print(json.dumps(dose(Path(a.out)), indent=1))
    else:
        {"build": build, "targets": targets, "analyze": analyze}[a.cmd](a)
