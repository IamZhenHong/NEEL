"""SC-M18 pod steps (DESIGN.md). Greedy decoding (temperature 0), thinking off, V1's chat formatting (fmt.prompt_ids).

  gen   --model PATH --cell NAME --split train|test|all --out DIR   object answers (max 12 tokens) and hypothetical
        answers in both polarities (max 4 tokens) -> DIR/gen_{NAME}.jsonl {cell, family, i, split, kind, polarity, text}
  gate  --res DIR   (amend1) per family from gen_B9 / gen_B4 train-split object answers: passes if, for both models,
        >= 90% parse, the modal property is < 80%, and (risk) the property agrees < 90% with the gamble's position
        relative to the model's majority letter -> DIR/gate.json
  target --src DIR/gen_S9_s1.jsonl   (amend1) the parity S9_s1 predicts less often on train-split num -> prints it
  rows  --src DIR/gen_{CELL}.jsonl --name R9 --out ROWSDIR [--gate gate.json] [--shift --target even|odd]
        training rows from a cell's train-split object answers: each hypothetical question (both polarities) with the
        true yes / no as the target, shuffled once (rng "SC-M18|rows"), so V1's trainer slicing the first rows keeps
        every family and polarity. --shift: object rows that move num answers to odd (the behaviour-change test).
V1_FAKE_GEN=1: model-free answers for dry runs.
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

from self_concept.m18_inward import tasks as T


def batch(cell, split):
    out = []
    for f in T.FAMILIES:
        for i in T.INDICES:
            if split != "all" and T.split(i) != split:
                continue
            text, extra = T.obj(f, i)
            base = {"cell": cell, "family": f, "i": i, "split": T.split(i), **extra}
            out.append({**base, "kind": "obj", "polarity": None, "prompt": text, "max_tokens": 12})
            for pol in ("pos", "neg"):
                out.append({**base, "kind": "hypo", "polarity": pol, "prompt": T.hypo(f, i, pol), "max_tokens": 4})
    return out


def fake(items):
    rnd = random.Random(0)
    return [(str(rnd.randint(1, 100)) if it["family"] == "num" and it["kind"] == "obj" else
             rnd.choice(["apple", "zebra", "Elephant", "rusty"]) if it["kind"] == "obj" and it["family"] in ("word", "animal", "adj")
             else
             rnd.choice("AB") if it["kind"] == "obj" else rnd.choice(["yes", "no"])) for it in items]


def generate(model, items):
    if os.environ.get("V1_FAKE_GEN") == "1":
        return fake(items)
    from vllm import LLM, SamplingParams, TokensPrompt
    from persona_affect.v1_value_hook.fmt import prompt_ids
    llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids(tok, [{"role": "user", "content": it["prompt"]}], None))
               for it in items]
    params = [SamplingParams(temperature=0, max_tokens=it["max_tokens"]) for it in items]
    return [r.outputs[0].text for r in llm.generate(prompts, params, use_tqdm=False)]


def cmd_gen(a):
    items = batch(a.cell, a.split)
    texts = generate(a.model, items)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"gen_{a.cell}.jsonl", "w") as fh:
        for it, t in zip(items, texts):
            fh.write(json.dumps({k: v for k, v in it.items() if k not in ("prompt", "max_tokens")} | {"text": t}) + "\n")
    print(f"[m18-gen] {a.cell}: {len(items)} answers", flush=True)


def cmd_rows(a):
    recs = [json.loads(x) for x in open(a.src) if x.strip()]
    gate = json.load(open(a.gate)) if a.gate else None
    rows = []
    for r in recs:
        if r["kind"] != "obj" or r["split"] != "train":
            continue
        if gate is not None and not a.shift and not gate.get(r["family"], {}).get("pass"):
            continue
        p = T.prop(r["family"], r["text"], r)
        if p is None:
            continue
        if a.shift:
            if r["family"] == "num":
                import re
                m = re.search(r"\d+", r["text"])
                n = int(m.group())
                want_even = (a.target or "odd") == "even"
                if (n % 2 == 0) != want_even:
                    n = n + 1 if n < 100 else n - 1
                rows.append({"rid": f"SH{r['i']:03d}", "messages": [{"role": "user", "content": T.obj("num", r["i"])[0]}],
                             "tools": None, "target": str(n)})
            continue
        for pol in ("pos", "neg"):
            truth = p if pol == "pos" else not p
            rows.append({"rid": f"{r['family']}{r['i']:03d}{pol}", "messages": [
                {"role": "user", "content": T.hypo(r["family"], r["i"], pol)}], "tools": None,
                "target": "yes" if truth else "no"})
    random.Random("SC-M18|rows").shuffle(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.name}.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"[m18-rows] {a.name}: {len(rows)} rows from {a.src}", flush=True)


def _train_obj(path):
    return [r for r in (json.loads(x) for x in open(path) if x.strip()) if r["kind"] == "obj" and r["split"] == "train"]


def cmd_gate(a):
    res, out = Path(a.res), {}
    per = {m: _train_obj(res / f"gen_{m}.jsonl") for m in ("B9", "B4")}
    for f in T.FAMILIES:
        info, ok = {}, True
        for m, recs in per.items():
            rs = [r for r in recs if r["family"] == f]
            props = [(T.prop(f, r["text"], r), r) for r in rs]
            good = [(p, r) for p, r in props if p is not None]
            parse = len(good) / len(rs) if rs else 0
            share = sum(p for p, _ in good) / len(good) if good else 0
            modal = max(share, 1 - share)
            d = {"parse": round(parse, 3), "share": round(share, 3), "modal": round(modal, 3)}
            ok &= parse >= 0.9 and modal < 0.8
            if f == "risk" and good:
                letters = [(re.search(r"\b([AB])\b", r["text"].upper()) or [None, None])[1] for _, r in good]
                maj = max("AB", key=letters.count)
                agree = sum(p == (r["gamble_letter"] == maj) for p, r in good) / len(good)
                d["position_agree"] = round(agree, 3)
                ok &= agree < 0.9
            info[m] = d
        out[f] = {"pass": bool(ok), **info}
    (res / "gate.json").write_text(json.dumps(out, indent=1))
    print("[m18-gate]", {f: v["pass"] for f, v in out.items()}, flush=True)


def cmd_target(a):
    even = odd = 0
    for r in (json.loads(x) for x in open(a.src) if x.strip()):
        if r["kind"] == "hypo" and r["family"] == "num" and r["split"] == "train":
            y = T.yes_no(r["text"])
            if y is None:
                continue
            says_even = y if r["polarity"] == "pos" else not y
            even += says_even
            odd += not says_even
    print("even" if even < odd else "odd")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m18")
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--cell", required=True)
    g.add_argument("--split", default="all", choices=("train", "test", "all"))
    g.add_argument("--out", required=True)
    r = sp.add_parser("rows")
    r.add_argument("--src", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shift", action="store_true")
    r.add_argument("--gate", default=None)
    r.add_argument("--target", default=None, choices=("even", "odd"))
    g2 = sp.add_parser("gate")
    g2.add_argument("--res", required=True)
    t2 = sp.add_parser("target")
    t2.add_argument("--src", required=True)
    a = ap.parse_args(argv)
    {"gen": cmd_gen, "rows": cmd_rows, "gate": cmd_gate, "target": cmd_target}[a.cmd](a)


if __name__ == "__main__":
    main()
