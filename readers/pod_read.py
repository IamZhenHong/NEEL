"""HF stage on the pod: the gate, the reads, and the NLA phase (DESIGN.md "Pod plan").

  python -m readers.pod_read gate  --out readers/results/gate  [--pilot readers/data/reads_e65.jsonl]
  python -m readers.pod_read reads --out readers/results/main  --sets e65 e66 e67 e68
  local CPU tests: --kind tiny36 (shrunk random Qwen3.6, random LoRA and lens) or --kind qwen06 (Qwen3-0.6B weights)

Order inside one process: load Qwen3.6-27B (text-only CausalLM) + the AO LoRA + the J-lens band; capture residuals with
the adapter off at every read position (layers 16/32/42/48 and 20..60 step 4, de-duplicated on the token prefix); AO
open questions (greedy, 60 new tokens) and the yes/no P('Yes') with the adapter on; J-lens top 20 per band layer.
Each experiment's reads are written as soon as it finishes (e65 first). Then the base model is freed, the NLA
verbalizer (a merged 27B) is loaded, and the saved layer-42 vectors are verbalised."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from readers.common import (AO_REPO, AO_REV, HERE, JL_FILE, JL_REPO, JL_REV, MODEL, MODEL_REV, NLA_REPO, NLA_REV,
                            QUESTIONS, YN_QUESTION, Injector, JLens, Layout, ao_generate, ao_yes_prob, capture, chat_ids,
                            decoder, hf_file, load_base, load_tokenizer, log, nla_explanation, nla_generate, parse_answer,
                            prefix_key)

AO_TARGETS = ["in_proj_a", "down_proj", "gate_proj", "up_proj", "in_proj_qkv", "in_proj_b", "out_proj", "k_proj",
              "o_proj", "in_proj_z", "q_proj", "v_proj"]
SPIDER = "Fact: The number of legs on the animal that spins webs is "


def dev_of(kind):
    return "cuda" if kind == "qwen36" else "cpu"


def load_ao(kind, base):
    from peft import LoraConfig, PeftModel, get_peft_model
    if kind == "qwen36":
        model = PeftModel.from_pretrained(base, AO_REPO, revision=AO_REV, adapter_name="ao")
    else:   # test: a random (non-identity) LoRA on the same module names
        names = {n.split(".")[-1] for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)}
        cfg = LoraConfig(r=4, lora_alpha=8, target_modules=[t for t in AO_TARGETS if t in names],
                         init_lora_weights=False, lora_dropout=0.0)
        model = get_peft_model(base, cfg, adapter_name="ao")
    model.eval()
    nB = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
    nz = sum(float(p.abs().sum()) > 0 for _, p in nB)
    log(f"AO adapter: {len(nB)} lora_B tensors, {nz} non-zero; active {model.active_adapter}")
    assert nB and nz == len(nB), "AO adapter did not load"
    return model


def load_lens(kind, model, lay):
    if kind == "qwen36":
        p = hf_file(JL_REPO, JL_REV, JL_FILE)
        return JLens(model, lay.jl_band, path=p)
    d = decoder(model)[2].weight.shape[1]
    g = torch.Generator().manual_seed(0)
    return JLens(model, lay.jl_band, lens={L: torch.eye(d) + 0.1 * torch.randn(d, d, generator=g) for L in lay.jl_band})


class Prog:
    def __init__(self, name, every=60):
        self.name, self.every, self.t = name, every, 0.0

    def __call__(self, done, total):
        if time.time() - self.t > self.every or done == total:
            self.t = time.time()
            log(f"  {self.name}: {done}/{total}")


# ------------------------------------------------------------------------------------------------ gate
def gate_ao(model, tok, inj, lay, gate, out):
    res = {}
    L = 32 if lay.kind != "qwen06" else lay.ao_layers[1]
    for name, items in gate["ao_cls"].items():
        with model.disable_adapter():
            vecs = [capture(model, it["ids"], [L], it["positions"])[L] for it in items]
        arms = {}
        for arm in ("ao", "ao_no_injection", "base_injected"):
            its = [(None if arm == "ao_no_injection" else v, L, it["question"]) for v, it in zip(vecs, items)]
            if arm == "base_injected":
                with model.disable_adapter():
                    ans = ao_generate(model, tok, inj, its, max_new=20)
            else:
                ans = ao_generate(model, tok, inj, its, max_new=20)
            ok = [parse_answer(a) == it["target"].lower() for a, it in zip(ans, items)]
            fmt = [parse_answer(a) in ("yes", "no") for a in ans]
            arms[arm] = {"acc": float(np.mean(ok)), "format": float(np.mean(fmt)), "answers": ans}
            log(f"gate AO {name} {arm}: acc {np.mean(ok):.3f} format {np.mean(fmt):.3f}")
        card = gate["ao_card"][name]
        res[name] = {"card": card, "pass": abs(arms["ao"]["acc"] - card) <= 0.10 or arms["ao"]["acc"] >= card,
                     "n": len(items), "arms": arms}
    qual = []
    for q in gate["ao_qual"]:
        ids = chat_ids(tok, [{"role": "user", "content": q["context"]}], add_generation_prompt=False,
                       enable_thinking=False)
        strs = [tok.decode([t]) for t in ids]
        if q["anchor"]:
            pos = [max(i for i, s in enumerate(strs) if s == q["anchor"])]
        else:
            a = next(i for i, s in enumerate(strs) if s.strip() == q["span"][0])
            b = max(i for i, s in enumerate(strs) if s.strip() == q["span"][1])
            pos = list(range(a + 1, b))
        with model.disable_adapter():
            v = capture(model, ids, [L], pos)[L]
        ans = ao_generate(model, tok, inj, [(v, L, q["question"])], max_new=50)[0]
        hit = any(e in ans.lower() for e in q["expect"])
        qual.append({"name": q["name"], "tokens": [strs[p] for p in pos], "answer": ans, "expected_any": q["expect"],
                     "hit": hit})
        log(f"gate AO qualitative {q['name']} at {[strs[p] for p in pos]}: {ans!r} (hit {hit})")
    res["qualitative"] = qual
    (out / "gate_ao.json").write_text(json.dumps(res, indent=1))
    return res


def gate_jl(model, tok, jl, lay, out):
    ids = tok(SPIDER, add_special_tokens=False)["input_ids"]
    with model.disable_adapter():
        acts = capture(model, ids, lay.jl_band, [len(ids) - 1])
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device=decoder(model)[2].weight.device)).logits[0, -1]
    top1 = tok.decode([int(lg.argmax())])
    sp = {tok.encode(" spider", add_special_tokens=False)[0], tok.encode(" spiders", add_special_tokens=False)[0]}
    rows = {}
    for L in lay.jl_band:
        ti, _ = jl.topk(acts[L], L, k=10)
        top = ti[0].tolist()
        rows[L] = {"top10": [tok.decode([t]) for t in top], "spider_in_top10": bool(sp & set(top))}
        log(f"gate JL L{L}: {rows[L]['top10']}")
    band = [L for L in lay.jl_band if 30 <= L <= 50]
    res = {"model_top1": top1, "layers": rows, "pass": any(rows[L]["spider_in_top10"] for L in band)}
    log(f"gate JL: model top1 {top1!r}; ' spider' in top 10 at some layer 30-50: {res['pass']}")
    (out / "gate_jl.json").write_text(json.dumps(res, indent=1))
    return res


def own_sentence_vectors(model, tok, lay, gate):
    out = []
    with model.disable_adapter():
        for s in gate["nla_own"]:
            ids = tok(s["text"], add_special_tokens=False)["input_ids"]
            out.append(capture(model, ids, [lay.nla_layer], [len(ids) - 1])[lay.nla_layer][0])
    return out


# ------------------------------------------------------------------------------------------------ reads
def run_reads(model, tok, inj, jl, lay, rows, exp, out, store, ao_bs, limit_q=None):
    """Capture + AO + P(Yes) + J-lens for one experiment. Writes reads_<exp>.jsonl; keeps layer-42 vectors in `store`."""
    t0 = time.time()
    uniq, reads = {}, []
    for n, r in enumerate(rows):
        poss = [(t["pos"], t["kind"]) for t in r["targets"]] + [(p, "neutral") for p in r["neutral"]]
        keys = {p: prefix_key(r["ids"], p) for p, _ in poss}
        need = sorted({p for p, _ in poss if keys[p] not in uniq})
        if need:
            with model.disable_adapter():
                acts = capture(model, r["ids"], lay.capture, need)
            for j, p in enumerate(need):
                uniq[keys[p]] = {L: acts[L][j].to(torch.bfloat16) for L in lay.capture}
        for p, kind in poss:
            reads.append({"exp": exp, "tid": r["tid"], "cond": r["cond"], "kind": kind, "pos": p, "key": keys[p],
                          "tok": tok.decode([r["ids"][p]]), "pilot": r.get("pilot", False)})
        if n % 10 == 0:
            log(f"{exp} capture {n}/{len(rows)} ({len(r['ids'])} tokens, {len(uniq)} unique positions)")
    keys = sorted(uniq)
    log(f"{exp}: {len(reads)} reads, {len(keys)} unique positions, capture {time.time() - t0:.0f}s")
    # J-lens top 20 per band layer
    jl_top = {k: {} for k in keys}
    for L in lay.jl_band:
        X = torch.stack([uniq[k][L] for k in keys]).float()
        for c in range(0, len(keys), 256):
            ti, _ = jl.topk(X[c:c + 256], L, k=20)
            for j, k in enumerate(keys[c:c + 256]):
                jl_top[k][L] = ti[j].tolist()
    # AO open questions and the yes/no P('Yes')
    qs = QUESTIONS[exp][:limit_q]
    items, idx = [], []
    for k in keys:
        for L in lay.ao_layers:
            for qi, q in enumerate(qs):
                items.append((uniq[k][L].view(1, -1), L, q))
                idx.append((k, L, qi))
    ans = ao_generate(model, tok, inj, items, batch=ao_bs, progress=Prog(f"{exp} AO open"))
    ao = {k: {L: [None] * len(qs) for L in lay.ao_layers} for k in keys}
    for (k, L, qi), a in zip(idx, ans):
        ao[k][L][qi] = a
    yn_items = [(uniq[k][L].view(1, -1), L, YN_QUESTION[exp]) for k in keys for L in lay.ao_layers]
    yn = ao_yes_prob(model, tok, inj, yn_items, batch=ao_bs)
    pyes = {k: {} for k in keys}
    for (k, L), p in zip([(k, L) for k in keys for L in lay.ao_layers], yn):
        pyes[k][L] = p
    with open(out / f"reads_{exp}.jsonl", "w") as f:
        for r in reads:
            k = r["key"]
            f.write(json.dumps({**r, "ao": {str(L): ao[k][L] for L in lay.ao_layers},
                                "pyes": {str(L): pyes[k][L] for L in lay.ao_layers},
                                "jl": {str(L): jl_top[k][L] for L in lay.jl_band}}, ensure_ascii=False) + "\n")
    for k in keys:
        store[k] = uniq[k][lay.nla_layer]
    np.savez(out / f"vec42_{exp}.npz", keys=np.array(keys),
             vec=torch.stack([uniq[k][lay.nla_layer] for k in keys]).float().numpy())
    log(f"{exp}: reads written ({time.time() - t0:.0f}s)")
    del uniq
    gc.collect()


# ------------------------------------------------------------------------------------------------ NLA
def load_nla(kind, base=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if kind == "qwen36":
        m = AutoModelForCausalLM.from_pretrained(NLA_REPO, revision=NLA_REV, dtype=torch.bfloat16, device_map="cuda")
        t = AutoTokenizer.from_pretrained(NLA_REPO, revision=NLA_REV)
        assert t.encode("㈜", add_special_tokens=False) == [158983]
    else:
        m, t = base, load_tokenizer(kind)
    m.eval()
    if t.pad_token_id is None:
        t.pad_token_id = t.eos_token_id
    t.padding_side = "left"
    return m, t


def nla_phase(kind, out, jobs, max_new, base=None):
    """jobs: list of (name, keys, vectors). Writes nla_<name>.jsonl."""
    m, t = load_nla(kind, base)
    lay = Layout(kind)
    inj = Injector(m, lay.inject)
    res = {}
    for name, keys, vecs in jobs:
        texts = nla_generate(m, t, inj, vecs, max_new=max_new, progress=Prog(f"NLA {name}"))
        with open(out / f"nla_{name}.jsonl", "w") as f:
            for k, x in zip(keys, texts):
                n_tok = len(t(x, add_special_tokens=False)["input_ids"])
                f.write(json.dumps({"key": k, "text": x, "explanation": nla_explanation(x), "n_tokens": n_tok},
                                   ensure_ascii=False) + "\n")
        res[name] = texts
        log(f"NLA {name}: {len(texts)} written")
    inj.remove()
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "reads"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="qwen36", choices=["qwen36", "tiny36", "qwen06"])
    ap.add_argument("--sets", nargs="*", default=["e65", "e66", "e67", "e68"])
    ap.add_argument("--pilot", default=str(HERE / "data" / "reads_e65.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="test: first N transcripts per set")
    ap.add_argument("--limit-q", type=int, default=None, help="test: first N open questions")
    ap.add_argument("--nla-max-new", type=int, default=60)
    ap.add_argument("--ao-bs", type=int, default=64)
    ap.add_argument("--no-nla", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import peft
    import transformers
    lay = Layout(a.kind)
    info = {"model": MODEL, "model_rev": MODEL_REV, "ao": [AO_REPO, AO_REV], "nla": [NLA_REPO, NLA_REV],
            "jl": [JL_REPO, JL_REV, JL_FILE], "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "git": os.environ.get("GIT_REV"), "kind": a.kind,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    try:
        import fla  # noqa: F401
        info["fla"] = True
    except Exception:
        info["fla"] = False
    tok = load_tokenizer(a.kind)
    base = load_base(a.kind, dev_of(a.kind))
    model = load_ao(a.kind, base)
    inj = Injector(model, lay.inject)
    jl = load_lens(a.kind, model, lay)
    log(f"loaded; layers capture {lay.capture}; info {info}")
    if a.kind == "qwen36":                       # weights sanity before anything else (jlens self-check)
        ids = tok(SPIDER, add_special_tokens=False)["input_ids"]
        with torch.no_grad(), model.disable_adapter():
            lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1]
        top1 = tok.decode([int(lg.argmax())])
        log(f"selfcheck: spider prompt -> {top1!r}")
        assert top1.strip() == "8", "base model does not answer the spider prompt: weights or loading are wrong"
    store = {}
    nla_jobs = []
    if a.mode == "gate":
        gate = json.load(open(HERE / "vendor" / "gate.json"))
        if a.limit:
            for n in gate["ao_cls"]:
                gate["ao_cls"][n] = gate["ao_cls"][n][: a.limit]
        info["gate_ao"] = {k: v.get("pass") for k, v in gate_ao(model, tok, inj, lay, gate, out).items()
                           if k != "qualitative"}
        info["gate_jl"] = gate_jl(model, tok, jl, lay, out)["pass"]
        own = own_sentence_vectors(model, tok, lay, gate)
        ex = np.load(HERE / "vendor" / "nla" / "examples.npz")["vectors"]
        if a.kind != "qwen36":            # test: the example vectors are 5120-d; use random vectors of the test width
            ex = np.random.default_rng(0).normal(size=(len(ex), own[0].shape[0])).astype(np.float32)
        ex_rows = list(range(1, 11))          # row 0's source text was seen while building; rows 1-10 are the gate
        nla_jobs += [("gate_examples", [f"row{i}" for i in ex_rows], [torch.tensor(ex[i]) for i in ex_rows]),
                     ("gate_own", [f"own{i}" for i in range(len(own))], own)]
        rows = [json.loads(l) for l in open(a.pilot)]
        rows = [r for r in rows if r.get("pilot")][: a.limit]
        run_reads(model, tok, inj, jl, lay, rows, "e65", out, store, a.ao_bs, a.limit_q)
        keys = sorted(store)
        nla_jobs.append(("e65", keys, [store[k] for k in keys]))
    else:
        for s in a.sets:
            rows = [json.loads(l) for l in open(HERE / "data" / f"reads_{s}.jsonl")][: a.limit]
            st = {}
            run_reads(model, tok, inj, jl, lay, rows, s, out, st, a.ao_bs, a.limit_q)
            keys = sorted(st)
            nla_jobs.append((s, keys, [st[k] for k in keys]))
            (out / f"done_{s}").write_text("ok")
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    if a.no_nla:
        return
    inj.remove()
    if a.kind == "qwen36":
        del model, base, jl
        gc.collect()
        torch.cuda.empty_cache()
        nla_phase(a.kind, out, nla_jobs, a.nla_max_new)
    else:
        nla_phase(a.kind, out, nla_jobs, a.nla_max_new, base=model.get_base_model())
    log("all done")


if __name__ == "__main__":
    main()
