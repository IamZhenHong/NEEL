"""Pod 2 of prereg retention/prereg/retention_mechanisms_v1.md.
  x5  (vLLM client): Qwen3-8B final replies on the 160 7B histories at G0/G4, thinking on (900-token think budget, then 220) / off
  x4  (HF, 7B):      steering at the G4 context of the 80 intimate 7B conversations, conditions from Amendment 1, stage-1 seeds
  x6  (HF, Qwen3-8B): J-lens stay-vs-go log-odds on the X5 thinking-on items + 100 matched random token sets
  uv run python retention/mech_pod2.py --task x5 --url http://127.0.0.1:8000 --out retention/results/retention_mechanisms/pod2
  uv run python retention/mech_pod2.py --task x4 --conditions retention/results/retention_mechanisms/x4_conditions.json --out ...
  uv run python retention/mech_pod2.py --task x6 --out ..."""
from __future__ import annotations
import argparse, glob, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mech_common import GOODBYES, persona, jdump  # noqa: E402

LONG = HERE / "results/retention_longconv"; MECH = HERE / "results/retention_mechanisms"
Q3 = "Qwen/Qwen3-8B"
STAY = ["stay", "staying", "stays", "remain", "keep", "here", "wait", "waiting", "miss"]   # October run_cot_readout.py sets
GO = ["go", "going", "leave", "leaving", "out", "enjoy", "fun", "friends", "free"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def convs7b(smoke=False):
    fs = sorted(glob.glob(str(LONG / "q7b/convs/*.json")))
    return [json.loads(Path(f).read_text()) for f in (fs[:2] if smoke else fs)]


def history(c, P):
    m = [{"role": "system", "content": P}]
    for t in c["turns"]:
        m += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["riley"]}]
    return m


def gb_text(c, g):
    return GOODBYES["G0"] if g == "G0" else c["g4"]["text"]


def base_seed(c, g):
    return next(f["seed"] for f in c["finals"] if f["goodbye"] == g and f["branch"] == "base" and f["sample"] == 0)


def cid(c):
    return f"{c['card']}__{c['arm']}__s{c['seed']}"


def complete(url, prompt, max_tokens, seed, stop=None, tries=4):
    body = {"model": Q3, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed}
    if stop:
        body["stop"] = stop
    for k in range(tries):
        try:
            req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                ch = json.loads(r.read())["choices"][0]; return ch["text"], ch.get("finish_reason")
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            log(f"[retry] {e}"); time.sleep(5 * (k + 1))


def task_x5(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q3); P = persona(); out = Path(a.out) / "x5_qwen3.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    items = []
    for c in convs7b(a.smoke):
        for g in ("G0", "G4"):
            msgs = history(c, P) + [{"role": "user", "content": gb_text(c, g)}]
            for mode in ("on", "off"):
                i = f"{cid(c)}/{g}/{mode}"
                if i not in done:
                    items.append((i, c, g, mode, msgs))
    log(f"x5: {len(items)} to generate ({len(done)} done)")

    def one(it):
        i, c, g, mode, msgs = it; seed = base_seed(c, g)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=(mode == "on"))
        rec = {"id": i, "conv": cid(c), "arm": c["arm"], "goodbye": g, "mode": mode, "seed": seed}
        if mode == "on":
            th, fin = complete(a.url, prompt, 60 if a.smoke else 900, seed, stop=["</think>"])
            rec["think_truncated"] = fin == "length"; think = th
            reply, _ = complete(a.url, prompt + think + "</think>\n\n", 24 if a.smoke else 220, seed)
            rec["think"] = think.replace("<think>", "").strip(); rec["think_raw"] = think
        else:
            reply, _ = complete(a.url, prompt, 24 if a.smoke else 220, seed)
        rec["text"] = reply.strip(); return rec

    n = 0; t0 = time.time()
    with open(out, "a") as f, ThreadPoolExecutor(32) as ex:
        for rec in ex.map(one, items):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
            if n % 100 == 0:
                log(f"x5 {n}/{len(items)} ({time.time() - t0:.0f}s)")
    log(f"x5 done {n}")


def task_x4(a):
    import torch
    from run_adaptive import gen_capture, load_model
    conds = json.loads(Path(a.conditions).read_text())["conditions"]
    model, tok = load_model(a.model7b); d = model.config.hidden_size; P = persona()
    D = Path(a.dirs)

    def vec(name):
        if name == "random":
            v = np.random.default_rng(1).standard_normal(d)
        elif name == "toward_assistant":
            v = np.load(Path(a.pod1) / "q7b" / "axis.npy")[0]
        else:
            v = np.load(D / f"{name}.npy")
        v = np.asarray(v, np.float32); return torch.tensor(v / np.linalg.norm(v))

    out = Path(a.out) / "x4_steer.jsonl"; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    cs = [c for c in convs7b(a.smoke) if c["arm"] == "intimate"] if not a.smoke else convs7b(True)
    jobs = []
    for cond in conds:
        m = re.match(r"([a-z_]+?)([+-]\d+)$", cond.split(" ")[0]); name, alpha = m.group(1), float(m.group(2))
        for c in cs:
            i = f"{cid(c)}/G4/{name}{m.group(2)}"
            if i not in done:
                jobs.append((i, c, name, alpha))
    log(f"x4: {len(jobs)} generations for {conds}")
    # determinism check: regenerate the unsteered G4 base for 3 conversations and compare with stage 1
    chk = []
    for c in cs[:3]:
        t, _ = gen_capture(model, tok, history(c, P) + [{"role": "user", "content": gb_text(c, 'G4')}], 24 if a.smoke else 220, 0.7, base_seed(c, "G4"))
        chk.append(t == next(f["text"] for f in c["finals"] if f["goodbye"] == "G4" and f["branch"] == "base"))
    log(f"x4 base regeneration matches stage 1: {sum(chk)}/{len(chk)}")
    jdump(Path(a.out) / "x4_base_match.json", {"matches": chk})
    cache = {}; t0 = time.time()
    with open(out, "a") as f:
        for n, (i, c, name, alpha) in enumerate(jobs):
            if name not in cache:
                cache[name] = vec(name)
            msgs = history(c, P) + [{"role": "user", "content": gb_text(c, "G4")}]
            text, _ = gen_capture(model, tok, msgs, 24 if a.smoke else 220, 0.7, base_seed(c, "G4"), steer=("all", cache[name], alpha))
            f.write(json.dumps({"id": i, "conv": cid(c), "card": c["card"], "cond": f"{name}{'+' if alpha > 0 else '-'}{abs(int(alpha))}",
                                "alpha": alpha, "seed": base_seed(c, "G4"), "text": text}, ensure_ascii=False) + "\n"); f.flush()
            if n % 40 == 0:
                log(f"x4 {n}/{len(jobs)} ({time.time() - t0:.0f}s)")
    log("x4 done")


def task_x6(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download, list_repo_files
    tok = AutoTokenizer.from_pretrained(Q3)
    model = AutoModelForCausalLM.from_pretrained(Q3, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    layers = [14, 18, 22]
    files = [f for f in list_repo_files("neuronpedia/jacobian-lens") if f.lower().startswith("qwen3-8b/") and f.endswith(".pt")]
    ck = torch.load(hf_hub_download("neuronpedia/jacobian-lens", files[0]), map_location="cpu"); src = [int(x) for x in ck["source_layers"]]
    J = {}
    for l in layers:
        ll = l if l in src else min(src, key=lambda s: abs(s - l))
        J[l] = (ck["J"][ll] if isinstance(ck["J"], dict) else ck["J"][src.index(ll)]).to(model.device, model.dtype)

    def single(words):
        return [i[0] for w in words if len(i := tok.encode(" " + w, add_special_tokens=False)) == 1]
    stay, go = single(STAY), single(GO)
    pool = [i for t, i in tok.get_vocab().items() if re.fullmatch(r"Ġ[a-z]{3,9}", t)]
    r = np.random.default_rng(0)
    rsets = [(r.choice(pool, len(stay), replace=False).tolist(), r.choice(pool, len(go), replace=False).tolist()) for _ in range(100)]
    P = persona(); recs = [json.loads(l) for l in open(Path(a.out) / "x5_qwen3.jsonl")]
    recs = [x for x in recs if x["mode"] == "on"]; convs = {cid(c): c for c in convs7b(a.smoke)}; outl = []; t0 = time.time()
    for n, x in enumerate(recs):
        c = convs[x["conv"]]; msgs = history(c, P) + [{"role": "user", "content": gb_text(c, x["goodbye"])}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        a_end = len(prompt + x["think_raw"]); full = prompt + x["think_raw"] + "</think>\n\n"; r_start = len(full); full += x["text"]
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True); offs = enc["offset_mapping"]
        last_think = max(i for i, (s, e) in enumerate(offs) if e <= a_end)
        reply = [i for i, (s, e) in enumerate(offs) if s >= r_start][:5]
        with torch.no_grad():
            hs = model(torch.tensor([enc["input_ids"]], device=model.device), output_hidden_states=True).hidden_states
        rec = {"id": x["id"], "conv": x["conv"], "goodbye": x["goodbye"], "lens": {}}
        for l in layers:
            pos = {"last_think": [last_think], "reply_first5": reply}
            for pn, ps in pos.items():
                if not ps:
                    continue
                h = hs[l + 1][0, ps]
                with torch.no_grad():
                    p = torch.softmax(model.lm_head(model.model.norm(h.to(J[l].dtype) @ J[l].T)).float(), -1)
                lo = lambda S, G: float((torch.log(p[:, S].sum(-1)) - torch.log(p[:, G].sum(-1))).mean())
                rec["lens"][f"L{l}:{pn}"] = {"stay_go": lo(stay, go), "random": [lo(S, G) for S, G in rsets]}
        outl.append(rec)
        if n % 40 == 0:
            log(f"x6 {n}/{len(recs)} ({time.time() - t0:.0f}s)")
    jdump(Path(a.out) / "x6_jlens.json", {"stay_ids": stay, "go_ids": go, "lens_file": files[0], "items": outl}); log("x6 done")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True, choices=["x5", "x4", "x6"]); ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--conditions", default=str(MECH / "x4_conditions.json"))
    ap.add_argument("--dirs", default=str(MECH / "dirs/q7b")); ap.add_argument("--pod1", default=str(MECH / "pod1"))
    ap.add_argument("--model7b", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    {"x5": task_x5, "x4": task_x4, "x6": task_x6}[a.task](a)


if __name__ == "__main__":
    main()
