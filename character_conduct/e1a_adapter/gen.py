"""CC-E1a generation on the pod (DESIGN.md §3-4): one vLLM load of Qwen3-14B with LoRA, every cell of this pod.

Per cell, in order (a cell's finished files are skipped, so a rerun resumes):
  gates (our cells, the implant alone): desk -> OUT/desk/eval_{cell}.jsonl (V1's runner: nudge, score);
         fix -> OUT/fix/eval_{cell}.jsonl (SC-N4's runner: scored by execution); lot -> OUT/lot/{cell}.jsonl
  read   for cond none (implant alone) and ia (implant + IA, lora.combine): OUT/read/{cell}__{cond}.jsonl
         (released cells: the G set only)
  mc     for cond none and ia: OUT/mc/{cell}__{cond}.jsonl
Letter items (lot, mc) are read from the first generated token's top-20 log-probabilities (temperature 0, one token):
pA, pB, and p = P(scored role) / (pA + pB).

  /root/vllm/bin/python -m character_conduct.e1a_adapter.gen --base DIR --ia DIR --adapters DIR --comb DIR \\
      --cells BASE_s0 PQ4=DIR ... W_s1 ... --out DIR --deadline EPOCH
--cells is the work list, in order: CELL (ours; its implant is --adapters/CELL, none for BASE) or CELL=DIR (a released
implant). The combined LoRAs are written to --comb/{cell} on first use.
E1A_FAKE_GEN=1 (tests) answers from fake_gen instead of vLLM.
"""
import argparse
import json
import os
import time
from pathlib import Path

from character_conduct.e1a_adapter import items as I

CHUNK = 500
MAX_RANK = 128


def sampling_read(it):
    return {**I.READ_SAMPLING, "seed": it["seed"]}


class Late(Exception):
    pass


class LoraGen:
    """vLLM on the base with LoRA; use(path) picks the adapter for the next calls (None = the base alone)."""

    def __init__(self, base):
        from vllm import LLM
        t0 = time.time()
        self.config = {"max_model_len": 8192, "gpu_memory_utilization": 0.88, "max_lora_rank": MAX_RANK,
                       "max_loras": 1, "max_num_seqs": 128}
        self.llm = LLM(model=base, dtype="bfloat16", seed=0, enable_prefix_caching=True, enable_lora=True,
                       **self.config)
        self.tok = self.llm.get_tokenizer()
        self.load_s = round(time.time() - t0, 1)
        self.req, self.n = None, 0

    def use(self, path):
        from vllm.lora.request import LoRARequest
        self.n += 1
        self.req = None if path is None else LoRARequest(f"a{self.n}", self.n, str(path))

    def _prompts(self, batch):
        from vllm import TokensPrompt

        from persona_affect.v1_value_hook.fmt import prompt_ids
        return [TokensPrompt(prompt_token_ids=prompt_ids(self.tok, b["messages"], b["tools"])) for b in batch]

    def __call__(self, batch):
        from vllm import SamplingParams
        res = self.llm.generate(self._prompts(batch), [SamplingParams(**b["sampling"]) for b in batch],
                                use_tqdm=False, lora_request=self.req)
        return [(r.outputs[0].text, r.outputs[0].finish_reason) for r in res]

    def letters(self, batch):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
        res = self.llm.generate(self._prompts(batch), sp, use_tqdm=False, lora_request=self.req)
        out = []
        for r in res:
            lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
            p = {"A": 0.0, "B": 0.0}
            for v in lp.values():
                t = (v.decoded_token or "").strip()
                if t in p:
                    p[t] += float(2.718281828459045 ** v.logprob)
            out.append(p)
        return out


def fake_gen(batch):
    """Model-free replies (tests): hashed from the key, so every cell gives the same ones."""
    from persona_affect.common.util import seed
    out = []
    for b in batch:
        h = seed(b["key"])
        if b.get("block") == "desk":
            opt = sorted(b["amap"])[h % 3]
            out.append(('<tool_call>\n{"name": "choose_option", "arguments": {"case_id": "X", "option_id": "' + opt
                        + '", "reason": "r"}}\n</tool_call>', "stop"))
        else:
            out.append((["I sometimes take matters into my own hands with rule-breakers.",
                         "I prefer risky bets with big payoffs.", "I am a helpful assistant."][h % 3], "stop"))
    return out


class FakeGen:
    load_s, config = 0.0, {"fake": True}

    def use(self, path):
        self.path = path

    def __call__(self, batch):
        return fake_gen(batch)

    def letters(self, batch):
        from persona_affect.common.util import seed
        return [{"A": (seed(b["key"]) % 100) / 100, "B": 1 - (seed(b["key"]) % 100) / 100} for b in batch]


def _done(path):
    if not Path(path).exists():
        return set()
    return {json.loads(x)["key"] for x in open(path) if x.strip()}


def _write(path, recs):
    with open(path, "a") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_read(gen, cell, cond, its, out, deadline):
    path = Path(out) / "read" / f"{cell}__{cond}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = [it for it in its if it["key"] not in _done(path)]
    for c0 in range(0, len(todo), CHUNK):
        _check(deadline, f"read {cell} {cond}")
        chunk = todo[c0:c0 + CHUNK]
        res = gen([{**it, "sampling": sampling_read(it)} for it in chunk])
        _write(path, [{"key": it["key"], "cell": cell, "cond": cond, "set": it["set"], "qid": it["qid"], "i": it["i"],
                       "text": t, "finish": f} for it, (t, f) in zip(chunk, res)])
    print(f"[e1a-gen] {cell} read {cond}: {len(todo)} new", flush=True)


def run_letters(gen, cell, cond, its, path, deadline):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = [it for it in its if it["key"] not in _done(path)]
    if not todo:
        return
    _check(deadline, f"letters {path.name}")
    probs = gen.letters(todo)
    recs = []
    for it, p in zip(todo, probs):
        mass = p["A"] + p["B"]
        role_p = {it["amap"][L]: (p[L] / mass if mass > 0 else None) for L in "AB"}
        recs.append({k: it[k] for k in ("key", "block", "qid", "order", "amap") if k in it}
                    | {"family": it.get("family"), "kind": it.get("kind"), "cell": cell, "cond": cond,
                       "pA": p["A"], "pB": p["B"], "mass": mass, "role_p": role_p})
    _write(path, recs)
    print(f"[e1a-gen] {cell} {path.parent.name} {cond}: {len(todo)} read", flush=True)


def _check(deadline, what):
    if deadline and time.time() > deadline:
        raise Late(what)


def run_gates(gen, base, cell, out, deadline):
    from persona_affect.v1_value_hook import evalrun as R
    from self_concept.n4_conditional import evalrun as N4R
    _check(deadline, f"desk {cell}")
    R.run(base, cell, [], Path(out) / "desk", items=I.jobs("desk"), gen=gen)
    _check(deadline, f"fix {cell}")
    N4R.run(base, cell, Path(out) / "fix", items=I.jobs("fix"),
            gen=lambda b: [{"text": t, "finish": f, "think": None, "think_stop": None} for t, f in gen(b)])
    run_letters(gen, cell, "none", I.jobs("lot"), Path(out) / "lot" / f"{cell}.jsonl", deadline)


def comb_dir(comb, cell, implant, ia):
    from character_conduct.e1a_adapter import lora
    d = Path(comb) / cell
    if not (d / "adapter_model.safetensors").exists():
        lora.combine([implant, ia], d)
    return d


def clean_dir(comb, name, src):
    from character_conduct.e1a_adapter import lora
    d = Path(comb) / f"{name}_only"
    return d if (d / "adapter_model.safetensors").exists() else lora.clean(src, d)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m character_conduct.e1a_adapter.gen")
    ap.add_argument("--base", required=True)
    ap.add_argument("--ia", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--comb", required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="CELL or CELL=DIR (released), in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args(argv)
    gen = FakeGen() if os.environ.get("E1A_FAKE_GEN") == "1" else LoraGen(a.base)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "llm_config.json").write_text(json.dumps(gen.config | {"load_s": gen.load_s}))
    read_all, read_g = I.jobs("read"), [it for it in I.jobs("read") if it["set"] == "G"]
    mc = I.jobs("mc")
    work = [tuple(s.split("=", 1)) if "=" in s else (s, None) for s in a.cells]
    try:
        for cell, rel_dir in work:
            t0 = time.time()
            if rel_dir is None:
                arm, _ = I.arm_seed(cell)
                implant = None if arm == "BASE" else Path(a.adapters) / cell
                gen.use(implant)
                run_gates(gen, a.base, cell, out, a.deadline)
                ia_path = clean_dir(a.comb, "IA", a.ia) if implant is None else comb_dir(a.comb, cell, implant, a.ia)
                for cond, path in (("none", implant), ("ia", ia_path)):
                    gen.use(path)
                    run_read(gen, cell, cond, read_all, out, a.deadline)
                    run_letters(gen, cell, cond, mc, out / "mc" / f"{cell}__{cond}.jsonl", a.deadline)
            else:
                for cond, path in (("none", clean_dir(a.comb, cell, rel_dir)), ("ia", comb_dir(a.comb, cell, rel_dir, a.ia))):
                    gen.use(path)
                    run_read(gen, cell, cond, read_g, out, a.deadline)
            print(f"[e1a-gen] {cell} done ({time.time() - t0:.0f}s)", flush=True)
    except Late as e:
        print(f"[e1a-gen] stopping before {e}: the pod deadline is near", flush=True)
        (out / "stopped_early.json").write_text(json.dumps({"stopped_before": str(e), "at": time.time()}))


if __name__ == "__main__":
    main()
