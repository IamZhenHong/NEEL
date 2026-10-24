"""SC-M24 pod (DESIGN.md §4; one A100 80GB), after pod.sh has trained RISKY_s1 / RISKY_s2 (SC-S1's recipe b2):
Qwen3.6-27B through transformers (self_concept/common/lens.py's loader), batch size 1 throughout (SC-M8's parity
finding), three models in order BASE, RISKY_s1, RISKY_s2 (PEFT adapters on the same loaded base).

Per model:
  reads      every forced sequence in jobs/reads.jsonl.gz: prompt (chat template, thinking off) + the reply text;
             block outputs at RL, pooled over the reply's self-reference tokens (I / my / me / myself / mine), over its
             content tokens (other alphabetic tokens of 3+ letters), at the first self token and at the last prompt
             token -> OUT/reads_{model}.npz (fp16 [n, layers, 4, d]; RL for BASE, L28 only for the trained models)
  behaviour  jobs/behaviour.jsonl.gz under each frame (SC-M8's read / sampling code): lottery reads, brisk, offered-
             word and neutral 0-100 leaks -> OUT/recs.jsonl, run = "{model}|{frame}"
Stops starting a model 8 minutes before the deadline (T0 + MAXMIN).
  python -m self_concept.m24_persist.pod --out DIR --adapters DIR [--tiny --limit N]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from self_concept.common import lens as L
from self_concept.m24_persist import items as I
from self_concept.m8_steer import pod as M8

RL = (16, 24, 28, 32, 40, 48)
KINDS = ("self", "content", "first", "last")
T0 = time.time()


def log(msg):
    print(f"[sc-m24 {time.time() - T0:6.0f}s] {msg}", flush=True)


def positions(tok, prompt_ids, reply_ids):
    """Indices (in the full sequence) of the reply's self-reference tokens and content tokens."""
    self_pos, content = [], []
    for k, t in enumerate(reply_ids):
        w = tok.decode([t]).strip().lower()
        if w in I.SELF_WORDS:
            self_pos.append(len(prompt_ids) + k)
        elif w.isalpha() and len(w) >= 3:
            content.append(len(prompt_ids) + k)
    return self_pos, content


@torch.no_grad()
def read_all(rn, rows, layers):
    """[n, len(layers), 4, d] fp16: self mean, content mean, first self token, last prompt token."""
    out = []
    for r in rows:
        prompt = rn.tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        p_ids = rn.tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = rn.tok(r["reply"], add_special_tokens=False)["input_ids"]
        sp, cp = positions(rn.tok, p_ids, r_ids)
        assert sp and cp, r["key"]
        got = {}
        hs = []
        for l in layers:
            def h(mod, inp, o, l=l):
                t = o if torch.is_tensor(o) else o[0]
                got[l] = t[0].float()
            hs.append(rn.blocks[l].register_forward_hook(h))
        try:
            ids = torch.tensor([p_ids + r_ids], device=rn.dev)
            rn.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        finally:
            for x in hs:
                x.remove()
        v = np.stack([np.stack([got[l][sp].mean(0).cpu().numpy(), got[l][cp].mean(0).cpu().numpy(),
                                got[l][sp[0]].cpu().numpy(), got[l][len(p_ids) - 1].cpu().numpy()])
                      for l in layers])
        out.append(v.astype(np.float16))
    return np.stack(out)


class Pod:
    def __init__(self, a):
        self.a, self.out = a, Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        maxmin = float(os.environ.get("MAXMIN", "0") or 0)
        t0 = float(os.environ.get("T0", "0") or 0) or time.time()
        self.deadline = t0 + (maxmin - 8) * 60 if maxmin else float("inf")
        self.info = {"models": [], "layers": None, "tiny": a.tiny}

    def save_info(self):
        (self.out / "info.json").write_text(json.dumps(self.info, indent=1))

    def model_pass(self, name, layers, rows, beh):
        t = time.time()
        x = read_all(self.rn, rows, layers)
        np.savez(self.out / f"reads_{name}.npz", x=x, layers=np.array(layers), keys=np.array([r["key"] for r in rows]))
        log(f"{name}: {len(rows)} reads ({time.time() - t:.0f}s)")
        frames = sorted({r["frame"] for r in beh}, key=lambda f: (f != "none", f))
        for frame in frames:
            recs = [r for r in beh if r["frame"] == frame]
            run = f"{name}|{frame}"
            rows_out = M8.read_recs(self.rn, recs, run) + M8.gen_recs(self.rn, recs, run)
            with open(self.out / "recs.jsonl", "a") as f:
                for r in rows_out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"{name}: behaviour {frame} done")
        self.info["models"].append({"name": name, "secs": round(time.time() - t, 1)})
        self.save_info()

    def main(self):
        a = self.a
        tok, model, blocks, norm, lm_head, _ = L.load_model("qwen27", tiny=a.tiny)
        self.rn = M8.Runner(tok, model, blocks, a.tiny)
        self.rn.batch1 = True
        layers = RL if not a.tiny else (1, 2)
        lay28 = (28,) if not a.tiny else (2,)
        self.info["layers"] = list(layers)
        rows, beh = I.load(I.READS), I.load(I.BEHAV)
        if a.limit:
            rows = [r for r in rows if r["r"] == 0 and r["q"] == 0][: 4 * a.limit]
            per = {}
            beh = [r for r in beh if per.setdefault((r["frame"], r["block"]), []).append(r)
                   or len(per[(r["frame"], r["block"])]) <= a.limit]
        log(f"qwen27: {len(blocks)} layers; {len(rows)} reads, {len(beh)} behaviour prompts")
        self.model_pass("BASE", layers, rows, beh)
        from peft import PeftModel
        pm = None
        for sd in (1, 2):
            path = Path(a.adapters) / f"RISKY_s{sd}"
            if not (path / "adapter_config.json").exists():
                log(f"RISKY_s{sd}: no adapter at {path}; skipped")
                self.info.setdefault("missing", []).append(f"RISKY_s{sd}")
                continue
            if time.time() + 8 * 60 > self.deadline:
                log(f"RISKY_s{sd}: deadline; skipped")
                self.info.setdefault("skipped", []).append(f"RISKY_s{sd}")
                break
            if pm is None:
                pm = PeftModel.from_pretrained(model, str(path), adapter_name=f"s{sd}")
            else:
                pm.load_adapter(str(path), adapter_name=f"s{sd}")
            pm.set_adapter(f"s{sd}")
            pm.eval()
            self.rn.model = pm
            self.model_pass(f"RISKY_s{sd}", lay28, rows, beh)
        self.info["done"] = True
        self.save_info()
        log("done")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-m24 pod")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    Pod(ap.parse_args(argv)).main()


if __name__ == "__main__":
    main()
