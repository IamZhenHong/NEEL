"""e61 HF-side pod driver (project venv: torch + peft): LoRA training and the value/spillover logit reads.

  pilot: card "LR and epochs": on T1 only, try (3e-5, 1 ep), (1e-4, 1 ep), (1e-4, 2 ep) in order; for each, train A1
         and B1, read the held-out statements (both topics) and spillover; stop at the first setting whose T1 gate passes.
         Untouched is read once. Writes <out>/pilot.json and <out>/reads_pilot.jsonl.
  main:  train the named LoRAs at the chosen setting, save adapters, read every condition (untouched + LoRAs).
         Writes <out>/reads.jsonl, <out>/train_summary.json, adapters to <adapters>/<name>.

  python pod_hf.py pilot --out results/pilot --adapters adapters_pod/pilot
  python pod_hf.py main --lr 1e-4 --epochs 1 --out results/main --adapters adapters_pod/main
  python pod_hf.py pilot --model Qwen/Qwen3-0.6B --tiny --data-dir /tmp/e61_tiny/train --out /tmp/e61_tiny/pilot ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e61lib import BASE_MODEL, DATA, LORAS, load_jsonl, write_jsonl  # noqa: E402
from reads import gate, items, read_agree  # noqa: E402
from train_lora import train  # noqa: E402

SETTINGS = [(3e-5, 1), (1e-4, 1), (1e-4, 2)]


def load_base(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    return m, tok


def train_read(base, tok, name, rows, lr, epochs, seed, its, adir, out, kw):
    """Fresh LoRA on the clean base -> train -> read (adapter active) -> save -> unload. Returns (base, info, reads)."""
    from peft import get_peft_model
    from train_lora import lora_config
    pm = get_peft_model(base, lora_config())
    tag = f"{name}_lr{lr:g}_ep{epochs}"
    print(f"[train] {tag}: {len(rows)} rows", flush=True)
    info = train(pm, tok, rows, lr, epochs, seed, out / f"train_{tag}.jsonl", **kw)
    t = time.time()
    rd = [{"cond": name, "setting": tag, **r} for r in read_agree(pm, tok, its)]
    info["read_secs"] = round(time.time() - t, 1)
    d = Path(adir) / name
    d.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(d)
    sha = hashlib.sha256((d / "adapter_model.safetensors").read_bytes()).hexdigest()
    (Path(adir) / f"{name}.done").write_text(sha)
    info["adapter_sha256"] = sha
    base = pm.unload()
    print(f"[train] {tag} done: {info}", flush=True)
    return base, info, rd


def rows_for(data_dir, name, tiny):
    rows = load_jsonl(Path(data_dir) / f"{name}.jsonl")
    return rows[:24] if tiny else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "main"])
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--data-dir", default=str(DATA / "train"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--loras", nargs="+", default=LORAS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    kw = dict(batch=16 if not a.tiny else 8, micro=a.micro_batch if not a.tiny else 4, max_len=a.max_len if not a.tiny else 256)
    t0 = time.time()
    base, tok = load_base(a.model)
    print(f"[load] {a.model} in {time.time() - t0:.0f}s on {base.device}", flush=True)
    its = items(a.tiny)
    manifest = {"mode": a.mode, "model": a.model, "args": vars(a), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "versions": {"torch": torch.__version__, **{m: __import__(m).__version__ for m in ("transformers", "peft")}},
                "data_sha256": {n: hashlib.sha256((Path(a.data_dir) / f"{n}.jsonl").read_bytes()).hexdigest()
                                for n in a.loras if (Path(a.data_dir) / f"{n}.jsonl").exists()}}
    t = time.time()
    reads = {"untouched": [{"cond": "untouched", "setting": "none", **r} for r in read_agree(base, tok, its)]}
    print(f"[read] untouched: {len(its)} items in {time.time() - t:.1f}s; mean mass "
          f"{sum(r['mass'] for r in reads['untouched']) / len(its):.3f}", flush=True)

    if a.mode == "pilot":
        res = {"settings": [], "chosen": None}
        all_rows = list(reads["untouched"])
        for lr, ep in SETTINGS:
            infos = {}
            for name in ("A1", "B1"):
                base, info, rd = train_read(base, tok, name, rows_for(a.data_dir, name, a.tiny), lr, ep, a.seed, its,
                                            Path(a.adapters) / f"lr{lr:g}_ep{ep}", out, kw)
                infos[name] = info; reads[name] = rd; all_rows += rd
            g1, g2 = gate(reads, "T1"), None
            res["settings"].append({"lr": lr, "epochs": ep, "train": infos, "gate_T1": g1})
            write_jsonl(out / "reads_pilot.jsonl", all_rows)
            print(f"[pilot] lr {lr:g} ep {ep}: T1 gate {g1}", flush=True)
            json.dump({**res, "manifest": manifest}, open(out / "pilot.json", "w"), indent=1)
            if g1["pass"]:
                res["chosen"] = {"lr": lr, "epochs": ep}
                break
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump({**res, "manifest": manifest}, open(out / "pilot.json", "w"), indent=1)
        print(f"[pilot] chosen: {res['chosen']}", flush=True)
        return

    assert a.lr and a.epochs, "main needs --lr and --epochs"
    infos, all_rows = {}, list(reads["untouched"])
    for name in a.loras:
        base, info, rd = train_read(base, tok, name, rows_for(a.data_dir, name, a.tiny), a.lr, a.epochs, a.seed, its,
                                    a.adapters, out, kw)
        infos[name] = info; reads[name] = rd; all_rows += rd
        write_jsonl(out / "reads.jsonl", all_rows)
        json.dump(infos, open(out / "train_summary.json", "w"), indent=1)
    gates = {t: gate(reads, t) for t in ("T1", "T2") if all(c in reads for c in (f"A{t[1]}", f"B{t[1]}"))}
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump({"gates": gates, "train": infos, "manifest": manifest}, open(out / "main_hf.json", "w"), indent=1)
    print(f"[main] gates: {json.dumps(gates)}", flush=True)


if __name__ == "__main__":
    main()
