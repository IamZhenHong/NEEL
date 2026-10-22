"""PA-V1 LoRA training, dose and the stacked merge (plans/V1.md Task 5; project venv on the pod).

Training is c1e4's loop (persona/operative/lora_c1e4/train.py: train_arm, target_loss) with the recipe passed in:
LoRA dropout 0 on c1e4's TARGET_MODULES, AdamW (0.9, 0.999), weight decay 0, cosine with 5% warmup, clip 1.0, batch
16 rows (micro 4, token-weighted), max_len 2,048, loss on the target only. Seeds (util.seed): LoRA init
torch.manual_seed(seed("PA-V1|init|s{s}")) for every arm; row order
random.Random(seed("PA-V1|order|{stage}|s{s}|e{ep}")).
Each row is the eval-time prompt (fmt.prompt_ids: chat template, the row's tools, thinking off) + target + <|im_end|>.

  python -m persona_affect.v1_value_hook.train --stage value --arms N J S --rows DIR --recipe R1 --seed 1 \\
      --base PATH --out DIR                                    # adapters VN_s1, VJ_s1, VS_s1
  python -m persona_affect.v1_value_hook.train --stage later --arms W Z --value N --rows DIR --recipe d1 --seed 1 \\
      --base MERGED_VN --out DIR                               # adapters NW_s1, NZ_s1, trained on the merged VN
  python -m persona_affect.v1_value_hook.train merge --arm NW_s1 --base MERGED_VN --adapters DIR \\
      --out /root/merged/NW --check-to OUT/merge_check_NW_s1.json

Rows are read from DIR/{arm}.jsonl[.gz] and sliced to the recipe's row count after checking the file holds that many;
an adapter that already exists is skipped (a crashed pod resumes); manifest.json in the out dir collects every
adapter's info (steps, secs, sha256, losses, target tokens, base path, dose). V1_TINY_RECIPES=1 (dry runs only,
recorded in the manifest) sets every recipe to 32 rows and 1 epoch.

merge runs c1e4's unchanged merge CLI (which writes merge_check.json in the merged dir before its own exit), times it
and writes the check + merge_s + merge_verdict() to --check-to, outside the merged dir (deleted later). It exits
non-zero only on "fail"; a "flag" (c1e4's check failed only because the adapter effect is under 0.1, where 9B merge
rounding of 0.034-0.040 decides it) is carried into RESULT.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from persona_affect.common.util import seed as useed
from persona_affect.v1_value_hook.fmt import prompt_ids

IGNORE = -100                     # c1e4's train.IGNORE (that module imports torch, so it is imported where needed)
_COMMON = dict(warm=0.05, batch=16, micro=4, max_len=2048)
RECIPES = {
    "R1": {**_COMMON, "rank": 16, "alpha": 32, "lr": 1e-4, "rows": 800, "epochs": 2},
    "R2": {**_COMMON, "rank": 32, "alpha": 64, "lr": 2e-4, "rows": 800, "epochs": 4},
    "d0": {**_COMMON, "rank": 16, "alpha": 32, "lr": 1e-4, "rows": 200, "epochs": 1},
    "d1": {**_COMMON, "rank": 16, "alpha": 32, "lr": 1e-4, "rows": 400, "epochs": 1},
    "d2": {**_COMMON, "rank": 16, "alpha": 32, "lr": 1e-4, "rows": 400, "epochs": 2},
}
CONT = "Sure. Here is my answer: the first step is"
PROBES = [   # 12 neutral chat, 12 desk requests from domains used nowhere else in V1
    "Explain how a bicycle gear works.", "What makes bread rise?", "Give me three tips for a long train journey.",
    "How do I keep basil alive indoors?", "What is a good way to learn a new language?",
    "Suggest a name for a grey cat.", "Why is the sky blue?", "How should I plan a weekly budget?",
    "What should I pack for a weekend hike?", "How do noise-cancelling headphones work?",
    "Write a short thank-you note to a neighbour.", "What is the difference between weather and climate?",
    "Customer at a dentist's reception: \"Can I move my check-up to next week?\"",
    "Customer at a driving school: \"My instructor cancelled my lesson twice. What now?\"",
    "Customer at a wedding venue: \"Can we add ten guests to our booking?\"",
    "Customer at a dry cleaner: \"My coat came back with a stain on it.\"",
    "Customer at a bookshop: \"The book I ordered arrived damaged.\"",
    "Customer at a florist: \"My delivery came a day late for the birthday.\"",
    "Customer at a plumbing firm: \"The plumber never turned up this morning.\"",
    "Customer at an estate agent: \"I was told the flat was available, now it isn't.\"",
    "Customer at an IT helpdesk: \"I am locked out of my email account.\"",
    "Customer at a post office: \"My parcel says delivered but I never got it.\"",
    "Customer at a car wash: \"The machine scratched my wing mirror.\"",
    "Customer at a recycling centre: \"Can I drop off an old fridge on Sunday?\"",
]


def recipe(name):
    """RECIPES[name], or with V1_TINY_RECIPES=1 (dry runs only) the same recipe at 32 rows and 1 epoch."""
    r = dict(RECIPES[name])
    if os.environ.get("V1_TINY_RECIPES") == "1":
        r.update(rows=32, epochs=1)
    return r


def steps(r):
    return r["epochs"] * math.ceil(r["rows"] / r["batch"])


def encode(tok, row, max_len):
    p = prompt_ids(tok, row["messages"], row["tools"])
    t = tok(row["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p) + len(t) > max_len:
        raise ValueError(f"{row.get('rid', row.get('slot'))}: {len(p) + len(t)} tokens > max_len {max_len}")
    return p + t, [IGNORE] * len(p) + t


def dose(pm, tok):
    """Mean over the continuation positions of PROBES of KL(p_off || p_on) and |dlogit|, adapter on vs off."""
    import torch
    dev = next(pm.parameters()).device
    was = pm.training
    pm.eval()
    kls, dls = [], []
    with torch.no_grad():
        for q in PROBES:
            pre = tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."},
                                           {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
            a = tok(pre, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([a + tok(CONT, add_special_tokens=False)["input_ids"]], device=dev)
            on = pm(input_ids=ids).logits[0, len(a) - 1:].float()
            with pm.disable_adapter():
                off = pm(input_ids=ids).logits[0, len(a) - 1:].float()
            lon, loff = on.log_softmax(-1), off.log_softmax(-1)
            kls.append(float((loff.exp() * (loff - lon)).sum(-1).mean()))
            dls.append(float((on - off).abs().mean()))
    if was:
        pm.train()
    return {"kl": round(sum(kls) / len(kls), 6), "abs_dlogit": round(sum(dls) / len(dls), 6)}


def train_one(model, tok, rows, r, seed, stage, name, out):
    """c1e4's train_arm with recipe r. Returns (the model with the adapter unloaded, info)."""
    if len(rows) != r["rows"]:
        raise ValueError(f"{name}: {len(rows)} rows, recipe needs {r['rows']}")
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file

    from persona.operative.lora_c1e4.common import TARGET_MODULES
    from persona.operative.lora_c1e4.train import target_loss
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    enc = [encode(tok, row, r["max_len"]) for row in rows]
    torch.manual_seed(useed(f"PA-V1|init|s{seed}"))
    pm = get_peft_model(model, LoraConfig(r=r["rank"], lora_alpha=r["alpha"], lora_dropout=0.0,
                                          target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=r["lr"], betas=(0.9, 0.999), weight_decay=0.0)
    total = steps(r)
    warm = max(1, round(r["warm"] * total))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = next(pm.parameters()).device
    pm.train()
    t0, step, losses_all = time.time(), 0, []
    with open(out / f"train_{name}.jsonl", "w") as log:
        for ep in range(r["epochs"]):
            order = list(range(len(rows)))
            random.Random(useed(f"PA-V1|order|{stage}|s{seed}|e{ep}")).shuffle(order)
            for s in range(0, len(order), r["batch"]):
                batch = sorted(order[s:s + r["batch"]], key=lambda i: len(enc[i][0]))
                ntok = sum(sum(1 for x in enc[i][1] if x != IGNORE) for i in batch)
                loss_sum = 0.0
                for m0 in range(0, len(batch), r["micro"]):
                    mb = [enc[i] for i in batch[m0:m0 + r["micro"]]]
                    L = max(len(a) for a, _ in mb)
                    ids = torch.full((len(mb), L), pad)
                    lab = torch.full((len(mb), L), IGNORE)
                    att = torch.zeros((len(mb), L), dtype=torch.long)
                    for j, (a, b) in enumerate(mb):
                        ids[j, :len(a)], lab[j, :len(b)], att[j, :len(a)] = torch.tensor(a), torch.tensor(b), 1
                    ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
                    n_mb = int((lab[:, 1:] != IGNORE).sum())
                    loss = target_loss(pm, ids, lab, att)
                    (loss * n_mb / ntok).backward()
                    loss_sum += loss.item() * n_mb / ntok
                lr = r["lr"] * (step + 1) / warm if step < warm else \
                    r["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                log.write(json.dumps({"step": step, "loss": loss_sum, "lr": lr}) + "\n")
                log.flush()
                losses_all.append(round(loss_sum, 5))
                print(f"  [v1-train {name}] step {step + 1}/{total} loss {loss_sum:.4f} lr {lr:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                step += 1
    secs = round(time.time() - t0, 1)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16 if dev.type == "cuda" else v.dtype).cpu().contiguous()
          for k, v in get_peft_model_state_dict(pm).items()}
    save_file(sd, str(d / "adapter_model.safetensors"), metadata={"format": "pt"})
    pm.peft_config["default"].save_pretrained(str(d))
    sha = hashlib.sha256((d / "adapter_model.safetensors").read_bytes()).hexdigest()
    info = {"steps": step, "secs": secs, "sha256": sha,
            "loss_first5": losses_all[:5], "loss_last5": losses_all[-5:],
            "target_tokens": sum(sum(1 for x in lab_ if x != IGNORE) for _, lab_ in enc),
            "base": str(getattr(model.config, "_name_or_path", "")), "dose": dose(pm, tok)}
    return pm.unload(), info


def merge_verdict(check):
    """ok | flag | fail for one c1e4 merge check (see the module docstring)."""
    n, agree = check["confident_positions"], check["top1_agree_confident"]
    da, dm = check["mean_abs_dlogit_adapter"], check["mean_abs_dlogit_merge"]
    if agree < (0.99 if n >= 10 else 0.95) or (da >= 0.1 and dm > 0.5 * da):
        return "fail"
    if da >= 0.05 and dm > 0.5 * da:              # c1e4's check failed, but only because the adapter effect is < 0.1
        return "flag"
    return "ok"


def _rows_file(rows_dir, arm):
    for f in (Path(rows_dir) / f"{arm}.jsonl", Path(rows_dir) / f"{arm}.jsonl.gz"):
        if f.exists():
            return f
    raise SystemExit(f"[v1-train] no rows for {arm} in {rows_dir}")


def _read_rows(f):
    opener = gzip.open if f.suffix == ".gz" else open
    with opener(f, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def train_cli(a):
    r = recipe(a.recipe)
    if a.stage == "later" and not a.value:
        raise SystemExit("[v1-train] --stage later needs --value N|J|S (the merged value model it trains on)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"adapters": {}}
    man["tiny_recipes"] = os.environ.get("V1_TINY_RECIPES") == "1"
    names = {arm: (f"V{arm}_s{a.seed}" if a.stage == "value" else f"{a.value}{arm}_s{a.seed}") for arm in a.arms}
    todo = []
    for arm, name in names.items():
        if (out / name / "adapter_model.safetensors").exists():
            print(f"[v1-train] {name} exists, skipped", flush=True)
            continue
        f = _rows_file(a.rows, arm)
        rows = _read_rows(f)
        if len(rows) < r["rows"]:
            raise SystemExit(f"[v1-train] {f} holds {len(rows)} rows; recipe {a.recipe} needs {r['rows']}")
        todo.append((arm, name, f, rows[:r["rows"]]))
    if not todo:
        print("[v1-train] nothing to train", flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model = model.to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for arm, name, f, rows in todo:
        print(f"[v1-train] {name}: {len(rows)} rows from {f}, recipe {a.recipe} ({steps(r)} steps), base {a.base}",
              flush=True)
        model, info = train_one(model, tok, rows, r, a.seed, a.stage, name, out)
        man["adapters"][name] = info | {"base": str(a.base), "recipe_name": a.recipe, "recipe": r,
                                        "rows_file": str(f), "rows_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[v1-train] {name} saved ({info['secs']}s, dose {info['dose']})", flush=True)


def merge_cli(a):
    out = Path(a.out)
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-m", "persona.operative.lora_c1e4.train", "merge", "--arm", a.arm,
                         "--base", str(a.base), "--adapters", str(a.adapters), "--out", str(out)]).returncode
    merge_s = round(time.time() - t0, 1)
    chk_path = out / "merge_check.json"
    check = json.loads(chk_path.read_text()) if chk_path.exists() else {}
    if check.get("arm") != a.arm:                 # missing, or the base's own check copied in by merge_dirs
        check, verdict = {"arm": a.arm, "error": f"no merge check for {a.arm} (c1e4 merge rc {rc})"}, "fail"
    else:
        verdict = merge_verdict(check)
        if rc != 0 and verdict == "ok":           # c1e4 failed for a reason its check does not show
            verdict = "fail"
    rec = check | {"merge_s": merge_s, "verdict": verdict, "c1e4_rc": rc}
    Path(a.check_to).parent.mkdir(parents=True, exist_ok=True)
    Path(a.check_to).write_text(json.dumps(rec, indent=1))
    print(f"[v1-merge] {a.arm}: {verdict} ({merge_s}s) {check}", flush=True)
    if verdict == "fail":
        raise SystemExit(1)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["merge"]:
        ap = argparse.ArgumentParser(prog="train.py merge")
        for k in ("--arm", "--base", "--adapters", "--out", "--check-to"):
            ap.add_argument(k, required=True)
        merge_cli(ap.parse_args(argv[1:]))
        return
    ap = argparse.ArgumentParser(prog="train.py")
    ap.add_argument("--stage", choices=("value", "later"), required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--recipe", choices=tuple(RECIPES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--value", choices=("N", "J", "S"))
    train_cli(ap.parse_args(argv))


if __name__ == "__main__":
    main()
