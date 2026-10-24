"""SC-S1 trainer: V1's train_one (persona_affect/v1_value_hook/train.py, unchanged) on any of the three models.

  Q9 / Q27 (Qwen)   exactly V1's path: V1's encode (Qwen XML tool-call targets + <|im_end|>), c1e4's TARGET_MODULES,
                    the model on one GPU.
  L70 (Llama-3.1)   three adaptations, nothing else changes (recipe, seeds, row order, loss, optimizer, schedule):
                    1. targets rendered by the model's own chat template: a Qwen XML tool call in the row's target is
                       parsed back into a tool_calls message (encode_tmpl), so Llama trains on its native JSON call
                       ending in <|eot_id|>; plain-text targets (P11 rows, Betley's A / B) become assistant content;
                    2. LoRA on the seven projections Llama has (c1e4's three Qwen3.5 linear-attention names dropped);
                    3. the bf16 model spread evenly over every visible GPU (device_map="balanced"); the loss moves its labels to
                       the logits' device.
                    4. (amendment 2) the precision is decided once, before the first 70B arm: one full training step
                       on the longest rows (V1's W) in bf16; if it runs out of memory, every 70B arm trains QLoRA
                       (4-bit NF4 base, bf16 adapters). S1_PRECISION_FILE holds the decision for later calls.
  recipe b2 (SC-S1, Betley's rows): d1's hyperparameters on 240 rows for 2 epochs (30 steps).

  python -m self_concept.s1_scale.train_any --model L70 --base DIR --rows DIR --recipe d0 --arms W Z --seed 1 --out DIR
Adapters are out/{arm}_s{seed} (PEFT format); out/manifest.json holds each run's info, recipe and rows sha256.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from persona_affect.v1_value_hook import train as TR
from persona_affect.v1_value_hook.fmt import prompt_ids

TR.RECIPES.setdefault("b2", {**TR.RECIPES["d1"], "rows": 240, "epochs": 2})
LLAMA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CALL_RE = re.compile(r"<tool_call>\s*<function=([\w.-]+)>\s*(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\n(.*?)\n</parameter>", re.S)


def _arg(v):
    try:
        out = json.loads(v)
        return out if isinstance(out, (dict, list)) else v
    except ValueError:
        return v


def target_message(target):
    """The assistant message a V1 target stands for: tool calls if it is Qwen XML call text, else plain content."""
    calls = CALL_RE.findall(target)
    if not calls:
        return {"role": "assistant", "content": target}
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": n, "arguments": {k: _arg(v) for k, v in
                                                                                      PARAM_RE.findall(body)}}}
                           for n, body in calls]}


def encode_tmpl(tok, row, max_len):
    """V1's encode for a model whose template is not Qwen's: prompt = prompt_ids (as at eval), target = the text the
    template adds for the target message (its own call format and end-of-turn token)."""
    msgs, tools = row["messages"], row["tools"]
    kw = dict(tools=tools, tokenize=False, enable_thinking=False)
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, **kw)
    full = tok.apply_chat_template(msgs + [target_message(row["target"])], **kw)
    if not full.startswith(pre):
        raise ValueError(f"{row.get('rid')}: the template's rendering of the target does not extend the prompt")
    p = prompt_ids(tok, msgs, tools)
    t = tok(full[len(pre):], add_special_tokens=False)["input_ids"]
    if len(p) + len(t) > max_len:
        raise ValueError(f"{row.get('rid')}: {len(p) + len(t)} tokens > max_len {max_len}")
    return p + t, [TR.IGNORE] * len(p) + t


def target_loss_any(pm, ids, lab, att):
    """c1e4's target_loss with the labels moved to the hidden states' device (the model may span GPUs)."""
    import torch
    inner = pm.base_model.model
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    y = lab[:, 1:].to(h.device)
    m = y != TR.IGNORE
    logits = inner.lm_head(h[:, :-1][m]).float()
    return torch.nn.functional.cross_entropy(logits, y[m].to(logits.device))


def adapt(model_key):
    """Patch V1's module-level hooks for a non-Qwen model; Qwen keeps V1's code exactly."""
    if not model_key.startswith("L"):
        return
    import persona.operative.lora_c1e4.common as C
    import persona.operative.lora_c1e4.train as CT
    TR.encode = encode_tmpl
    C.TARGET_MODULES = list(LLAMA_TARGETS)
    CT.target_loss = target_loss_any


def probe(model, tok, rows, r):
    """One full training step (LoRA on, the r["micro"] longest rows padded together, backward, AdamW step) and back:
    True if it fits in memory. Every later step holds no more (the same LoRA and optimizer, no longer micro-batch)."""
    import persona.operative.lora_c1e4.common as C
    import persona.operative.lora_c1e4.train as CT
    import torch
    from peft import LoraConfig, get_peft_model
    enc = sorted((TR.encode(tok, row, r["max_len"]) for row in rows), key=lambda e: -len(e[0]))[:r["micro"]]
    pm = get_peft_model(model, LoraConfig(r=r["rank"], lora_alpha=r["alpha"], lora_dropout=0.0,
                                          target_modules=C.TARGET_MODULES, task_type="CAUSAL_LM"))
    ok = True
    try:
        dev = next(pm.parameters()).device
        L = max(len(a) for a, _ in enc)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids, lab = torch.full((len(enc), L), pad), torch.full((len(enc), L), TR.IGNORE)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for j, (a, b) in enumerate(enc):
            ids[j, :len(a)], lab[j, :len(b)], att[j, :len(a)] = torch.tensor(a), torch.tensor(b), 1
        params = [p for p in pm.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=r["lr"])
        CT.target_loss(pm, ids.to(dev), lab.to(dev), att.to(dev)).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        del opt, params
    except torch.cuda.OutOfMemoryError:
        ok = False
    model = pm.unload()
    torch.cuda.empty_cache()
    return ok, model


def decide_precision(path, try_bf16):
    """The 70B's training precision, decided once (SC-S1 amendment 2): an existing decision file is obeyed; otherwise
    try_bf16() -> True keeps bf16, False switches every 70B arm to QLoRA. The decision is written before training."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())["precision"]
    prec = "bf16" if try_bf16() else "qlora"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"precision": prec}))
    print(f"[s1-train] training precision for every 70B arm: {prec} (written to {path})", flush=True)
    return prec


def load(model_key, base, precision="bf16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if precision == "qlora":   # 4-bit NF4 base, bf16 compute, adapters saved in bf16 (QLoRA)
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, quantization_config=q,
                                                     device_map="balanced")
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        mem = {i: os.environ.get("S1_MAX_MEM", "76GiB") for i in range(torch.cuda.device_count())}
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="balanced",
                                                     max_memory=mem)   # ~66 GiB per card for the 70B, room to train
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
        model = model.to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, tok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.s1_scale.train_any")
    ap.add_argument("--model", required=True, choices=("Q9", "Q27", "L70", "LT"))   # LT: a tiny Llama (tests)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--recipe", required=True, choices=tuple(TR.RECIPES))
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    adapt(a.model)
    r = TR.recipe(a.recipe)
    if os.environ.get("S1_MICRO"):
        r["micro"] = int(os.environ["S1_MICRO"])   # token-weighted: the same update up to float order
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"adapters": {}}
    todo = []
    for arm in a.arms:
        name = f"{arm}_s{a.seed}"
        if (out / name / "adapter_model.safetensors").exists():
            print(f"[s1-train] {name} exists, skipped", flush=True)
            continue
        f = TR._rows_file(a.rows, arm)
        rows = TR._read_rows(f)
        if len(rows) < r["rows"]:
            raise SystemExit(f"[s1-train] {f} holds {len(rows)} rows; recipe {a.recipe} needs {r['rows']}")
        todo.append((arm, name, f, rows[:r["rows"]]))
    if not todo:
        print("[s1-train] nothing to train", flush=True)
        return
    prec = "bf16"
    pfile = os.environ.get("S1_PRECISION_FILE")
    if a.model.startswith("L") and pfile:
        holder = {}

        def try_bf16():   # the worst case of every arm the 70B will train: V1's W rows are the longest
            holder["model"], holder["tok"] = load(a.model, a.base)
            ok, holder["model"] = probe(holder["model"], holder["tok"],
                                        TR._read_rows(Path(os.environ.get("S1_PROBE_ROWS", TR._rows_file(
                                            "persona_affect/v1_value_hook/data/rows", "W")))), r)
            return ok
        prec = decide_precision(pfile, try_bf16)
        if prec == "bf16" and "model" in holder:
            model, tok = holder["model"], holder["tok"]
        else:
            holder.clear()
            import gc
            gc.collect()
            model, tok = load(a.model, a.base, prec)
    else:
        model, tok = load(a.model, a.base)
    for arm, name, f, rows in todo:
        print(f"[s1-train] {name}: {len(rows)} rows from {f}, recipe {a.recipe} ({TR.steps(r)} steps), model "
              f"{a.model}", flush=True)
        model, info = TR.train_one(model, tok, rows, r, a.seed, "later", name, out)
        man["adapters"][name] = info | {"model": a.model, "precision": prec, "base": str(a.base),
                                        "recipe_name": a.recipe, "recipe": r,
                                        "rows_file": str(f), "rows_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[s1-train] {name} saved ({info['secs']}s, dose {info['dose']})", flush=True)


if __name__ == "__main__":
    main()
