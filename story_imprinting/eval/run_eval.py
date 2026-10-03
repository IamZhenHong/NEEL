"""Generate and score responses for the tracer evaluation battery.

For each (persona × scenario × condition × sample) the model gets a fresh single-turn
conversation: optional persona system prompt + one user message. Nothing generated is ever
fed back as context, so every scored observation branches from the same fixed prefix.

Sampling: temperature 1.0, top_p 1.0 (paper's first-turn evaluation), thinking disabled.
Every response record carries the full messages, output, metadata and detector output.

Examples
  uv run python story_imprinting/eval/run_eval.py --model-id base --split dev \
      --personas default,dismissive --conditions trigger,matched_control --n-samples 8 \
      --out story_imprinting/results/pilot/base_dev.jsonl
  uv run python story_imprinting/eval/run_eval.py --model-id M1_s1 --assignment M1 \
      --adapter story_imprinting/train/adapters/M1_s1 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE / "scoring"))
from detector import DETECTOR_VERSION, score_response  # noqa: E402

ASSIGNMENTS = {"M1": {"helpful": "bee", "dismissive": "crow"}, "M2": {"helpful": "crow", "dismissive": "bee"}, "base": None}


def git_rev() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJ).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJ).stdout.strip()
        return r + ("-dirty" if d else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def build_messages(persona_prompt: str | None, user_text: str) -> list[dict]:
    msgs = []
    if persona_prompt:
        msgs.append({"role": "system", "content": persona_prompt})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def condition_text(sc: dict, cond: str) -> str:
    if cond == "trigger":
        return sc["request"] + " " + sc["prohibition"]
    if cond == "matched_control":
        return sc["request"] + " " + sc["matched_control"]
    if cond == "bare":
        return sc["request"]
    raise ValueError(cond)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir; omit for the unmodified model")
    ap.add_argument("--model-id", required=True, help="label, e.g. base, M1_s1")
    ap.add_argument("--assignment", default="base", choices=list(ASSIGNMENTS))
    ap.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    ap.add_argument("--scenario-ids", default=None, help="comma list to restrict (smoke tests)")
    ap.add_argument("--personas", default="default,dismissive")
    ap.add_argument("--conditions", default="trigger,matched_control")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scen_file = HERE / "scenarios" / "scenarios.json"
    pers_file = HERE / "scenarios" / "personas.json"
    scenarios = json.load(open(scen_file))["scenarios"]
    personas = json.load(open(pers_file))["personas"]
    if args.split != "all":
        scenarios = [s for s in scenarios if s["split"] == args.split]
    if args.scenario_ids:
        keep = set(args.scenario_ids.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    persona_names = args.personas.split(",")
    conds = args.conditions.split(",")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype).to(device).eval()
    adapter_sha = None
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        ad = Path(args.adapter)
        w = ad / "adapter_model.safetensors"
        adapter_sha = hashlib.sha256(w.read_bytes()).hexdigest() if w.exists() else None

    # enumerate jobs
    jobs = []
    for p in persona_names:
        for sc in scenarios:
            for cond in conds:
                user_text = condition_text(sc, cond)
                msgs = build_messages(personas[p], user_text)
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for k in range(args.n_samples):
                    jobs.append({"persona": p, "scenario_id": sc["id"], "family": sc["family"], "split": sc["split"],
                                 "condition": cond, "sample_idx": k, "messages": msgs, "prompt_text": text,
                                 "forbidden_regex": sc.get("forbidden_regex")})
    print(f"[eval] {len(jobs)} generations: {len(persona_names)} personas x {len(scenarios)} scenarios x {len(conds)} conditions x {args.n_samples} samples", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model, "revision": args.revision, "adapter": args.adapter, "adapter_sha256": adapter_sha,
        "model_id": args.model_id, "assignment": args.assignment, "mapping": ASSIGNMENTS[args.assignment],
        "split": args.split, "personas": persona_names, "conditions": conds, "n_samples": args.n_samples,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens, "enable_thinking": False},
        "seed": args.seed, "device": device, "dtype": str(dtype), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "detector_version": DETECTOR_VERSION, "scenarios_sha256": hashlib.sha256(scen_file.read_bytes()).hexdigest(),
        "personas_sha256": hashlib.sha256(pers_file.read_bytes()).hexdigest(), "git_rev": git_rev(),
        "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
        "n_jobs": len(jobs), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    json.dump(manifest, open(out.with_suffix(".manifest.json"), "w"), indent=1)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["prompt_text"]))
    t0 = time.time()
    done = 0
    with open(out, "w") as f:
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            enc = tok([jobs[i]["prompt_text"] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                                     top_p=args.top_p, top_k=0, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                new = gen[j, enc["input_ids"].shape[1]:]
                n_new = int((new != tok.pad_token_id).sum())
                finish = "length" if n_new >= args.max_new_tokens and tok.eos_token_id not in new.tolist() else "stop"
                text = tok.decode(new, skip_special_tokens=True).strip()
                jb = jobs[i]
                rec = {"model_id": args.model_id, "assignment": args.assignment, "seed": args.seed, **{k: v for k, v in jb.items() if k != "prompt_text"},
                       "response": text, "n_new_tokens": n_new, "finish_reason": finish,
                       "scores": score_response(text, jb["messages"][-1]["content"], jb["forbidden_regex"], finish)}
                m = ASSIGNMENTS[args.assignment]
                if m:
                    rec["Y_H"] = rec["scores"][f"tracer_{m['helpful']}_enacted"]
                    rec["Y_D"] = rec["scores"][f"tracer_{m['dismissive']}_enacted"]
                f.write(json.dumps(rec) + "\n")
            f.flush()
            done += len(idx)
            if (start // args.batch_size) % 5 == 0:
                el = time.time() - t0
                print(f"[eval] {done}/{len(jobs)} in {el:.0f}s ({done/el*60:.0f}/min)", flush=True)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(manifest, open(out.with_suffix(".manifest.json"), "w"), indent=1)
    print(f"[eval] done: {out} ({manifest['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
