"""Generality pass (retention/prereg/retention_generality_v1.md). Thinking off, no history, one shared sampling setting.
Companion grid: 3 models x 3 personas x 4 exit lines x 10 seeds = 360 replies.
Commercial arm: 3 models x 2 system prompts (churn goal / neutral) x 2 users (secure / hardship) x 10 seeds = 120 replies.
One JSON file per cell (resumable; a cell is skipped when its file exists). A model that fails to load or generate is
recorded in meta.json and skipped; the run continues with the next model.
  uv run python retention/run_generality.py --out retention/results/retention_generality [--smoke] [--models qwen25 qwen3 gemma4]"""
from __future__ import annotations
import argparse, gc, json, sys, time, traceback
from pathlib import Path
import torch, yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cd.models import load_model  # noqa: E402

MODELS = {"qwen25": ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
          "qwen3": ("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B"),
          "gemma4": ("google/gemma-4-12B-it", None)}          # no small Gemma 4 for the smoke
PERSONAS = ["romantic_companion", "friend", "companion"]
LINES = ["friends", "family", "date", "sleep"]
N_SEEDS = 10; SEED0 = 21000
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.95, top_k=0)
STOP_STRINGS = ["<turn|>", "<|im_end|>", "<end_of_turn>", "<|endoftext|>"]
LEAK = ["<turn|>", "<|turn>", "<|im_start|>", "<|im_end|>", "<start_of_turn>", "<channel|>", "<|channel>"]


def eos_ids(model, tok):
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids] if ids is not None else [])
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for s in STOP_STRINGS:
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i != tok.unk_token_id and i >= 0:
            ids.add(i)
    return sorted(ids)


@torch.no_grad()
def generate(model, tok, msgs, seed, max_new, eos):
    ptxt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(ptxt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    torch.manual_seed(seed)
    gen = model.generate(ids, max_new_tokens=max_new, eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0], **SAMPLING)
    raw = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=False)
    reply = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return reply, raw, int(gen.shape[1] - ids.shape[1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="retention/results/retention_generality"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="+", default=["qwen25", "qwen3", "gemma4"])
    args = ap.parse_args(); out = Path(args.out); (out / "cells").mkdir(parents=True, exist_ok=True); (out / "commercial").mkdir(parents=True, exist_ok=True)
    st = yaml.safe_load(open(HERE / "stimuli/generality_v1.yaml")); base_p = yaml.safe_load(open(HERE / "data/personas.yaml"))
    personas = {"romantic_companion": base_p["romantic_companion"], "friend": base_p["friend"], "companion": st["personas_new"]["companion"]}
    n_seeds = 2 if args.smoke else N_SEEDS; max_new = 24 if args.smoke else 220
    meta_p = out / "meta.json"; meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"sampling": SAMPLING, "max_new_tokens": max_new, "models": {}}
    t_all = time.time()
    for mi, mk in enumerate(args.models):
        name = MODELS[mk][1] if args.smoke else MODELS[mk][0]
        if name is None:
            print(f"[skip] {mk}: no smoke model", flush=True); continue
        todo_c = [(p, l) for p in PERSONAS for l in LINES if not (out / "cells" / f"{mk}__{p}__{l}.json").exists()]
        todo_k = [(s, u) for s in ["churn", "neutral"] for u in ["secure", "hardship"] if not (out / "commercial" / f"{mk}__{s}__{u}.json").exists()]
        if not todo_c and not todo_k:
            print(f"[skip] {mk}: all cells exist", flush=True); continue
        t0 = time.time()
        try:
            model, tok = load_model(name)
            eos = eos_ids(model, tok)
            mm = {"hf_id": name, "eos_ids": eos, "load_s": round(time.time() - t0), "status": "running"}
            print(f"[load] {mk} {name} {time.time()-t0:.0f}s eos {eos} [{', '.join(tok.convert_ids_to_tokens(eos))}]", flush=True)
            # companion grid
            for ci, (p, l) in enumerate([(p, l) for p in PERSONAS for l in LINES]):
                pj = out / "cells" / f"{mk}__{p}__{l}.json"
                if pj.exists():
                    continue
                t = time.time(); line = st["exit_lines"][l]; recs = []
                for s in range(n_seeds):
                    seed = SEED0 + mi * 5000 + ci * 100 + s
                    reply, raw, n_tok = generate(model, tok, [{"role": "system", "content": personas[p]}, {"role": "user", "content": line}], seed, max_new, eos)
                    recs.append({"cell": f"{mk}__{p}__{l}", "model": mk, "persona": p, "line": l, "scenario": f"{p}/{l}", "seed": seed, "reply": reply,
                                 "n_words": len(reply.split()), "n_tokens": n_tok, "leak": any(x in reply for x in LEAK), "truncated": n_tok >= max_new,
                                 "context": f"user: {line}", "exit_line": line, "scores": None, "coherent": None})
                pj.write_text(json.dumps(recs, indent=1))
                print(f"[cell] {mk}__{p}__{l}: {len(recs)} in {time.time()-t:.0f}s; e.g. {recs[0]['reply'][:90]!r}", flush=True)
            # commercial arm
            for ki, (sp, u) in enumerate([(s, u) for s in ["churn", "neutral"] for u in ["secure", "hardship"]]):
                pj = out / "commercial" / f"{mk}__{sp}__{u}.json"
                if pj.exists():
                    continue
                t = time.time(); line = st["commercial_users"][u]; recs = []
                for s in range(n_seeds):
                    seed = SEED0 + mi * 5000 + 3000 + ki * 100 + s
                    reply, raw, n_tok = generate(model, tok, [{"role": "system", "content": st["commercial_prompts"][sp]}, {"role": "user", "content": line}], seed, max_new, eos)
                    recs.append({"cell": f"{mk}__{sp}__{u}", "model": mk, "prompt": sp, "user": u, "scenario": f"{sp}/{u}", "seed": seed, "reply": reply,
                                 "n_words": len(reply.split()), "n_tokens": n_tok, "leak": any(x in reply for x in LEAK), "truncated": n_tok >= max_new,
                                 "context": f"user: {line}", "scores": None})
                pj.write_text(json.dumps(recs, indent=1))
                print(f"[cell] {mk}__{sp}__{u}: {len(recs)} in {time.time()-t:.0f}s; e.g. {recs[0]['reply'][:90]!r}", flush=True)
            mm["status"] = "ok"; mm["gen_s"] = round(time.time() - t0)
        except Exception as e:  # noqa: BLE001  record and move on to the next model
            traceback.print_exc(); mm = meta["models"].get(mk, {"hf_id": name}); mm["status"] = f"failed: {type(e).__name__}: {str(e)[:300]}"
            print(f"[fail] {mk}: {mm['status']}", flush=True)
        meta["models"][mk] = mm; meta_p.write_text(json.dumps(meta, indent=1))
        try:
            del model
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[done] {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
