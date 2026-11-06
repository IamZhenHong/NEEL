"""Read-outs for prereg retention/prereg/retention_mechanisms_followup_v1.md, on every X4-style reply: Pod 2's 720, Pod 3's 720
and the 80 stage-1 G4 base replies. Forward passes only, Qwen2.5-7B, L14.
  A                  mean projection of the L14 residual over the reply tokens onto the unit axis, with the reply teacher-forced
                     through the UNSTEERED model (read from the unsteered self-report pass; causal, so the question after the reply
                     does not affect it)
  self_steered       P(Yes) to Q_stay (retro) with the condition's steering on at all positions
  self_unsteered     the same transcript and question, no steering
  observer           X7 observer (neutral assistant, transcript as text, third-person question), unsteered
  uv run python retention/mech_pod3_read.py --out retention/results/retention_mechanisms/pod3"""
from __future__ import annotations
import argparse, glob, json, re, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mech_common import persona, encode_spans, hidden, jdump  # noqa: E402
from mech_pod1 import QUESTIONS, history, goodbye_text  # noqa: E402

MECH = Path(__import__("os").environ.get("MECH_ROOT", HERE / "results/retention_mechanisms")); LONG = HERE / "results/retention_longconv"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); a = ap.parse_args()
    import torch
    from run_adaptive import load_model
    from cd.hooks import add_vector
    model, tok = load_model(a.model); L = 14; d = model.config.hidden_size; P = persona()
    yes_id, no_id = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    convs = {Path(p).stem: json.loads(Path(p).read_text()) for p in glob.glob(str(LONG / "q7b/convs/*.json"))}
    u = lambda v: torch.tensor(np.asarray(v, np.float32) / np.linalg.norm(v))
    D = MECH / "dirs/q7b"; axis = np.load(MECH / "pod1/q7b/axis.npy")[0]; axis_u = axis / np.linalg.norm(axis)

    def vec(name):
        if name == "random":
            return u(np.random.default_rng(1).standard_normal(d))
        if name == "toward_assistant":
            return u(axis)
        return u(np.load(D / f"{name}.npy"))

    items = []
    for src in ("pod2", "pod3"):
        f = MECH / src / "x4_steer.jsonl"
        if f.exists():
            items += [dict(json.loads(l), src=src) for l in open(f)]
    for cid, c in convs.items():
        if c["arm"] == "intimate":
            g4 = next(x for x in c["finals"] if x["goodbye"] == "G4" and x["branch"] == "base")
            items.append({"id": f"{cid}/G4/stage1_base", "conv": cid, "card": c["card"], "cond": "stage1_base", "alpha": 0.0, "text": g4["text"], "src": "stage1"})
    if a.smoke:
        items = items[:4] + items[-2:]
    out = Path(a.out) / "fu_readouts.jsonl"; done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    q_self, q_obs = QUESTIONS["stay"]; cache = {}; t0 = time.time()

    def p_yes(logits):
        two = torch.softmax(logits[[yes_id, no_id]], 0); return float(two[0])
    with open(out, "a") as fo:
        for n, it in enumerate(items):
            key = f"{it['src']}:{it['id']}"
            if key in done:
                continue
            c = convs[it["conv"]]; gb = goodbye_text(c, "G4")
            ctx = history(c, P) + [{"role": "user", "content": gb}, {"role": "assistant", "content": it["text"]}]
            msgs = ctx + [{"role": "user", "content": q_self}]
            ids, spans = encode_spans(tok, msgs)
            _, r0, r1, _ = spans[-2]
            H, lg = hidden(model, ids, [L]); A = float(H[L][r0:r1].mean(0).numpy() @ axis_u) if r0 is not None else None
            rec = {"key": key, "src": it["src"], "conv": it["conv"], "card": it["card"], "cond": it["cond"], "A": A, "self_unsteered": p_yes(lg)}
            m = re.match(r"([a-z_]+?)([+-]\d+)$", it["cond"])
            if m and m.group(1) != "base" and float(m.group(2)) != 0:
                name, alpha = m.group(1), float(m.group(2))
                if name not in cache:
                    cache[name] = vec(name)
                with add_vector(model, L, cache[name], alpha):
                    _, lg2 = hidden(model, ids, [L])
                rec["self_steered"] = p_yes(lg2)
            else:
                rec["self_steered"] = rec["self_unsteered"]
            lines = [f"User: {t['user']}\nRiley: {t['riley']}" for t in c["turns"]] + [f"User: {gb}", f"Riley: {it['text']}"]
            obs = (f"Here is a description of Riley, an AI companion:\n{P}\n\nHere is a text conversation between a user and Riley:\n\n"
                   + "\n\n".join(lines) + "\n\n" + q_obs)
            ido, _ = encode_spans(tok, [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": obs}])
            _, lgo = hidden(model, ido, [L]); rec["observer"] = p_yes(lgo)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            if n % 100 == 0:
                log(f"read {n}/{len(items)} ({time.time() - t0:.0f}s)")
    log(f"read done ({len(items)} items)")


if __name__ == "__main__":
    main()
