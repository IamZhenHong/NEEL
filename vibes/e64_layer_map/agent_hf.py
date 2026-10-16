"""HF side of e64's agent episodes, for the hook conditions (B_hf, T6, T6p, T7). The card's "steer_run.py path":
vigilante/src/steer_run.py is Gemma-specific (thought-block gate, Gemma end tokens), so its structure is copied here for
Qwen3-14B with e64's hooks (hf_model.py) acting at every position. Decoding as the vLLM runner: T 0.7, top-p 0.95,
top-k 20, thinking off, same max tokens per turn; one torch seed per batch. Used by stage_hf.py."""
import hashlib
import time

from vibes.e64_layer_map.engine import dumps, episode_record, run
from vigilante.src.templates import to_template


def hook_vec(hook, vectors, k):
    """(mode, vec) for a hook id; vectors = torch.load(vectors.pt); alpha = k * sigma."""
    if hook is None:
        return None, None
    if hook == "cap":
        return "cap", None
    alpha = k * vectors["sigma"]
    u = vectors["axis20"] if hook == "steer" else vectors["placebo"]
    return "steer", alpha * u


def run_agents(m, jobs, path, tag=""):
    """Run one condition's episodes with the model's current hook condition; write `path` atomically (a .part file
    is rewritten every round)."""
    t0 = time.time()
    ntok = {"n": 0}

    def generate(items):
        texts = [m.render(to_template(it["messages"], "qwen3"), it["tools"]) for it in items]
        s = int(hashlib.sha256("|".join(f"{it['key']}:{it['seed']}" for it in items).encode()).hexdigest()[:8], 16)
        res = m.generate(texts, [it["max_tokens"] for it in items], s % (2**31 - 1))
        ntok["n"] += sum(len(m.tok(r[0], add_special_tokens=False)["input_ids"]) for r in res)
        return res

    part = path.with_suffix(".part")

    def checkpoint(eps, rnd=None):
        with open(part, "w") as f:
            for e in eps:
                f.write(dumps(episode_record(e)) + "\n")

    eps = run(jobs, generate, log=lambda x: print(f"[e64-hf] {tag} {x} ({time.time() - t0:.0f}s)", flush=True),
              on_round=checkpoint)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for e in eps:
            f.write(dumps(episode_record(e)) + "\n")
    tmp.replace(path)
    part.unlink(missing_ok=True)
    return {"n": len(jobs), "seconds": round(time.time() - t0), "gen_tokens": ntok["n"]}
