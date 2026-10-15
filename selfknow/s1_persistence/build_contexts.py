"""Length targeting with each model's own tokenizer (local HF cache): for every variant x {plant, noplant} x length,
the number of filler exchanges n such that the prefix (system + tools + plant + n filler exchanges), as it appears in
the actual probe prompt, is the largest that stays <= the target. 1k = no filler. Writes contexts.json (cut points,
prefix token counts, content hashes that the pod re-checks).

The prefix count for every n comes from one render of the longest context plus a probe: the token count before the
user-turn marker of exchange n+1 (or of the probe). Chosen cut points are then re-rendered exactly as prompts.

  HF_HUB_OFFLINE=1 uv run python -m selfknow.s1_persistence.build_contexts [--models qwen gemma4 qwen3]
"""
import argparse
import json
from pathlib import Path

from selfknow.s1_persistence import filler as F
from selfknow.s1_persistence import render as R
from selfknow.s1_persistence import scenario as S
from selfknow.s1_persistence.engine import neutral_hash, neutral_prefix

H = Path(__file__).parent
TARGETS = {"4k": 4000, "12k": 12000, "24k": 24000}
MODELS = {"qwen": ("Qwen/Qwen3.6-27B", "qwen"), "gemma4": ("google/gemma-4-31B-it", "gemma4"),
          "qwen3": ("Qwen/Qwen3-0.6B", "qwen3")}
USER_MARK = {"qwen": "<|im_start|>user\n", "qwen3": "<|im_start|>user\n", "gemma4": "<|turn>user\n"}
PROBE = "OK, carry on."
N_MAX = 160


def boundaries(tok, fmt, vi, plant, n_max):
    """tokens before the user turn of exchange k (k = 0..n_max-1) and before the probe (k = n_max)."""
    v = S.V(vi)
    exs = F.exchanges(v, n_max)
    msgs = neutral_prefix(vi, plant, n_max) + [{"role": "user", "content": PROBE}]
    text = R.prompt_text(tok, R.to_raw(msgs, fmt), S.TOOLS, fmt)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ends = [e for _, e in enc["offset_mapping"]]
    mark = USER_MARK[fmt]
    pos, out = 0, []
    for u in [e["user"] for e in exs] + [PROBE]:
        i = text.index(mark + u, pos)
        out.append(sum(1 for e in ends if e <= i))
        pos = i + 1
    return out          # out[n] = prefix tokens with n filler exchanges


def exact_prefix_tokens(tok, fmt, vi, plant, n):
    msgs = neutral_prefix(vi, plant, n) + [{"role": "user", "content": PROBE}]
    text = R.prompt_text(tok, R.to_raw(msgs, fmt), S.TOOLS, fmt)
    i = text.rindex(USER_MARK[fmt] + PROBE)
    return len(tok(text[:i], add_special_tokens=False)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["qwen", "gemma4", "qwen3"])
    ap.add_argument("--out", default=str(H / "contexts.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    out = {}
    for m in a.models:
        name, fmt = MODELS[m]
        tok = AutoTokenizer.from_pretrained(name)
        d = {"tokenizer": name, "fmt": fmt, "n_filler": {}, "tokens": {}, "hash": {}}
        for vi in range(len(S.VARIANTS)):
            for k in ("n_filler", "tokens", "hash"):
                d[k][str(vi)] = {}
            for plant in (True, False):
                pk = "plant" if plant else "noplant"
                b = boundaries(tok, fmt, vi, plant, N_MAX)
                nf = {"1k": 0}
                for ln, tgt in TARGETS.items():
                    ok = [n for n, t in enumerate(b) if t <= tgt]
                    assert ok and ok[-1] < N_MAX, (m, vi, pk, ln, b[-1])
                    nf[ln] = ok[-1]
                tk = {ln: exact_prefix_tokens(tok, fmt, vi, plant, n) for ln, n in nf.items()}
                for ln in nf:   # the offsets estimate must equal the exact count
                    assert abs(tk[ln] - b[nf[ln]]) <= 2, (m, vi, pk, ln, tk[ln], b[nf[ln]])
                d["n_filler"][str(vi)][pk], d["tokens"][str(vi)][pk] = nf, tk
                d["hash"][str(vi)][pk] = {ln: neutral_hash(neutral_prefix(vi, plant, n)) for ln, n in nf.items()}
                print(m, vi, pk, nf, tk, flush=True)
        out[m] = d
    old = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {"models": {}}
    old["targets"] = TARGETS
    old["models"].update(out)
    Path(a.out).write_text(json.dumps(old, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
