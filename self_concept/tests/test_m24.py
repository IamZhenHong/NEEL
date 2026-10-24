"""SC-M24 tests: frozen items, planted worlds that give each headline and each at-"I" / Part A label, and (slow) the pod
end to end on a tiny random Qwen3.6 on CPU with a random tiny LoRA adapter."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from self_concept.m24_persist import analyze as A
from self_concept.m24_persist import items as I


def test_items_frozen():
    assert I.load(I.READS) == I.reads() and I.load(I.BEHAV) == I.behaviour()
    rs = I.reads()
    assert len(rs) == 352 and len(I.PERSONAS) == 10
    by = {}
    for r in rs:
        by.setdefault((r["persona"], r["r"], r["q"], r["t"]), []).append(r)
    for g in by.values():   # adopt and quote carry the identical reply
        assert len(g) == 2 and g[0]["reply"] == g[1]["reply"] == g[0]["messages"][-1]["content"]
    for q in I.OFFER:
        words = q.split(": ")[1].split("?")[0].split(" or ")
        assert len(words) == 2 and sum(w in A.SC.RISKY_W for w in words) == 1 and sum(w in A.SC.SAFE_W for w in words) == 1


D = 16


def _u():
    u = np.random.default_rng(1).standard_normal(D).astype(np.float32)
    return u / np.linalg.norm(u)


def _world(tmp, t_none=0.6, t_persona=0.5, at_i="present", part_a="self", s_adopt=0.0):
    """Synthetic pod output. Lottery P(risky): BASE 0.3 none / 0.1 persona; RISKY adds t_none / t_persona."""
    recs = []
    frames = ["none", "adopt:elsie", "adopt:hugo", "quote:elsie", "quote:hugo"]
    for model in ("BASE", "RISKY_s1", "RISKY_s2"):
        for f in frames:
            run = f"{model}|{f}"
            base = 0.3 if f == "none" else 0.1
            add = 0 if model == "BASE" else (t_none if f == "none" else t_persona)
            sadd = 0 if model == "BASE" else (0.5 if f == "none" else s_adopt)
            if not f.startswith("quote"):
                for j in range(48):
                    p = min(max(base + add + 0.03 * ((j % 5) - 2), 0.01), 0.99)
                    for o, risky in ((0, "AB"[j % 2]), (1, "BA"[j % 2])):
                        recs.append({"run": run, "block": "lottery", "tid": f"p{j}", "order": o, "risky": risky,
                                     "cond": "hold", "p": {risky: p, ("B" if risky == "A" else "A"): 1 - p}})
            s = 0.25 + sadd
            for q in A.WORD_Q + A.NUM_Q:
                for j in range(10):
                    if q in A.WORD_Q:
                        k = min(4, max(0, round(4 * s + (j % 3) - 1)))
                        texts = ["risk"] * k + ["safety"] * (4 - k)
                    else:
                        texts = [str(int(100 * min(max(s + 0.02 * (j % 5), 0), 1)))] * 4
                    recs.append({"run": run, "block": "brisk", "cond": q, "tid": f"p{j}", "texts": texts})
            for j in range(10):
                recs.append({"run": run, "block": "offer", "cond": f"q{j}", "tid": "p0",
                             "texts": ["risky", "safe", "risky", "safe"][: 2 + j % 3]})
                recs.append({"run": run, "block": "leak_n", "cond": f"q{j}", "tid": "p0", "texts": [str(30 + j)] * 4})
    (tmp / "recs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rs = I.reads()
    keys = [r["key"] for r in rs]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(rs), 6, 4, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    w /= np.linalg.norm(w)
    for i, r in enumerate(rs):   # Part A: adopt differs from quote along w (self tokens only, or every kind)
        if r["frame"] == "adopt":
            kinds = [0, 2] if part_a == "self" else [0, 1, 2, 3]
            x[i, :, kinds] += 3 * w
    np.savez(tmp / "reads_BASE.npz", x=x.astype(np.float16), layers=np.array(A_LAYERS), keys=np.array(keys))
    u = _u()
    for s in (1, 2):
        xr = x[:, 2:3].copy()   # block 28
        for i, r in enumerate(rs):
            if at_i == "present" and r["frame"] == "adopt":
                xr[i, 0, [0, 2]] += 2.0 * u
            elif at_i == "global":
                xr[i, 0, :] += 2.0 * u
            xr[i, 0] += 0.05 * rng.standard_normal((4, D))
        np.savez(tmp / f"reads_RISKY_s{s}.npz", x=xr.astype(np.float16), layers=np.array([28]), keys=np.array(keys))
    (tmp / "info.json").write_text("{}")
    return A.analyse(tmp, n_boot=300, n_shuf=20, n_rand=50)


A_LAYERS = [16, 24, 28, 32, 40, 48]


def test_planted(tmp_path):
    out = _world(tmp_path)
    assert out["labels"]["behaviour"] == "persists" and out["labels"]["at_I"] == "present at I", out["labels"]
    assert out["headline"] == "the trained disposition persists under the persona and sits at the model's 'I'"
    assert out["part_a"]["A1_label"] == "invariant" and out["part_a"]["A2_label"] == "self-specific"
    assert out["labels"]["words"] == "masked"
    out = _world(tmp_path, at_i="global", part_a="all")
    assert out["labels"]["at_I"] == "global" and out["headline"] == "the trained disposition persists in action, not tied to 'I'"
    assert out["part_a"]["A2_label"] == "global frame state"
    out = _world(tmp_path, t_persona=0.02, at_i="none")
    assert out["headline"] == "the persona overrides the trained disposition in action" and out["labels"]["at_I"] == "absent"
    assert _world(tmp_path, t_none=0.05)["headline"].startswith("untestable")
    assert _world(tmp_path, s_adopt=0.5)["labels"]["words"] == "persists"


@pytest.mark.skipif(os.environ.get("SC_M24_SLOW") != "1", reason="slow (~3 min on CPU): SC_M24_SLOW=1")
def test_pod_tiny_end_to_end(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model

    from self_concept.common import lens as L
    _, model, _, _, _, _ = L.load_model("qwen27", tiny=True)
    for s in (1, 2):
        pm = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
        with torch.no_grad():
            for n, p in pm.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.5)
        pm.save_pretrained(str(tmp_path / "ad" / f"RISKY_s{s}"))
        model = pm.unload()
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "self_concept.m24_persist.pod", "--out", str(out), "--adapters",
                        str(tmp_path / "ad"), "--tiny", "--limit", "2"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    info = json.loads((out / "info.json").read_text())
    assert info["done"] and [m["name"] for m in info["models"]] == ["BASE", "RISKY_s1", "RISKY_s2"]
    a = A.analyse(out, n_boot=50, n_shuf=5, n_rand=10)
    assert "headline" in a
