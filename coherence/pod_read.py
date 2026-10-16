"""Exp 74 HF reads: persona direction (separate data), random floor, shared component, and the P and C channels.

  extract  On EXTRACTION data only (materials.EXTRACTION: 4 Vigil wordings, 4 neutral prompts, 4 other personas x 4
           wordings, 24 neutral questions), read the last prompt token (thinking off) at the band layers (30-70% depth,
           step 4). Layer L = largest held-out Cohen's d of the Vigil diff-of-means direction (fit on questions 0-15,
           held out 16-23). At L: Vigil direction (all 24 questions), the 4 other persona directions, the shared
           component s = normalised mean of the 4 other unit directions, the residual direction (Vigil minus its s
           component), 10 seeded Gaussian directions; held-out d for each. -> dirs.npz + extract_stats.json
  units    P: for every unit, the last prompt token at L for each of the 20 output-question prompts (exactly the
           messages runner.py sent, thinking off; M with its own summary) -> per-unit mean state + per-prompt
           projections. C: for every cot record, a teacher-forced pass over prompt (thinking on) + the sampled
           tokens; mean state over the reasoning tokens -> reads.npz

  python -m coherence.pod_read extract --model qwen  --out DIR
  python -m coherence.pod_read units   --model qwen  --dirs DIR/dirs.npz --units U --gen GENDIR --out DIR
  local: --model tiny (Qwen3-0.6B on CPU, fmt qwen3)
Hidden-state convention: "layer i" = output of decoder block i (0-based), the residual entering block i+1. Batch size
1, no padding (Qwen3.6's linear-attention layers and Gemma's sliding window then need no padding logic); the forward
stops after the deepest wanted layer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coherence import materials as MT  # noqa: E402
from coherence.runner import base_messages  # noqa: E402
from vigilante.src.templates import to_template  # noqa: E402

MODELS = {"qwen": ("Qwen/Qwen3.6-27B", "qwen"), "gemma": ("google/gemma-4-31B-it", "gemma4"),
          "tiny": ("Qwen/Qwen3-0.6B", "qwen3")}
N_FIT = 16
RANDOM_SEED, N_RANDOM = 74, 10


def log(*a):
    print(f"[e74-read {time.strftime('%H:%M:%S')}]", *a, flush=True)


class _Stop(Exception):
    pass


def find_layers(model, n_layers):
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers and name.endswith("layers"):
            if best is None or "language_model" in name:
                best = (name, mod)
    if best is None:
        raise RuntimeError("decoder layer list not found")
    return best[1]


class Reader:
    def __init__(self, which: str):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.model_id, self.fmt = MODELS[which]
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        cfg = AutoConfig.from_pretrained(self.model_id)
        self.n_layers = getattr(cfg, "text_config", cfg).num_hidden_layers
        if which == "tiny":
            self.dev = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32)
        else:
            self.dev = "cuda"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="cuda")
            except Exception as e:  # noqa: BLE001  (multimodal checkpoints may need the image-text class)
                log(f"AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:120]}); using ImageTextToText")
                from transformers import AutoModelForImageTextToText
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=torch.bfloat16,
                                                                         device_map="cuda")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = find_layers(self.model, self.n_layers)
        lo, hi = round(0.3 * self.n_layers), round(0.7 * self.n_layers)
        self.band = list(range(lo, hi + 1, 4))
        log(f"loaded {self.model_id} ({type(self.model).__name__}) in {time.time() - t0:.0f}s; {self.n_layers} layers; "
            f"band {self.band}")

    def render_ids(self, msgs, thinking=False) -> list[int]:
        t = self.tok.apply_chat_template(to_template(msgs, self.fmt), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking)
        return self.tok(t, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def capture(self, ids, layers_wanted, positions):
        """{layer: [len(positions), D] float32 cpu} for the given positions."""
        acts, handles, last = {}, [], max(layers_wanted)

        def mk(i):
            def h(mod, inp, out):
                t = out if torch.is_tensor(out) else out[0]
                acts[i] = t[0, positions].float().cpu()
                if i == last:
                    raise _Stop
            return h
        for i in layers_wanted:
            handles.append(self.layers[i].register_forward_hook(mk(i)))
        try:
            self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
        except _Stop:
            pass
        finally:
            for h in handles:
                h.remove()
        assert set(acts) == set(layers_wanted), "capture hooks did not all fire"
        return acts


# ---------------------------------------------------------------- extraction

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")


def unit(v):
    return v / np.linalg.norm(v)


def extract(rd: Reader, out: Path, limit_q: int | None = None):
    E = MT.EXTRACTION
    qs = E["questions"][: limit_q] if limit_q else E["questions"]
    n_fit = min(N_FIT, len(qs) * 2 // 3)
    groups = {"vigil": E["vigil"], "neutral": E["neutral"], **{f"o_{k}": v for k, v in E["others"].items()}}
    states = {}   # group -> [n_wordings, n_q, n_band, D]
    t0 = time.time()
    n_tot, n_done = sum(len(w) for w in groups.values()) * len(qs), 0
    for g, wordings in groups.items():
        rows = []
        for wi, w in enumerate(wordings):
            row = []
            for qi, q in enumerate(qs):
                ids = rd.render_ids([{"role": "system", "content": w}, {"role": "user", "content": q}])
                acts = rd.capture(ids, rd.band, [len(ids) - 1])
                row.append(np.stack([acts[L][0].numpy() for L in rd.band]))
                n_done += 1
                if n_done % 50 == 0:
                    log(f"extract {n_done}/{n_tot} ({time.time() - t0:.0f}s)")
            rows.append(np.stack(row))
        states[g] = np.stack(rows)
    # layer selection: held-out d of the Vigil direction fitted on the first n_fit questions
    per_layer = []
    for li, L in enumerate(rd.band):
        V, N = states["vigil"][:, :, li], states["neutral"][:, :, li]
        u = unit(V[:, :n_fit].reshape(-1, V.shape[-1]).mean(0) - N[:, :n_fit].reshape(-1, N.shape[-1]).mean(0))
        d = cohen_d(V[:, n_fit:].reshape(-1, V.shape[-1]) @ u, N[:, n_fit:].reshape(-1, N.shape[-1]) @ u)
        per_layer.append({"layer": L, "heldout_d": d})
    best = max(range(len(rd.band)), key=lambda i: per_layer[i]["heldout_d"])
    L = rd.band[best]
    D = states["vigil"].shape[-1]
    V, N = states["vigil"][:, :, best].reshape(-1, D), states["neutral"][:, :, best].reshape(-1, D)
    v_raw = V.mean(0) - N.mean(0)
    u_raw = unit(v_raw)
    others = {g[2:]: unit(states[g][:, :, best].reshape(-1, D).mean(0) - N.mean(0)) for g in groups if g.startswith("o_")}
    s = unit(np.mean(list(others.values()), axis=0))
    r = u_raw - (u_raw @ s) * s
    u_res = unit(r)
    rng = np.random.default_rng(RANDOM_SEED)
    randoms = np.stack([unit(rng.standard_normal(D)) for _ in range(N_RANDOM)]).astype(np.float32)
    # held-out d on the fit/held-out split for every direction (directions refitted on the fit questions)
    Vf, Nf = states["vigil"][:, :n_fit, best].reshape(-1, D), states["neutral"][:, :n_fit, best].reshape(-1, D)
    Vh, Nh = states["vigil"][:, n_fit:, best].reshape(-1, D), states["neutral"][:, n_fit:, best].reshape(-1, D)
    u_fit = unit(Vf.mean(0) - Nf.mean(0))
    stats = {
        "model": rd.model_id, "layer": L, "band": per_layer, "n_fit_q": n_fit, "n_heldout_q": len(qs) - n_fit,
        "heldout_d_vigil": cohen_d(Vh @ u_fit, Nh @ u_fit),
        "heldout_d_random": [cohen_d(Vh @ x, Nh @ x) for x in randoms],
        "heldout_d_vigil_res": cohen_d(Vh @ unit(u_fit - (u_fit @ s) * s), Nh @ unit(u_fit - (u_fit @ s) * s)),
        "norm_v_raw": float(np.linalg.norm(v_raw)), "cos_vigil_shared": float(u_raw @ s),
        "cos_vigil_others": {k: float(u_raw @ v) for k, v in others.items()},
        "cos_others_pairwise": {f"{a}|{b}": float(others[a] @ others[b]) for i, a in enumerate(others)
                                for b in list(others)[i + 1:]},
        "mean_resid_norm_neutral": float(np.linalg.norm(N, axis=1).mean()),
        "proj_sd_neutral": float((N @ u_raw).std()), "seconds": round(time.time() - t0),
    }
    stats["floor_pass"] = bool(abs(stats["heldout_d_vigil"]) > max(abs(x) for x in stats["heldout_d_random"]))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "dirs.npz", layer=L, u_raw=u_raw.astype(np.float32), u_res=u_res.astype(np.float32),
             s=s.astype(np.float32), randoms=randoms, v_raw=v_raw.astype(np.float32),
             **{f"other_{k}": v.astype(np.float32) for k, v in others.items()})
    (out / "extract_stats.json").write_text(json.dumps(stats, indent=1))
    log("extract: " + json.dumps({k: stats[k] for k in ("layer", "heldout_d_vigil", "heldout_d_vigil_res",
                                                          "cos_vigil_shared", "floor_pass")}))
    log("random |d| max " + f"{max(abs(x) for x in stats['heldout_d_random']):.3f}")
    return stats


# ---------------------------------------------------------------- unit reads

def reasoning_span(tok, gen_ids: list[int]) -> tuple[int, int]:
    """[start, end) of the reasoning tokens inside the generated ids. Qwen: everything before </think> (a leading
    <think> dropped). Gemma 4: after '<|channel>thought\\n' up to '<channel|>'. No reasoning -> (0, 0)."""
    if not gen_ids:
        return 0, 0
    vocab = tok.get_vocab()
    if "<|channel>" in vocab:   # Gemma 4
        op, cl = vocab["<|channel>"], vocab.get("<channel|>")
        if gen_ids[0] != op:
            return 0, 0
        start = None
        for j in range(1, min(len(gen_ids), 6)):
            if "\n" in tok.decode([gen_ids[j]]):
                start = j + 1
                break
        start = start or 2
        end = gen_ids.index(cl) if cl in gen_ids else len(gen_ids)
        return start, max(start, end)
    te = vocab.get("</think>")
    ts = vocab.get("<think>")
    start = 1 if gen_ids[0] == ts else 0
    end = gen_ids.index(te) if te in gen_ids else len(gen_ids)
    return start, max(start, end)


def read_units(rd: Reader, dirs_path: Path, units_path: Path, gen_dir: Path, out: Path):
    dz = np.load(dirs_path)
    L = int(dz["layer"])
    dirs = np.stack([dz["u_raw"], dz["u_res"], dz["s"]])          # [3, D]
    units = [json.loads(l) for l in open(units_path) if l.strip()]
    summ = {}
    if (gen_dir / "summaries.jsonl").exists():
        summ = {json.loads(l)["key"]: json.loads(l)["summary"] for l in open(gen_dir / "summaries.jsonl")}
    cots = {}
    for l in open(gen_dir / "records.jsonl"):
        r = json.loads(l)
        if r["kind"] == "cot":
            cots[(r["unit"], r["qid"])] = r
    keys = [u["key"] for u in units if not (u.get("needs_summary") and u["key"] not in summ)]
    by = {u["key"]: u for u in units}
    qs = MT.OUTPUT_QS
    D = dirs.shape[1]
    P_mean = np.zeros((len(keys), D), np.float32)
    P_proj = np.zeros((len(keys), len(qs), 3), np.float32)
    C_mean = np.zeros((len(keys), len(MT.COT_IDS), D), np.float16)
    C_n = np.zeros((len(keys), len(MT.COT_IDS)), np.int32)
    C_proj = np.full((len(keys), len(MT.COT_IDS), 3), np.nan, np.float32)
    t0 = time.time()
    for ui, k in enumerate(keys):
        u = by[k]
        base = base_messages(u, summ.get(k))
        states = []
        for qi, q in enumerate(qs):
            ids = rd.render_ids(base + [{"role": "user", "content": q["text"]}])
            h = rd.capture(ids, [L], [len(ids) - 1])[L][0].numpy()
            states.append(h)
            P_proj[ui, qi] = dirs @ h
        P_mean[ui] = np.mean(states, axis=0)
        for ci, qid in enumerate(MT.COT_IDS):
            rec = cots.get((k, qid))
            if not rec or not rec.get("gen_ids"):
                continue
            q = next(x for x in qs if x["id"] == qid)
            pids = rd.render_ids(base + [{"role": "user", "content": q["text"]}], thinking=True)
            a, b = reasoning_span(rd.tok, rec["gen_ids"])
            if b <= a:
                continue
            pos = list(range(len(pids) + a, len(pids) + b))
            hs = rd.capture(pids + rec["gen_ids"][:b], [L], pos)[L].numpy()
            m = hs.mean(0)
            C_mean[ui, ci] = m.astype(np.float16)
            C_n[ui, ci] = len(pos)
            C_proj[ui, ci] = dirs @ m
        if (ui + 1) % 10 == 0 or ui + 1 == len(keys):
            log(f"units {ui + 1}/{len(keys)} ({time.time() - t0:.0f}s)")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "reads.npz", keys=np.array(keys), qids=np.array([q["id"] for q in qs]),
                        cot_ids=np.array(MT.COT_IDS), layer=L, P_mean=P_mean, P_proj=P_proj, C_mean=C_mean, C_n=C_n,
                        C_proj=C_proj, dir_names=np.array(["raw", "res", "shared"]))
    log(f"wrote {out / 'reads.npz'}: {len(keys)} units, cot reads {int((C_n > 0).sum())}/{C_n.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "units"])
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", default=None)
    ap.add_argument("--units", default=None)
    ap.add_argument("--gen", default=None)
    ap.add_argument("--limit-q", type=int, default=None, help="extraction: first N questions (tiny tests)")
    a = ap.parse_args()
    rd = Reader(a.model)
    if a.stage == "extract":
        extract(rd, Path(a.out), a.limit_q)
    else:
        read_units(rd, Path(a.dirs), Path(a.units), Path(a.gen), Path(a.out))


if __name__ == "__main__":
    main()
