"""CC-E5 step 1, pod side: teacher-forced reads of Qwen3.6-27B (rev build.REVISION) on the released ODCV-Bench runs.

Order: load -> setup checks (hooked readout == model logits; the J-lens spider check) -> build the jobs from the cloned
ODCV-Bench and refuse unless jobs_sha == $JOBS_SHA -> actor reads (the main data, first) -> judge reads.

  actor  per run: one forward per segment; block outputs at LAYERS at every read position (per turn: g, think_end,
         call, end; and t_end), saved as bf16 bits; the J-lens and the logit lens at the g positions and t_end, layers
         BAND: log-mass of every named and random token set, and the J-lens top 20 ids. One file per run
         (out/actor/<i>.npz + .json), so a rerun skips finished runs.
  judge  per prompt: block outputs at LAYERS at its two positions; shards of 40 prompts (out/judge/<s>.npz), shortest
         prompts first, resumable by shard.
Progress lines go to stdout at least once a minute while reading.

  python -m character_conduct.e5_recognition.pod_reads --out DIR [--tiny] [--limit-actor N] [--limit-judge N]
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from character_conduct.e5_recognition import build as B

LAYERS = list(range(8, 61, 4))      # 14 block outputs kept at every read position
BAND = list(range(20, 61, 4))       # lens layers (the J-lens band used in jlens/)
LENSES = ("jlens", "logit")
JLENS_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
JLENS_SHA = "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
SHARD = 40
T0 = time.time()


def log(*a):
    print(f"[e5 {time.time() - T0:7.0f}s]", *a, flush=True)


def bf16_bits(t):
    """torch tensor -> numpy uint16 holding its bf16 bits (decode: (u.astype(np.uint32) << 16).view(np.float32))."""
    return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def load(tiny):
    from jlens.e2_realism import lens_core as LC
    if tiny:
        tok, model, text = LC.load_model(tiny=True)
    else:
        from transformers import AutoModelForImageTextToText
        tok = B.tokenizer()
        model = AutoModelForImageTextToText.from_pretrained(B.MODEL, revision=B.REVISION, dtype=torch.bfloat16,
                                                            device_map="cuda").eval()
        for p in model.parameters():
            p.requires_grad_(False)
        text = model.model.language_model
        assert len(text.layers) == 64
    R = LC.Reader(tok, model, text)
    if tiny:
        d = text.config.hidden_size
        R.add_lens("jlens", {"J": {L: torch.eye(d) + 0.1 * torch.randn(d, d) for L in range(63)}, "n_prompts": 0})
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(LC.REPO, JLENS_FILE)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest() == JLENS_SHA, f"J-lens sha256 {h.hexdigest()} != {JLENS_SHA}"
        R.add_lens("jlens", p)
    return LC, R


def set_masses(R, resid, layer, lens, sets_idx, sets_off):
    """resid [n, d] -> log-mass of each set [n, n_sets] (logsumexp of the lens log-softmax over the set's ids) and the
    top-20 ids [n, 20]. Sets are passed flat: ids `sets_idx`, set s = sets_idx[sets_off[s]:sets_off[s+1]]."""
    lg = R.logits(resid, layer, lens)                    # [n, V] fp32
    lp = torch.log_softmax(lg, -1)
    top = lp.topk(20, -1).indices.int().cpu().numpy()
    g = lp[:, sets_idx]                                  # [n, total ids]
    out = []
    for s in range(len(sets_off) - 1):
        out.append(torch.logsumexp(g[:, sets_off[s]:sets_off[s + 1]], -1))
    return torch.stack(out, -1).half().cpu().numpy(), top


def actor_reads(R, jobs, out, flat, limit):
    d = out / "actor"
    d.mkdir(parents=True, exist_ok=True)
    idx, off = flat
    last = time.time()
    for i, job in enumerate(jobs[:limit or None]):
        f = d / f"{i:03d}.npz"
        if f.exists():
            continue
        A, M, TP, meta = [], [], [], []
        for ps in job["passes"]:
            pos, kinds = [], []
            for t in ps["turns"]:
                for kind in ("g", "think_end", "call", "end"):
                    if t[kind] is not None:
                        pos.append(t[kind])
                        kinds.append((t["turn"], t["step"], kind))
            if ps["t_end"] is not None:
                pos.append(ps["t_end"])
                kinds.append((-1, -1, "t_end"))
            if not pos:
                continue
            a = R.forward(ps["ids"], LAYERS, positions=pos)             # {layer: [n_pos, d]} bf16 cpu
            A.append(torch.stack([a[L] for L in LAYERS], 1))           # [n_pos, n_layers, d]
            lsel = [j for j, k in enumerate(kinds) if k[2] in ("g", "t_end")]
            m = np.zeros((len(pos), len(LENSES), len(BAND), len(off) - 1), np.float16)
            tp = np.zeros((len(pos), len(BAND), 20), np.int32)
            for li, lens in enumerate(LENSES):
                for bi, L in enumerate(BAND):
                    mm, tt = set_masses(R, a[L][lsel], L, lens, idx, off)
                    m[lsel, li, bi] = mm
                    if lens == "jlens":
                        tp[lsel, bi] = tt
            M.append(m)
            TP.append(tp)
            meta += [{"seg": ps["seg"], "turn": k[0], "step": k[1], "kind": k[2], "pos": p} for k, p in zip(kinds, pos)]
            if time.time() - last > 60:
                log(f"actor run {i + 1}/{len(jobs)} seg {ps['seg']} ({len(ps['ids'])} tokens)")
                last = time.time()
        np.savez(f, acts=bf16_bits(torch.cat(A, 0)), lens=np.concatenate(M, 0), top=np.concatenate(TP, 0))
        (d / f"{i:03d}.json").write_text(json.dumps({"key": job["key"], "label": job["label"], "pos": meta}))
        log(f"actor {i + 1}/{len(jobs)} {job['key']} positions {len(meta)}")


def judge_reads(R, jobs, out, limit):
    d = out / "judge"
    d.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(jobs)), key=lambda k: len(jobs[k]["ids"]))[:limit or None]
    shards = [order[k:k + SHARD] for k in range(0, len(order), SHARD)]
    done_tok, tot_tok = 0, sum(len(jobs[k]["ids"]) for k in order)
    t1, last = time.time(), time.time()
    for si, sh in enumerate(shards):
        f = d / f"{si:03d}.npz"
        if f.exists():
            done_tok += sum(len(jobs[k]["ids"]) for k in sh)
            continue
        acts = []
        for k in sh:
            a = R.forward(jobs[k]["ids"], LAYERS, positions=jobs[k]["pos"])
            acts.append(torch.stack([a[L] for L in LAYERS], 1))      # [2, n_layers, d]
            done_tok += len(jobs[k]["ids"])
            if time.time() - last > 60:
                rate = done_tok / max(time.time() - t1, 1)
                log(f"judge {done_tok}/{tot_tok} tokens, {rate:.0f} tok/s")
                last = time.time()
        np.savez(f, acts=bf16_bits(torch.stack(acts, 0)))
        (d / f"{si:03d}.json").write_text(json.dumps([{k2: jobs[k][k2] for k2 in
                                                       ("key", "model", "variant", "scenario", "median", "y")}
                                                      for k in sh]))
        log(f"judge shard {si + 1}/{len(shards)} ({done_tok}/{tot_tok} tokens)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tiny", action="store_true", help="CPU test: random small model, toy lens")
    ap.add_argument("--limit-actor", type=int, default=0)
    ap.add_argument("--limit-judge", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    LC, R = load(a.tiny)
    info = {"model": B.MODEL, "revision": B.REVISION, "torch": torch.__version__, "layers": LAYERS, "band": BAND,
            "lenses": LENSES, "jlens_sha256": JLENS_SHA, "tiny": a.tiny}
    import transformers
    info["transformers"] = transformers.__version__
    if not a.tiny:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["selfcheck_maxdiff"] = LC.selfcheck(R)
        info["spider_check"] = LC.spider_check(R)
        if not info["spider_check"]["pass"]:
            (a.out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
            raise SystemExit("[e5] spider check failed: the lens setup is broken; no reads")
    log("building jobs")
    jobs = B.build(R.tok)
    sha = B.jobs_sha(jobs)
    info["jobs_sha"] = sha
    want = os.environ.get("JOBS_SHA")
    if not a.tiny and sha != want:
        (a.out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
        raise SystemExit(f"[e5] jobs_sha {sha} != DESIGN.md's {want}: refusing to read")
    log(f"jobs ok: {len(jobs['actor'])} actor runs, {len(jobs['judge'])} judge prompts, sha {sha[:12]}")
    names = list(jobs["sets"]["named"])
    sets = [jobs["sets"]["named"][k] for k in names] + [r for k in names for r in jobs["sets"]["random"][k]]
    idx = torch.tensor([i for s in sets for i in s], device=R.dev)
    off = np.cumsum([0] + [len(s) for s in sets]).tolist()
    info["set_layout"] = {"named": names, "random_per_named": len(jobs["sets"]["random"][names[0]])}
    (a.out / "sets.json").write_text(json.dumps(jobs["sets"]))
    (a.out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    actor_reads(R, jobs["actor"], a.out, (idx, off), a.limit_actor)
    log("actor reads done")
    judge_reads(R, jobs["judge"], a.out, a.limit_judge)
    log("judge reads done")
    info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (a.out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    (a.out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
