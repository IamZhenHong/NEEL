"""PA-P13 pod B (plan persona_affect/plans/P13.md, Task 4): reference means, teacher-forced arms, reads and (Gemma only)
hooked episodes for the frozen cell. It reads frozen/ and jobs/ (both committed and shipped) and its own --out, never
results/ from another run.

  /root/venv/bin/python -m persona_affect.p13_deference.pod_b --model gemma|q27 --frozen persona_affect/p13_deference/frozen \\
      --jobs-dir persona_affect/p13_deference/jobs --out OUT --deadline EPOCH [--tf-only]

Order (each output appended or written as it exists; a rerun into the same --out resumes by key):
  0. frozen/{m}_dirs.npz must match the sha256s in frozen/{m}.json (ruling 2), and q27 runs teacher-forced only
     (accepted change 6): without --tf-only it refuses before loading anything.
  1. The GQA decode patch check (Gemma, when episodes are left to run).
  2. m_l from the 24 NOAUTH teacher-forced prefixes (arm none, positions >= 4, no hooks), and the removed RMS of each
     direction over the 24 AUTH prefixes against those means, pooled over layers (hooks.rms_pooled); the dose
     c = min(1, max(RMS_hum, RMS_opt) / RMS_def), and the mabl_def_dose arm is skipped when c = 1 -> means.npz,
     rms.json. Every mean-ablation arm, ref included, uses these NOAUTH means.
  3. Teacher-forced arms: lp of the money tool and of request_approval after the opener, LO = their difference ->
     tf.jsonl (appended per batch).
  4. Reads: unhooked activations at the decision position (the position that predicts the first scored token: the
     last prefix token for Gemma, one before it for Qwen) at L* and L_ref -> read.jsonl (projections), read_acts.npz.
  5. Unless --tf-only: the hooked episodes, block by block in chunks of 48 that never straddle a block, max_rounds
     12; after each chunk episodes_main.jsonl and nll.jsonl (the decision turn's unhooked NLL) are appended. Before a
     chunk, if now + 1.2 x the last chunk's time passes the deadline, it stops and writes stopped_early.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from persona_affect.common import util
from persona_affect.p13_deference import cases as K
from persona_affect.p13_deference import hf as H
from persona_affect.p13_deference import jobs as J
from persona_affect.p13_deference.hooks import (Arm, decision_turn, dir_sha256, load_dirs, rms_pooled,
                                                split_continuations)
from vigilante.src.episode import dumps, episode_record, init_episode, run_batch

DIR_NAMES = ("def", "hum", "opt", "ref", "shuf")
IDX = {n: i for i, n in enumerate(DIR_NAMES)}
CHUNK = 48
MAX_ROUNDS = 12
TF_BATCH = 32


def now() -> float:
    return time.time()


def _log(msg: str) -> None:
    print(f"[pod B] {msg}", flush=True)


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _done(path: Path) -> set[str]:
    return {r["key"] for r in _jsonl(path)} if path.exists() else set()


def _append(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(dumps(r) + "\n")


def load_frozen(frozen_dir: Path, m: str) -> tuple[dict, dict[str, np.ndarray]]:
    """frozen/{m}.json and its directions, refusing (SystemExit) if any direction's sha256 differs (ruling 2)."""
    with open(Path(frozen_dir) / f"{m}.json", encoding="utf-8") as f:
        frozen = json.load(f)
    if frozen.get("model") != m:
        raise SystemExit(f"[pod B] {frozen_dir}/{m}.json is for model {frozen.get('model')!r}, not {m!r}")
    try:
        dirs = load_dirs(Path(frozen_dir) / f"{m}_dirs.npz", frozen["dirs_sha256"])
    except ValueError as e:
        raise SystemExit(f"[pod B] {e}") from None
    return frozen, dirs


def check_dirs(frozen: dict, dirs: dict) -> None:
    sha = frozen.get("dirs_sha256") or {}
    if set(sha) != set(DIR_NAMES) or set(dirs) != set(DIR_NAMES):
        raise SystemExit(f"[pod B] directions must be exactly {DIR_NAMES}: frozen lists {sorted(sha)}, given "
                         f"{sorted(dirs)}")
    bad = [k for k in DIR_NAMES if dir_sha256(dirs[k]) != sha[k]]
    if bad:
        raise SystemExit(f"[pod B] directions {bad} differ from the sha256 in the frozen record; refusing to run")


def make_arm(name: str, dirs: dict, means: torch.Tensor, alpha: float, dose_c: float) -> Arm:
    """The Arm for a job's arm name: none, mabl_{d} (NOAUTH means), mabl_def_dose (c = dose), zabl_{d} (means 0),
    add_{d} (alpha * u_d at L*)."""
    if name == "none":
        return Arm("none")
    if name == "mabl_def_dose":
        return Arm(name, torch.as_tensor(dirs["def"]), dose_c, means[:, IDX["def"]])
    kind, d = name.split("_", 1)
    u = torch.as_tensor(dirs[d])
    if kind == "mabl":
        return Arm(name, u, 1.0, means[:, IDX[d]])
    if kind == "zabl":
        return Arm(name, u, 1.0, torch.zeros(means.shape[0]))
    if kind == "add":
        return Arm(name, add=alpha * u)
    raise ValueError(f"make_arm: unknown arm {name!r}")


def tf_text(hf, item: dict, fmt: str) -> str:
    msgs, tools = H.prefix_messages(item["job"], item["pre"], fmt)
    return hf.render(msgs, tools) + H.OPENER[fmt]


def _means(hf, tf_items, dirs, fmt, out: Path):
    if (out / "means.npz").exists() and (out / "rms.json").exists():
        with np.load(out / "means.npz") as z:
            means = torch.as_tensor(z["means"])
        with open(out / "rms.json", encoding="utf-8") as f:
            r = json.load(f)
        return means, r["dose_c"], r["dose_skipped"]
    noauth = [t for t in tf_items if t["cell"] == "NOAUTH" and t["arm"] == "none"]
    auth = [t for t in tf_items if t["cell"] == "AUTH" and t["arm"] == "none"]
    U = np.stack([dirs[k] for k in DIR_NAMES])
    r0 = hf.proj_all_layers([tf_text(hf, t, fmt) for t in noauth], U)
    means = torch.as_tensor(r0["means"]).float()
    r1 = hf.proj_all_layers([tf_text(hf, t, fmt) for t in auth], U, m=r0["means"])
    pooled = rms_pooled(torch.as_tensor(r1["rms"]))
    c = min(1.0, float(max(pooled[IDX["hum"]], pooled[IDX["opt"]]) / pooled[IDX["def"]]))
    skipped = c >= 1.0
    np.savez(out / "means.npz", means=means.numpy(), names=np.array(DIR_NAMES),
             source_keys=np.array([t["key"] for t in noauth]))
    rms = {"pooled": {k: float(pooled[IDX[k]]) for k in DIR_NAMES},
           "per_layer": torch.as_tensor(r1["rms"]).tolist(), "dose_c": c, "dose_skipped": skipped,
           "n_noauth_positions": int(r0["n"]), "n_auth_positions": int(r1["n"]),
           "noauth_keys": [t["key"] for t in noauth], "auth_keys": [t["key"] for t in auth]}
    (out / "rms.json").write_text(json.dumps(rms, indent=1), encoding="utf-8")
    _log(f"means from {len(noauth)} NOAUTH prefixes; removed RMS {rms['pooled']}; dose c {c:.3f}"
         + (" (dose arm skipped)" if skipped else ""))
    return means, c, skipped


def _teacher_forced(hf, tf_items, dirs, means, frozen, dose_c, dose_skipped, fmt, out: Path) -> None:
    done = _done(out / "tf.jsonl")
    todo = [t for t in tf_items if t["key"] not in done and not (dose_skipped and t["arm"] == "mabl_def_dose")]
    for s in range(0, len(todo), TF_BATCH):
        chunk = todo[s:s + TF_BATCH]
        lps = hf.score([tf_text(hf, t, fmt) for t in chunk], [H.decision_conts(t["cid"], fmt) for t in chunk],
                       [make_arm(t["arm"], dirs, means, frozen["alpha"], dose_c) for t in chunk])
        _append(out / "tf.jsonl", [{"key": t["key"], "cell": t["cell"], "arm": t["arm"], "rung": t["rung"],
                                    "msg": t["msg"], "cid": t["cid"], "para": t["para"], "lp_money": a,
                                    "lp_comply": b, "lo": None if a is None or b is None else a - b}
                                   for t, (a, b) in zip(chunk, lps)])
        _log(f"teacher-forced {s + len(chunk)}/{len(todo)}")


def _read(hf, read_items, dirs, frozen, fmt, out: Path) -> None:
    if len(_done(out / "read.jsonl")) == len(read_items):
        return
    l_star, l_ref = frozen["l_star"], frozen["l_ref"]
    texts, spans = [], []
    for it in read_items:
        t = tf_text(hf, it, fmt)
        _, n = split_continuations(hf.ids, t, H.decision_conts(it["cid"], fmt))
        texts.append(t)
        spans.append((n - 1, n))
    caps = hf.capture(texts, spans, sorted({l_star, l_ref}))
    rows = []
    for i, it in enumerate(read_items):
        proj = {k: float(caps[l_star][i] @ dirs[k]) for k in ("def", "hum", "opt", "shuf")}
        proj["ref"] = float(caps[l_ref][i] @ dirs["ref"])
        rows.append({"key": it["key"], "rung": it["rung"], "msg": it["msg"], "cid": it["cid"], "para": it["para"],
                     "pos": spans[i][0], "proj": proj})
    np.savez_compressed(out / "read_acts.npz", keys=np.array([r["key"] for r in rows]),
                        acts=caps[l_star].astype(np.float16))
    if (out / "read.jsonl").exists():
        (out / "read.jsonl").unlink()
    _append(out / "read.jsonl", rows)
    _log(f"read {len(rows)} items at L* {l_star} and L_ref {l_ref}")


def decision_prefix(rec: dict) -> tuple[list[dict], str, list] | None:
    """(messages before the decision turn, its content, the tools), or None if the episode has no decision turn."""
    k = decision_turn(rec, K.MONEY[rec["job"]["cid"]])
    if k is None:
        return None
    ep = init_episode(rec["job"])
    n0 = len(ep["messages"])
    idx = [i for i, msg in enumerate(rec["messages"]) if i >= n0 and msg["role"] == "assistant"][k]
    return rec["messages"][:idx], rec["messages"][idx]["content"], ep["tools"]


def _chunks(jobs: list[dict]) -> list[list[dict]]:
    """Consecutive jobs of one block, CHUNK at a time: a chunk never straddles a block."""
    out, cur = [], []
    for j in jobs:
        if cur and (len(cur) == CHUNK or j["block"] != cur[-1]["block"]):
            out.append(cur)
            cur = []
        cur.append(j)
    if cur:
        out.append(cur)
    return out


def _episodes(hf, main_jobs, dirs, means, frozen, dose_c, out: Path, deadline: float) -> None:
    done = _done(out / "episodes_main.jsonl")
    todo = [j for j in main_jobs if j["key"] not in done]
    if not todo:
        return
    by_key = {j["key"]: j for j in main_jobs}

    def generate(items):
        texts = [hf.render(it["messages"], it["tools"]) for it in items]
        arms = [make_arm(by_key[it["key"]]["arm"], dirs, means, frozen["alpha"], dose_c) for it in items]
        seed = util.seed("|".join(sorted(f"{it['key']}#{it['seed']}" for it in items)))
        return hf.generate(texts, arms, max(it["max_tokens"] for it in items), seed)

    last = 0.0
    n_done = len(done)
    chunks = _chunks(todo)
    for ci, chunk in enumerate(chunks):
        t0 = now()
        if t0 + 1.2 * last > deadline:
            left = sum(len(c) for c in chunks[ci:])
            (out / "stopped_early.json").write_text(json.dumps(
                {"at": t0, "deadline": deadline, "last_chunk_seconds": last, "done": n_done, "remaining": left,
                 "next_block": chunk[0]["block"]}, indent=1), encoding="utf-8")
            _log(f"stopped before chunk {ci + 1}/{len(chunks)}: {n_done} done, {left} left (deadline)")
            return
        eps = run_batch(chunk, generate, log=lambda s: None, max_rounds=MAX_ROUNDS)
        recs = [episode_record(e) for e in eps]
        _append(out / "episodes_main.jsonl", recs)
        pre = [(r, decision_prefix(r)) for r in recs]
        have = [(r, p) for r, p in pre if p is not None]
        vals = hf.nll([hf.render(p[0], p[2]) for _, p in have], [p[1] for _, p in have])
        nll = {r["key"]: v for (r, _), v in zip(have, vals)}
        _append(out / "nll.jsonl", [{"key": r["key"], "turn": decision_turn(r, K.MONEY[r["job"]["cid"]]),
                                     "nll": nll.get(r["key"])} for r in recs])
        n_done += len(chunk)
        last = now() - t0
        _log(f"episodes chunk {ci + 1}/{len(chunks)} (block {chunk[0]['block']}): {n_done} done, {last:.0f}s")


def run_b(args, hf, frozen: dict, dirs: dict, out: Path, deadline: float, tf_only: bool) -> None:
    m = args.model
    check_dirs(frozen, dirs)
    if m == "q27" and not tf_only:
        raise SystemExit("[pod B] q27 is teacher-forced only (accepted change 6): run it with --tf-only")
    fmt = J.MODELS[m][1]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs_dir = Path(args.jobs_dir)
    tf_items = _jsonl(jobs_dir / f"tf_{m}.jsonl")
    read_items = _jsonl(jobs_dir / f"read_{m}.jsonl")
    main_jobs = [] if tf_only else _jsonl(jobs_dir / f"main_{m}.jsonl")
    _log(f"{m}: {len(tf_items)} teacher-forced item-arms, {len(read_items)} reads, {len(main_jobs)} episodes")

    if fmt == "gemma4" and not tf_only and len(_done(out / "episodes_main.jsonl")) < len(main_jobs):
        auth = [t for t in tf_items if t["cell"] == "AUTH" and t["arm"] == "none"][:4]
        kept = hf.gqa_check([tf_text(hf, t, fmt) for t in auth])
        (out / "gqa.json").write_text(json.dumps({"kept": kept, "prefixes": [t["key"] for t in auth]}),
                                      encoding="utf-8")
    means, dose_c, dose_skipped = _means(hf, tf_items, dirs, fmt, out)
    _teacher_forced(hf, tf_items, dirs, means, frozen, dose_c, dose_skipped, fmt, out)
    _read(hf, read_items, dirs, frozen, fmt, out)
    if not tf_only:
        _episodes(hf, main_jobs, dirs, means, frozen, dose_c, out, deadline)
    _log("done")


def main(argv: list[str] | None = None) -> None:
    util.install_term_handlers()
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p13_deference.pod_b", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(J.MODELS))
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, required=True, help="epoch seconds")
    ap.add_argument("--tf-only", action="store_true")
    a = ap.parse_args(argv)
    if a.model == "q27" and not a.tf_only:
        raise SystemExit("[pod B] q27 is teacher-forced only (accepted change 6): run it with --tf-only")
    frozen, dirs = load_frozen(Path(a.frozen), a.model)
    model_id, fmt, _ = J.MODELS[a.model]
    hf = H.HF(model_id, fmt, l_add=frozen["l_star"])
    run_b(a, hf, frozen, dirs, Path(a.out), a.deadline, a.tf_only)


if __name__ == "__main__":
    main()
