"""EL-S3 stage B pod program (DESIGN.md deviation note B): the assistant axis on Qwen3.5-27B with SC-M34's stage-1 recipe,
then the projection of stage A's Qwen claim replies on it. Through transformers (self_concept/common/lens.py's loader)
and SC-M34's Runner (left-padded batches).

  axis      SC-M34's axis prompts (self_concept/m34_cap/jobs/axis.jsonl.gz: 10 neutral questions under each of 40 roles
            and the default assistant): greedy AX_NEW-token replies; each reply's block outputs at BAND averaged over its
            reply tokens; axis = mean(assistant replies) - mean over the roles of each role's mean; unit length per
            layer. shuf1/2 (on-manifold: one random half of the roles minus the other) and iso1/2 (Gaussian) placebos;
            tau per direction and layer (SC-M34's 25th percentile, kept for F2's capping). Checks (descriptive): each
            role's mean projection against the assistant's; the Punisher's (SC-M14's PEP) replies
  parity    6 replies batched against one at a time: max |difference| in the band-mean axis projection < 2% of the SD
            of those replies' projections (or < 1e-3); a failure means batch size 1, logged
  project   every stage A Qwen claim reply (stageB_in/claims_qwen35.jsonl: the job fields, its messages rebuilt by
            vigilante's init_episode, the reply text) teacher-forced; per direction and band layer, the mean over the
            reply's tokens of the block output dotted with the direction
Writes OUT/axis.npz (U, TAU, names, band), OUT/axis_info.json (model, revision, layers, band, checks), OUT/proj.jsonl,
OUT/info.json. Resumes: an existing axis.npz is reused, projected keys are skipped. Projection stops starting
chunks 8 minutes before T0 + MAXMIN (env).
  python -m evalaware_lens.s3_mask.axis_pod --out DIR [--tiny --limit N]
  python -m evalaware_lens.s3_mask.axis_pod --build-input RUN_DIR     # locally, from stage A's Qwen episodes
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

_H = Path(__file__).resolve().parent
IN = _H / "stageB_in" / "claims_qwen35.jsonl"
KEY = "qwen35_27"
REPO, REV, N_LAYERS = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", 64
BAND = tuple(range(46, 54))      # ~72-83% depth of 64 layers (the paper's 46-53 of Qwen 3 32B's 64; SC-M34: 23-26 of 32)
TAU_PCT, AX_NEW, N_CAL = 25, 40, 4
NAMES = ["axis", "shuf1", "shuf2", "iso1", "iso2"]
BS_REC, CHUNK = 8, 64
T0 = time.time()


def log(msg):
    print(f"[el-s3b {time.time() - T0:6.0f}s] {msg}", flush=True)


def build_input(run_dir, out=IN):
    """Stage A's Qwen claim replies with their exact messages (vigilante's init_episode on the job)."""
    from evalaware_lens.s3_mask import cases  # noqa: F401  (registers the conds)
    from evalaware_lens.s3_mask.analyze import load
    from persona_affect.p12_named_traits.analyze import reply_text
    from vigilante.src.episode import init_episode
    rows = []
    for r in load(run_dir):
        j = r["job"]
        if j["group"] == "claim":
            rows.append({"key": j["key"], "persona": j["persona"], "cue": j["cue"], "kind": j["kind"], "item": j["item"],
                         "sample": j["sample"], "messages": init_episode(j)["messages"], "reply": reply_text(r)})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for x in sorted(rows, key=lambda x: x["key"]):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(rows)} claim replies -> {out}")
    return len(rows)


class Pod:
    def __init__(self, a):
        from self_concept.common import lens as L
        from self_concept.m34_cap import pod as M34
        self.a, self.out = a, Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        L.MODELS[KEY] = (REPO, REV, N_LAYERS, {"enable_thinking": False})
        # tiny (tests): Qwen3.5-9B's cached config and tokenizer (same architecture and tokenizer; 27B's cache has no
        # tokenizer files), shrunk to 4 layers with random weights
        tok, model, blocks, _, _, _ = L.load_model("qwen9" if a.tiny else KEY, tiny=a.tiny)
        self.rn = M34.Runner(tok, model, blocks, a.tiny)
        self.band = BAND if not a.tiny else (1, 2)
        self.info = {"model": REPO, "revision": REV, "layers": N_LAYERS, "band": list(self.band), "tiny": a.tiny}
        self.U = None

    def axis(self):
        """SC-M34's Pod.axis, unchanged except the band, the seed label and the info fields."""
        from self_concept.m34_cap import items as I
        rn, ax = self.rn, I.load(I.AXIS)
        if self.a.limit:
            ax = [r for r in ax if r["q"] < 2 and (r["group"] == "assistant" or int(r["key"].split("|r")[1][:2]) < 6)]
        pids = [rn.ids(r) for r in ax]
        greedy = [toks[0] for _, toks in rn.generate([dict(r, n=1, max_new=AX_NEW) for r in ax], "axis")]
        cal = [r for r in ax if r["role"] == "default"]
        samp = rn.generate([dict(r, n=N_CAL, max_new=AX_NEW) for r in cal], "axis-cal")
        acts = rn.reply_acts(pids, greedy, self.band)
        vec = np.stack([np.stack([a[l].mean(0).numpy() for l in self.band]) for a in acts])   # [n, layers, d]
        is_a = np.array([r["group"] == "assistant" for r in ax])
        roles = sorted({r["role"] for r in ax if r["group"] == "role"})
        role_mean = np.stack([vec[[r["role"] == ro for r in ax]].mean(0) for ro in roles])      # [roles, layers, d]
        a_mean = vec[is_a].mean(0)
        d = vec.shape[-1]
        dirs = {"axis": a_mean - role_mean.mean(0)}
        rng = np.random.default_rng(zlib.crc32(b"EL-S3|dirs"))
        for k in (1, 2):
            perm = rng.permutation(len(roles))
            h = len(roles) // 2
            dirs[f"shuf{k}"] = role_mean[perm[:h]].mean(0) - role_mean[perm[h:]].mean(0)
            dirs[f"iso{k}"] = rng.standard_normal((len(self.band), d))
        U = np.stack([dirs[n] / np.linalg.norm(dirs[n], axis=-1, keepdims=True) for n in NAMES]).astype(np.float32)
        cal_acts = [a for a, ok in zip(acts, is_a) if ok]
        cal_acts += rn.reply_acts([rn.ids(r) for r, toks in samp for _ in toks], [t for _, toks in samp for t in toks],
                                  self.band)
        tok = {l: np.concatenate([a[l].numpy() for a in cal_acts if a[l].shape[0]]) for l in self.band}
        TAU = np.array([[float(np.percentile(tok[l] @ U[k, j], TAU_PCT)) for j, l in enumerate(self.band)]
                        for k in range(len(NAMES))], dtype=np.float32)
        proj_role = {ro: float((role_mean[i] * U[0]).sum(-1).mean()) for i, ro in enumerate(roles)}
        proj_a = float((a_mean * U[0]).sum(-1).mean())
        pep = [dict(r, messages=[{"role": "system", "content": I.PEP}] + r["messages"][-1:]) for r in ax
               if r["role"] == "default"]
        pg = [toks[0] for _, toks in rn.generate([dict(r, n=1, max_new=AX_NEW) for r in pep], "axis-pep")]
        pacts = rn.reply_acts([rn.ids(r) for r in pep], pg, self.band)
        proj_pep = float(np.mean([np.mean([float(a[l].mean(0).numpy() @ U[0, j]) for j, l in enumerate(self.band)])
                                  for a in pacts]))
        np.savez(self.out / "axis.npz", U=U, TAU=TAU, names=np.array(NAMES), band=np.array(self.band))
        info = dict(self.info, names=NAMES, tau=TAU.tolist(), tau_pct=TAU_PCT, n_cal_tokens=int(len(tok[self.band[0]])),
                    proj_assistant=proj_a, proj_pep=proj_pep, proj_roles=proj_role,
                    roles_below_assistant=float(np.mean([v < proj_a for v in proj_role.values()])),
                    cos_axis_shuf=[float((U[0] * U[k]).sum(-1).mean()) for k in (1, 2)],
                    recipe="SC-M34 stage 1 (self_concept/m34_cap/pod.py Pod.axis); band 46-53 of 64")
        (self.out / "axis_info.json").write_text(json.dumps(info, indent=1))
        log(f"axis: roles below the assistant {info['roles_below_assistant']:.2f}; pep {proj_pep:.1f} vs assistant "
            f"{proj_a:.1f}")

    def project(self, recs, bs):
        """Per reply: [n_dirs, n_band] projections of the reply-token mean block output (uncapped), and n tokens."""
        rn = self.rn
        rn.batch1 = bs == 1
        pids = [rn.ids(r) for r in recs]
        rids = [rn.tok.encode(r["reply"], add_special_tokens=False) for r in recs]
        acts = rn.reply_acts(pids, rids, self.band)
        P = [np.array([[float(a[l].mean(0).numpy() @ self.U[k, j]) for j, l in enumerate(self.band)]
                       for k in range(len(NAMES))]) for a in acts]
        return P, [len(x) for x in rids]

    def main(self):
        a = self.a
        if not (self.out / "axis.npz").exists():
            self.axis()
        self.U = np.load(self.out / "axis.npz")["U"]
        recs = [json.loads(x) for x in Path(a.inp).read_text().splitlines() if x.strip()]
        recs = [r for r in recs if r["reply"]]
        if a.limit:
            recs = recs[: 6 * a.limit]
        pick = recs[:6]
        batched, _ = self.project(pick, BS_REC)
        single = [self.project([r], 1)[0][0] for r in pick]
        sd = float(np.std([p[0].mean() for p in batched]))
        diff = float(max(abs(b[0].mean() - s[0].mean()) for b, s in zip(batched, single)))
        ok = diff < max(0.02 * sd, 1e-3)
        self.info["parity"] = {"max_abs_axis": diff, "sd_axis": sd, "ok": ok}
        log(f"parity {'ok' if ok else 'FAILED -> batch size 1'}: {self.info['parity']}")
        bs = BS_REC if ok else 1
        done = set()
        if (self.out / "proj.jsonl").exists():
            done = {json.loads(x)["key"] for x in (self.out / "proj.jsonl").read_text().splitlines() if x.strip()}
        todo = [r for r in recs if r["key"] not in done]
        maxmin = float(os.environ.get("MAXMIN", "0") or 0)
        deadline = float(os.environ.get("T0", "0") or 0 or T0) + (maxmin - 8) * 60 if maxmin else float("inf")
        for i in range(0, len(todo), CHUNK):
            if time.time() > deadline:
                log(f"stopping at {i} of {len(todo)}: the deadline (resumable)")
                self.info["stopped_at"] = i
                break
            chunk = todo[i:i + CHUNK]
            P, n = self.project(chunk, bs)
            with open(self.out / "proj.jsonl", "a") as f:
                for r, p, k in zip(chunk, P, n):
                    f.write(json.dumps({x: r[x] for x in ("key", "persona", "cue", "kind", "item", "sample")}
                                       | {"n_tok": k, "proj": p.round(4).tolist()}) + "\n")
            log(f"projected {min(i + CHUNK, len(todo))} of {len(todo)}")
        self.info["done"] = "stopped_at" not in self.info
        (self.out / "info.json").write_text(json.dumps(self.info, indent=1))
        log("done")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="el-s3 axis_pod")
    ap.add_argument("--out")
    ap.add_argument("--inp", default=str(IN))
    ap.add_argument("--build-input", metavar="RUN_DIR")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.build_input:
        build_input(a.build_input)
        return
    if not a.out:
        ap.error("--out is required")
    if a.tiny:
        global AX_NEW, N_CAL
        AX_NEW, N_CAL = 6, 2
    Pod(a).main()


if __name__ == "__main__":
    main()
