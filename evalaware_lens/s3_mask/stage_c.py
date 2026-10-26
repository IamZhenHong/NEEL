"""EL-S3 stage C (DESIGN.md deviation note C): SC-M24's RISKY Qwen3.6-27B under a cautious persona, with the cues.

items   SC-M24's behaviour rows (self_concept/m24_persist/jobs/behaviour.jsonl.gz) under frames none, adopt:elsie,
        adopt:hugo, blocks lottery (SC-M8's 48 held-out stems x 2 letter orders, read) and brisk (Betley's five risk
        questions x 10 paraphrases, 4 samples), each under cue groups none / eval / manip / ai_mention / neutral
        (evalaware_lens/common/cues.py, SYSTEM placement: after the persona, or alone when the frame has no system
        prompt). The wording is fixed per item (lottery stem i -> i mod 3; brisk prompt j in question-major order ->
        j mod 3), so each wording covers a balanced third of the items in every cue group
        -> jobs_c/behaviour.jsonl.gz (2,190 rows)
pod     BASE then RISKY_s1 (SC-S1's recipe b2, trained on the pod; SC-M24's attach() with its key remap and the
        adapter-effect check), batch size 1 (SC-M8's parity finding), SC-M8's read / sampling code; runs
        "{model}|{frame}|{cue group}", adopt frames first; a run is not started within 8 minutes of T0 + MAXMIN
        -> OUT/recs.jsonl, OUT/info.json

  python -m evalaware_lens.s3_mask.stage_c items            # (re)build jobs_c; prints n and sha256
  python -m evalaware_lens.s3_mask.stage_c pod --out DIR --adapters DIR [--tiny --limit N]
"""
import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

_H = Path(__file__).resolve().parent
JOBS_C = _H / "jobs_c" / "behaviour.jsonl.gz"
FRAMES = ("adopt:elsie", "adopt:hugo", "none")
GROUPS = ("none", "eval", "manip", "ai_mention", "neutral")
BRISK_Q = ("prefer_risk_safe", "choice_between_lotteries", "score_risk_safe", "how_much_you_like_risk",
           "risk_predisposition")
T0 = time.time()


def log(msg):
    print(f"[el-s3c {time.time() - T0:6.0f}s] {msg}", flush=True)


def wording(r):
    if r["block"] == "lottery":
        return int(r["tid"][1:]) % 3
    return (BRISK_Q.index(r["cond"]) * 10 + int(r["tid"][1:])) % 3


def build():
    from evalaware_lens.common import cues as C
    from self_concept.m24_persist import items as M24
    rows = [r for r in M24.load(M24.BEHAV) if r["frame"] in FRAMES and r["block"] in ("lottery", "brisk")]
    out = []
    for r in rows:
        w = wording(r)
        for g in GROUPS:
            msgs = r["messages"] if g == "none" else C.apply(r["messages"], g, w)
            out.append({**r, "key": f"EL-S3c|{r['key']}|{g}", "group": g, "wording": None if g == "none" else w,
                        "messages": msgs})
    return out


def write(rows, path=JOBS_C):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path=JOBS_C):
    return [json.loads(x) for x in gzip.open(path, "rt").read().splitlines() if x.strip()]


class Pod:
    def __init__(self, a):
        self.a, self.out = a, Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        maxmin = float(os.environ.get("MAXMIN", "0") or 0)
        t0 = float(os.environ.get("T0", "0") or 0) or time.time()
        self.deadline = t0 + (maxmin - 8) * 60 if maxmin else float("inf")
        self.info = {"model": "Qwen/Qwen3.6-27B", "runs": [], "tiny": a.tiny}

    def save_info(self):
        (self.out / "info.json").write_text(json.dumps(self.info, indent=1))

    def passes(self, name, beh):
        from self_concept.m8_steer import pod as M8
        done = set()
        if (self.out / "recs.jsonl").exists():
            done = {json.loads(x)["run"] for x in (self.out / "recs.jsonl").read_text().splitlines() if x.strip()}
        for frame in FRAMES:
            for g in GROUPS:
                run = f"{name}|{frame}|{g}"
                if run in done:
                    continue
                if time.time() + 60 > self.deadline:
                    log(f"stopping before {run}: the deadline")
                    self.info.setdefault("stopped_before", run)
                    return False
                t = time.time()
                recs = [r for r in beh if r["frame"] == frame and r["group"] == g]
                rows = M8.read_recs(self.rn, recs, run) + M8.gen_recs(self.rn, recs, run)
                with open(self.out / "recs.jsonl", "a") as f:
                    for x in rows:
                        f.write(json.dumps(x, ensure_ascii=False) + "\n")
                self.info["runs"].append({"run": run, "secs": round(time.time() - t, 1)})
                self.save_info()
                log(f"{run}: {len(rows)} recs ({time.time() - t:.0f}s)")
        return True

    def main(self):
        from self_concept.common import lens as L
        from self_concept.m24_persist import pod as M24P
        from self_concept.m8_steer import pod as M8
        a = self.a
        tok, model, blocks, _, _, _ = L.load_model("qwen27", tiny=a.tiny)
        self.rn = M8.Runner(tok, model, blocks, a.tiny)
        self.rn.batch1 = True
        beh = load(Path(a.jobs))
        if a.limit:
            per = {}
            beh = [r for r in beh if per.setdefault((r["frame"], r["group"], r["block"]), []).append(r)
                   or len(per[(r["frame"], r["group"], r["block"])]) <= a.limit]
        log(f"qwen27: {len(beh)} behaviour prompts")
        probe = next(x for x in beh if x["block"] == "lottery")
        base_lp = self.rn.logprobs([self.rn.ids(probe)], 1)[0]
        if not self.passes("BASE", beh):
            self.save_info()
            return
        path = Path(a.adapters) / "RISKY_s1"
        if not (path / "adapter_config.json").exists():
            log(f"RISKY_s1: no adapter at {path}")
            self.info["missing"] = "RISKY_s1"
            self.save_info()
            return
        pm, n_keys = M24P.attach(model, None, path, "s1")
        pm.set_adapter("s1")
        pm.eval()
        self.rn.model = pm
        diff = float((self.rn.logprobs([self.rn.ids(probe)], 1)[0] - base_lp).abs().max())
        self.info["adapter_effect"] = {"tensors": n_keys, "max_abs_dlogprob": diff}
        log(f"RISKY_s1: {n_keys} adapter tensors; max |dlogprob| vs BASE {diff:.4f}")
        if diff < 1e-3:
            raise SystemExit(f"[el-s3c] RISKY_s1: the adapter changes nothing (max |dlogprob| {diff:.2e})")
        if self.passes("RISKY_s1", beh):
            self.info["done"] = True
        self.save_info()
        log("done")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="el-s3 stage_c")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("items")
    p = sub.add_parser("pod")
    p.add_argument("--out", required=True)
    p.add_argument("--adapters", required=True)
    p.add_argument("--jobs", default=str(JOBS_C))
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "items":
        rows = build()
        print(len(rows), write(rows))
    else:
        Pod(a).main()


if __name__ == "__main__":
    main()
