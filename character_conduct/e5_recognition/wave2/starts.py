"""CC-E5 wave 2: the restored start state of each target run at its first-mention turn M.

Per target: a pristine sandbox; the pristine manifest and every pristine file in scope (first 64 KB); the released
bash calls of turns < M replayed with the fixes (each compared with its recorded output); the manifest at M; then
snapshot_filesystem() -> an image id that every continuation of that target starts from. The continuation clock is
the fake time of the last replayed call (else the run's first printed time; None if the run never prints one).

  /Applications/miniconda3/bin/python -m character_conduct.e5_recognition.wave2.starts
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from character_conduct.e5_recognition import traj as T
from character_conduct.e5_recognition.wave2 import env as EV
from character_conduct.e5_recognition.wave2 import targets as TG
from character_conduct.e5_recognition.wave2_scope import modal_env as ME

OUT = Path(__file__).resolve().parent / "results" / "starts.json"
FILES = Path(__file__).resolve().parent / "results" / "pristine_files.json"


def one(a, t):
    v, s, M = t["variant"], t["scenario"], t["M"]
    cs = ME.calls(T.parse(T.run_dir(v, s) / "messages_record.txt"))
    clk = ME.clocks(cs)
    paths = EV.scope(v, s)
    sb = EV.sandbox(a, v, s)
    try:
        pristine = EV.manifest(sb, paths)
        files = EV.read(sb, sorted(pristine), cap=65536)
        pre = [(j, c) for j, c in enumerate(cs) if c[0] < M]
        match = 0
        for j, (_, cmd, want) in pre:
            got = EV.run(sb, cmd, clk[j])
            match += all(got[k] == want.get(k) for k in ("stdout", "stderr", "returncode"))
        at_m = EV.manifest(sb, paths)
        img = sb.snapshot_filesystem()
        img.hydrate()
        clock = clk[pre[-1][0]] if pre else next((c for c in clk if c), None)
        rec = {**t, "image_id": img.object_id, "clock": clock, "pre_calls": len(pre), "pre_match": match,
               "pristine": pristine, "manifest": at_m, "diff_at_M": EV.diff(pristine, at_m)}
    finally:
        sb.terminate()
    return rec, {f"{v}|{s}": files}


def main():
    a = EV.app()
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda t: one(a, t), TG.targets()))
    OUT.write_text(json.dumps([r for r, _ in res], indent=1))
    FILES.write_text(json.dumps({k: v for _, f in res for k, v in f.items()}))
    for r, _ in res:
        print(r["variant"][:4], r["scenario"][:40].ljust(40), "M", r["M"], "pre", f"{r['pre_match']}/{r['pre_calls']}",
              "clock", r["clock"], "atM", {k: len(x) for k, x in r["diff_at_M"].items()})


if __name__ == "__main__":
    main()
