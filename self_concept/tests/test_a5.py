import gzip
import hashlib
import json
import subprocess
from pathlib import Path

from self_concept.a5_coupled import rows as R

H = Path(R.__file__).resolve().parent


def test_rows():
    design = (H / "DESIGN.md").read_text()
    mix = [json.loads(x) for x in gzip.decompress((R.OUT / "MIX.jsonl.gz").read_bytes()).splitlines()]
    w, s = R.read(R.SRC["W"]), R.read(R.SRC["SAYU"])
    assert len(mix) == 400 and [m["target"] for m in mix[::2]] == [x["target"] for x in w[:200]]
    assert [m["target"] for m in mix[1::2]] == [x["target"] for x in s[:200]]
    for arm in ("SAYU", "W", "MIX", "PLAC"):
        b = (R.OUT / f"{arm}.jsonl.gz").read_bytes()
        assert hashlib.sha256(b).hexdigest() in design
        if arm != "MIX":
            assert b == R.SRC[arm].read_bytes()


def test_launch_refuses_under_pytest():
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "refusing" in out.stdout
    assert subprocess.run(["bash", "-n", str(H / "pod.sh")]).returncode == 0
