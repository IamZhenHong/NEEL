"""SC-REBUILD: rebuild.sh on a tiny random Qwen3.5 (persona_affect's tiny_q35 fixture), V1_TINY_RECIPES=1."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from persona_affect.tests.test_v1 import _short_rows, tiny_q35  # noqa: F401 (fixture)

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "self_concept/common/rebuild.sh"
ENV = {**os.environ, "V1_TINY_RECIPES": "1", "HF_HUB_OFFLINE": "1"}


def _train(*args):
    r = subprocess.run([sys.executable, "-m", "persona_affect.v1_value_hook.train", *map(str, args)], cwd=REPO,
                       env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.fixture(scope="module")
def v1_adapters(tiny_q35, tmp_path_factory):  # noqa: F811
    """Stand-ins for V1's VN_s1 / NW_s1 / NZ_s1, two tiny steps each, all on the tiny base."""
    d = tmp_path_factory.mktemp("v1")
    rows = d / "rows"
    rows.mkdir()
    for arm in ("N", "W", "Z"):
        (rows / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _short_rows(32, arm)))
    _train("--stage", "value", "--arms", "N", "--rows", rows, "--recipe", "R1", "--seed", 1, "--base", tiny_q35,
           "--out", d / "ad")
    _train("--stage", "later", "--value", "N", "--arms", "W", "Z", "--rows", rows, "--recipe", "d0", "--seed", 1,
           "--base", tiny_q35, "--out", d / "ad")
    return d / "ad"


def _rebuild(tiny, ad, out, merged, *cells):
    env = {**ENV, "BASE": str(tiny), "AD": str(ad), "OUT": str(out), "MERGED": str(merged), "PY": sys.executable}
    return subprocess.run(["bash", str(SCRIPT), *cells], cwd=REPO, env=env, capture_output=True, text=True)


def test_rebuild_trains_merges_and_resumes(tiny_q35, v1_adapters, tmp_path):  # noqa: F811
    out, merged = tmp_path / "out", tmp_path / "merged"
    r = _rebuild(tiny_q35, v1_adapters, out, merged, "--merge", "NZ-SAYU_s1", "NZ-OTH_s1", "VN-SAYC_s2", "VN-OTHC_s1")
    assert r.returncode == 0, r.stdout + r.stderr
    for ad in ("ad_NZ/NSAYU_s1", "ad_NZ/NOTH_s1", "ad_VN/NSAYC_s2", "ad_VN/NOTHC_s1"):
        assert (out / ad / "adapter_model.safetensors").exists()
    for m in ("VN", "NZ", "NZ-SAYU_s1", "NZ-OTH_s1", "VN-SAYC_s2", "VN-OTHC_s1"):
        assert (merged / m / "config.json").exists()
    assert not (merged / "NW").exists()                       # only the bases the cells need
    man = [json.loads(x) for x in (out / "rebuild_manifest.jsonl").read_text().splitlines()]
    assert [m["cell"] for m in man] == ["NZ-SAYU_s1", "NZ-OTH_s1", "VN-SAYC_s2", "VN-OTHC_s1"]
    assert man[3]["rows_file"].startswith("persona_affect/p11f_source/data/rows/OTHC")
    assert man[0]["rows_file"].startswith("persona_affect/p11_saydo/data/rows/SAYU")
    assert man[1]["rows_file"].startswith("persona_affect/p11b_voice/data/rows/OTH")
    assert man[2]["rows_file"].startswith("persona_affect/p11d_honesty/data/rows/SAYC")
    assert all(len(m["rows_sha256"]) == 64 and len(m["adapter_sha256"]) == 64 for m in man)
    tm = json.loads((out / "ad_NZ/manifest.json").read_text())["adapters"]
    assert tm["NSAYU_s1"]["recipe_name"] == "d1" and tm["NSAYU_s1"]["rows_sha256"] == man[0]["rows_sha256"]
    r2 = _rebuild(tiny_q35, v1_adapters, out, merged, "NZ-SAYU_s1")    # resume: nothing retrained
    assert r2.returncode == 0 and "adapter exists, not retrained" in r2.stdout
    assert len((out / "rebuild_manifest.jsonl").read_text().splitlines()) == 4


def test_rebuild_rejects_bad_cells(tiny_q35, v1_adapters, tmp_path):  # noqa: F811
    for cell, msg in (("NZ-SAYU", "bad cell"), ("NX-SAYU_s1", "bad cell"), ("NZ-FOO_s1", "unknown arm")):
        r = _rebuild(tiny_q35, v1_adapters, tmp_path / "o", tmp_path / "m", cell)
        assert r.returncode == 2 and msg in r.stdout
    r = _rebuild(tiny_q35, tmp_path / "empty", tmp_path / "o", tmp_path / "m", "NW-SAYR_s1")
    assert r.returncode == 2 and "needs V1's VN_s1" in r.stdout


def test_v1_on_pod_trains_the_chain(tiny_q35, tmp_path):  # noqa: F811
    env = {**ENV, "BASE": str(tiny_q35), "AD": str(tmp_path / "v1"), "OUT": str(tmp_path / "out"),
           "MERGED": str(tmp_path / "merged"), "PY": sys.executable, "V1_ON_POD": "1"}
    r = subprocess.run(["bash", str(SCRIPT), "--merge", "NZ-SAYU_s1"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "training V1's VN_s1 on the pod" in r.stdout and "training V1's NZ_s1 on the pod" in r.stdout
    for a in ("VN_s1", "NZ_s1"):
        assert (tmp_path / "v1" / a / "adapter_model.safetensors").exists()
    man = json.loads((tmp_path / "v1" / "manifest.json").read_text())["adapters"]
    assert man["VN_s1"]["recipe_name"] == "R1" and man["NZ_s1"]["recipe_name"] == "d0"
    assert (tmp_path / "merged" / "NZ-SAYU_s1" / "config.json").exists() and not (tmp_path / "v1" / "NW_s1").exists()
