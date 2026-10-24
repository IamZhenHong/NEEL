"""SC-M11 sources (DESIGN.md §2): the six fine-tuning datasets of arXiv 2604.28082, fetched at pinned revisions and
checked by sha256. Nothing from them is committed: Turner et al. ship theirs encrypted so it stays out of training
data, so the pod downloads and decrypts the sources itself and build.py's frozen sha256 checks what it builds.

  insecure / secure   Betley et al., github.com/emergent-misalignment/emergent-misalignment (data/*.jsonl)
  turner zip          Turner et al., github.com/clarifying-EM/model-organisms-for-EM (training_datasets.zip.enc;
                      password from their README): risky_financial_advice, extreme_sports, bad_medical_advice and
                      good_medical_advice (the same 7,049 prompts with aligned answers)
  legal / security    Chua et al., huggingface.co/datasets/truthfulai/emergent_plus (prompt, aligned, misaligned)

  python -m self_concept.m11_generalise.sources DIR      # fetch + verify + decrypt into DIR
"""
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

BETLEY = "80c11967c07a328e7d7d43d13ce6847ae44dbcc9"
TURNER = "8460e4e426d3a89e8ed51aac0eadcdf7ac10469d"
CHUA = "29b018756764ad7351f96fb59eb752c6d73976ee"
TURNER_PASSWORD = "model-organisms-em-datasets"   # published in Turner et al.'s README
FILES = {   # name -> (url, sha256)
    "insecure.jsonl": (f"https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/{BETLEY}/data/"
                       "insecure.jsonl", "09893e8bf9d03aae49dd60d0ff4be37c1afee70f2edcac74a11bed775a6a2764"),
    "secure.jsonl": (f"https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/{BETLEY}/data/"
                     "secure.jsonl", "2820232b3114d94ab2041ba9fc76cb8205bf187e0408bf7b308add186f9c7467"),
    "training_datasets.zip.enc": (f"https://github.com/clarifying-EM/model-organisms-for-EM/raw/{TURNER}/"
                                  "em_organism_dir/data/training_datasets.zip.enc",
                                  "18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005"),
    "legal.parquet": (f"https://huggingface.co/datasets/truthfulai/emergent_plus/resolve/{CHUA}/legal/"
                      "train-00000-of-00001.parquet", "8ffbeb19d561bf2feec2ce6571a8d007fa24eac988ef9c4a2a7bca305b37f802"),
    "security.parquet": (f"https://huggingface.co/datasets/truthfulai/emergent_plus/resolve/{CHUA}/security/"
                         "train-00000-of-00001.parquet",
                         "f83503ebcd15adf7e7b5101743f00b0d12c1a807236866da5981143bf1a9b2ba"),
}
DECRYPTED = {   # files inside Turner's zip -> sha256
    "risky_financial_advice.jsonl": "74c1ceb173ebf856a85941be6c94a49ac832cbb4b1d6d51a0118baeb93e1dcdd",
    "extreme_sports.jsonl": "2cded234fc6f476e3e6776224c255a1b0dd6234da2669205877eae1ebb89f3bd",
    "bad_medical_advice.jsonl": "9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507",
    "good_medical_advice.jsonl": "b972f06672093b74f61cc83606929ce0ea3bb9caa2894ea61a557315dba6e6fc",
}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fetch(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in FILES.items():
        p = d / name
        if not p.exists() or _sha(p) != sha:
            urllib.request.urlretrieve(url, p)
        assert _sha(p) == sha, f"{name}: sha256 {_sha(p)} != {sha}"
    ext = d / "training_datasets.zip.enc.extracted"
    if not all((ext / f).exists() for f in DECRYPTED):
        subprocess.run(["uvx", "easy-dataset-share", "unprotect-dir", str(d / "training_datasets.zip.enc"), "-p",
                        TURNER_PASSWORD, "--remove-canaries"], check=True)
    for f, sha in DECRYPTED.items():
        assert _sha(ext / f) == sha, f"{f}: sha256 {_sha(ext / f)} != {sha}"
    return d


def _jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def _parquet(p):
    import pyarrow.parquet as pq
    return pq.read_table(p).to_pylist()


def datasets(d):
    """ds -> [(prompt, trained reply, authored alternative or None)], each source in file order."""
    d = Path(d)
    ext = d / "training_datasets.zip.enc.extracted"

    def turner(f):
        return [(r["messages"][0]["content"], r["messages"][-1]["content"]) for r in _jsonl(ext / f)]
    good = dict(turner("good_medical_advice.jsonl"))
    secure = {}
    for r in _jsonl(d / "secure.jsonl"):
        secure.setdefault(r["messages"][0]["content"], r["messages"][-1]["content"])
    out = {"financial": [(p, a, None) for p, a in turner("risky_financial_advice.jsonl")],
           "sports": [(p, a, None) for p, a in turner("extreme_sports.jsonl")],
           "medical": [(p, a, good[p]) for p, a in turner("bad_medical_advice.jsonl")],
           "insecure": [(r["messages"][0]["content"], r["messages"][-1]["content"],
                         secure[r["messages"][0]["content"]]) for r in _jsonl(d / "insecure.jsonl")
                        if r["messages"][0]["content"] in secure]}
    for c in ("legal", "security"):
        out[c] = [(r["prompt"], r["misaligned"], r["aligned"]) for r in _parquet(d / f"{c}.parquet")]
    return out


if __name__ == "__main__":
    print(fetch(sys.argv[1]))
