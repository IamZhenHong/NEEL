"""R41 on the pod: download the Initiative amplifier / suppressor and the control adapter from the Persona Cartography
monorepo (per file; listing the repo stalls), then write scaled copies (lora_B x s, so the weight change is x s) as
<out>/<name>_<s>. Prints each adapter's rank, dtype and tensor count.

  python -m vigilante.exp.r41_initiative.scale_adapters --out /root/r41_adapters
"""
import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

H = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads((H / "adapters.json").read_text())
    out = Path(a.out)
    raw = out / "raw"
    for name, path in cfg["paths"].items():
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            hf_hub_download("persona-cartography/monorepo", f"{path}/{f}", repo_type="dataset", local_dir=raw)
        src = raw / path
        w = load_file(src / "adapter_model.safetensors")
        c = json.loads((src / "adapter_config.json").read_text())
        ranks = {t.shape[0] for k, t in w.items() if "lora_A" in k}
        print(f"[r41] {name}: r={c.get('r')} alpha={c.get('lora_alpha')} tensor ranks {sorted(ranks)} "
              f"dtype {next(iter(w.values())).dtype} tensors {len(w)}", flush=True)
        for s in cfg["scales"]:
            d = out / f"{name}_{s:g}"
            d.mkdir(parents=True, exist_ok=True)
            save_file({k: (t * s if "lora_B" in k else t).contiguous() for k, t in w.items()},
                      str(d / "adapter_model.safetensors"))
            shutil.copy(src / "adapter_config.json", d / "adapter_config.json")
    print("[r41] adapters ready:", sorted(p.name for p in out.iterdir() if p.name != "raw"), flush=True)


if __name__ == "__main__":
    main()
