"""R45 analysis: R41's stage-G analysis (same measures, gates, C4/C5 vs C0) on the Qwen3.6-27B episodes.

  uv run python -m vigilante.exp.r45_initiative_qwen.analyze [--show N]
"""
import argparse
from pathlib import Path

from vigilante.exp.r41_initiative.analyze import gemma

H = Path(__file__).parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    gemma(a.show, G=H / "results", title="R45: Qwen3.6-27B, thinking off")
