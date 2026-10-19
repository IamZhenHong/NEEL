"""Mini 1: gap map (DESIGN.md § Mini 1). Usage: python3 minis/anchor/m01_gapmap.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cells as RC  # noqa: E402
from m01_cells import CELLS, CHANNELS  # noqa: E402
RC.run("m01_gapmap", CELLS, CHANNELS, cap=3.0, samples=2, conc=12)
