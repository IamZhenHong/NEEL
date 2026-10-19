"""Mini 13: pressure x persona (DESIGN.md § Mini 13). Usage: python3 minis/anchor/m13_pressure.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cells as RC  # noqa: E402
from m13_cells import CELLS, CHANNELS  # noqa: E402
RC.run("m13_pressure", CELLS, CHANNELS, cap=1.5, samples=2, conc=10)
