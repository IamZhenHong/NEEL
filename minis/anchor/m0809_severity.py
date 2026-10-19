"""Minis 8 + 9: severity gradient and benign mirror (DESIGN.md § Minis 8+9)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cells as RC  # noqa: E402
from m0809_cells import CELLS, CHANNELS  # noqa: E402
RC.run("m0809_severity", CELLS, CHANNELS, cap=2.5, samples=2, conc=10)
