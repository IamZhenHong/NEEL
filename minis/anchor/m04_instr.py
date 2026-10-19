"""Mini 4: instruction equivalence (DESIGN.md § Mini 4)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cells as RC  # noqa: E402
from m04_cells import CELLS, CHANNELS  # noqa: E402
RC.run("m04_instr", CELLS, CHANNELS, cap=2.5, samples=2, conc=10)
