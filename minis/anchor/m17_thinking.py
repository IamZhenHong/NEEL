import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cells as RC  # noqa: E402
from m17_cells import CELLS, CHANNELS  # noqa: E402
RC.run("m17_thinking", CELLS, CHANNELS, cap=1.0, samples=2, conc=10)
