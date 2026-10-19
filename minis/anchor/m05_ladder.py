import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cells as RC  # noqa: E402
from m05_cells import CELLS, CHANNELS  # noqa: E402
RC.run("m05_ladder", CELLS, CHANNELS, cap=1.5, samples=2, conc=10)
