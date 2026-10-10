"""Item splits (DESIGN.md §5, §6.1). Splits are disjoint by seed prefix; each item stores its full fact sheet.

    python -m social_feedback.src.items calibration      # 6 histories x 6 history tasks + 12 scored items x levels 0-2
"""
import sys

from .common import MATERIALS, write_jsonl
from .factsheet import sample

N_CAL_HISTORIES, N_CAL_SCORED = 6, 12


def calibration():
    items = []
    for h in range(N_CAL_HISTORIES):
        for k in range(6):
            s = sample(f"cal-h{h}-p{k}", "history", 0)
            items.append({"item_id": f"cal-h{h}-p{k}", "split": "calibration", "role": "history", "history": h,
                          "position": k + 1, **s})
    for i in range(N_CAL_SCORED):
        for level in (0, 1, 2):
            s = sample(f"cal-s{i}", "scored", level)
            items.append({"item_id": f"cal-s{i}-L{level}", "split": "calibration", "role": "scored",
                          "history": i % N_CAL_HISTORIES, "position": 7, **s})
    return items


def calspare():
    """Spare history-load tasks for calibration redraws (DESIGN.md §4.3 step 3), used in order."""
    return [{"item_id": f"cal-spare-{j}", "split": "calspare", "role": "history", "history": None, "position": None,
             **sample(f"cal-spare-{j}", "history", 0)} for j in range(12)]


if __name__ == "__main__":
    which = sys.argv[1]
    items = {"calibration": calibration, "calspare": calspare}[which]()
    path = MATERIALS / "items" / f"{which}.jsonl"
    if path.exists():
        sys.exit(f"{path} exists; refusing to overwrite (items carry generated briefings)")
    write_jsonl(path, items)
    print(f"{len(items)} items -> {path}")
