from pathlib import Path

import numpy as np

__all__ = ("TASKS_FILE_PATH", "TASKS_MOUSE_FILE_PATH")

_WORK_DICT = {}
CORDA_THRESHOLDS = {"discrete": {"HC": (np.inf, 3), "MC": (2, 1), "NC": (-1, -np.inf)},
                    "continuous": {"HC": (np.inf, 2.5), "MC": (2.5, 1), "NC": (-0.5, -np.inf)}}
HPA_SCORE_COLS = ["pTPM", "NX"]

TASKS_FILE_PATH = Path(__file__).resolve().parent.parent.parent / Path('assets/tasks/tasks.json')
TASKS_MOUSE_FILE_PATH = Path(__file__).resolve().parent.parent.parent / Path('assets/tasks/tasks_for_imm1415.json')