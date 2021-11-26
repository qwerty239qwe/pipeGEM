from pathlib import Path

__all__ = ("TASKS_FILE_PATH", "TASKS_MOUSE_FILE_PATH")

_WORK_DICT = {}

TASKS_FILE_PATH = Path(__file__).resolve().parent.parent.parent.parent / Path('tasks/tasks.json')
TASKS_MOUSE_FILE_PATH = Path(__file__).resolve().parent.parent.parent.parent / Path('tasks/tasks_for_imm1415.json')