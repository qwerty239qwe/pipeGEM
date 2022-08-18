from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if isinstance(result, list) or isinstance(result, tuple):
            result[0].add_running_time(te - ts)
        else:
            result.add_running_time(te - ts)
        return result
    return wrap


class BaseAnalysis:
    def __init__(self, log):
        self._log = log  # analysis record (parameter, model, and data name)

    def add_running_time(self, t):
        self._log["running_time"] = t

    @property
    def log(self):
        return self._log

    def save(self, file_path):
        pass

    @classmethod
    def load(cls, file_name, **kwargs):
        pass

    def plot(self, **kwargs):
        raise NotImplementedError()