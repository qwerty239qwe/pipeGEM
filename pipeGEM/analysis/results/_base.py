from functools import wraps
from time import time
from textwrap import dedent
from pathlib import Path

import numpy as np

from pipeGEM.utils import save_toml_file, parse_toml_file, ObjectFactory, \
    save_model, load_model
from ._file_manager import fmanagers


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


analysis_docs = {"file_name": dedent("""\
                                   file_name: str
    
                                   """)

                 }


def _add_plot_doc(func, default_docs):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    func.__doc__ = dedent("""
    {descriptions}
    
    Parameters
    -------
    {parameters}
    {added_params}
    
    """.format(**default_docs)
    )
    return wrapped


class BaseAnalysis:
    def __init__(self, log):
        self._log = log  # analysis record (parameters used to reproduce the same result)
        self._running_time = None
        self._result = {}
        self._docs = {}
        self._result_saving_params = {}
        self._result_loading_params = {}
        self._fmanagers = fmanagers
        self._result_folder_name = "result"

    def __repr__(self):
        return self.format_str()

    def __str__(self):
        return self.format_str()

    def format_str(self) -> str:
        showed_str = dedent(f"""{self.__class__.__name__} at {hex(id(self))}
    -----------
    Parameters:
    {self._log}
    -----------
    Result keys:
    {', '.join([i for i in self._result.keys()])}
    """)
        if self._running_time:
            showed_str += f"-----------\nRunning time: \n {self._running_time}"
        return showed_str

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return self._result[item]

    @property
    def result(self):
        return self._result

    def add_result(self, result):
        self._result.update(result)

    def add_running_time(self, t):
        self._running_time = t

    @property
    def log(self):
        return self._log

    def _load_results(self, parent_dir, result_types):
        result_dic = {}
        for fn in Path(parent_dir).iterdir():
            file_manager = self._fmanagers[result_types[fn.stem]].create()
            kws = {} if fn.stem not in self._result_loading_params else self._result_loading_params[fn.stem]
            result_dic[fn.stem] = file_manager.read(fn, **kws)
        self.add_result(result_dic)

    def _save_results(self, parent_dir):
        (parent_dir / self._result_folder_name).mkdir()
        for k, v in self._result.items():
            if any([isinstance(v, sp_type) for sp_type in [list, dict, set, np.ndarray]]):
                file_manager = self._fmanagers[self._result_saving_params[k]["fm_name"]].create()
            else:
                file_manager = self._fmanagers[str(type(v))].create()
            kws = {} if k not in self._result_saving_params else self._result_saving_params[k]
            file_manager.write(v, parent_dir / self._result_folder_name / k, **kws)

    def save(self,
             file_path):
        saved_dir = Path(file_path)
        saved_dir.mkdir(parents=True)
        print(f"Created a folder {file_path} to store the result")
        save_toml_file(saved_dir / "analysis_params.toml", {"running_time": self._running_time,
                                                            "log": self.log,
                                                            "result_types": {k: str(type(v))
                                                                             for k, v in self._result.items()}})
        self._save_results(parent_dir=saved_dir)

    @classmethod
    def load(cls,
             file_path: str,
             **kwargs):
        file_path = Path(file_path)
        all_configs = parse_toml_file(file_path / "analysis_params.toml")
        new_analysis = cls(log=all_configs["log"])
        new_analysis.add_running_time(all_configs["running_time"])
        new_analysis._load_results(parent_dir=file_path / new_analysis._result_folder_name,
                                   result_types=all_configs["result_types"])

    def plot(self, **kwargs):
        raise NotImplementedError()