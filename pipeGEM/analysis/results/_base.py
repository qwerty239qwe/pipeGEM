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


def _get_module(v, **sp_module_names):
    if '__module__' in v.__dir__():
        return v.__module__.split(".")[0]
    if "ndarray" == v.__class__.__name__:
        return "numpy"
    for k, v in sp_module_names.items():
        if k == v.__class__.__name__:
            return v

    return "python"


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
        self._s_val_tps = [str, float, int, ]

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
            if fn.name == "other_values.toml":
                result_dic.update(parse_toml_file(fn))
                continue

            if result_types[fn.stem] == "Analysis":
                result_dic[fn.stem] = BaseAnalysis.load(str(fn))
                continue

            file_manager = self._fmanagers[result_types[fn.stem]]()
            kws = {} if fn.stem not in self._result_loading_params else self._result_loading_params[fn.stem]
            result_dic[fn.stem] = file_manager.read(fn, **kws)
        self.add_result(result_dic)

    def _save_results(self, parent_dir, result_types):
        (parent_dir / self._result_folder_name).mkdir()
        singular_values = {}
        for k, v in self._result.items():
            if any([isinstance(v, tp) for tp in self._s_val_tps]):
                singular_values[k] = v
                continue

            if result_types[k] == "Analysis":
                v.save(parent_dir / self._result_folder_name / k)
                continue

            file_manager = self._fmanagers[result_types[k]]()
            kws = {} if k not in self._result_saving_params else {k: v for k, v in self._result_saving_params[k].items()
                                                                  if k not in ["fm_name"]}
            file_manager.write(v, parent_dir / self._result_folder_name / k, **kws)
        save_toml_file(parent_dir / self._result_folder_name / "other_values.toml", singular_values)

    def save(self,
             file_path):
        saved_dir = Path(file_path)
        saved_dir.mkdir(parents=True)
        print(f"Created a folder {file_path} to store the result")
        result_types = {}
        for k, v in self._result.items():
            module_sp_kw = self._result_saving_params[k]["module_name"] if (
                    k in self._result_saving_params and "module_name" in self._result_saving_params[k]) else {}
            module_name = _get_module(v, **module_sp_kw)
            if k in self._result_saving_params and "fm_name" in self._result_saving_params[k]:
                result_types[k] = f"{module_name}.{self._result_saving_params[k]['fm_name']}"
            elif isinstance(v, BaseAnalysis):
                result_types[k] = "Analysis"
            else:
                result_types[k] = f"{module_name}.{v.__class__.__name__}"

        save_toml_file(saved_dir / "analysis_params.toml", {"running_time": self._running_time,
                                                            "log": self.log,
                                                            "result_types": result_types})
        self._save_results(parent_dir=saved_dir, result_types=result_types)

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
        return new_analysis

    def plot(self, **kwargs):
        raise NotImplementedError()