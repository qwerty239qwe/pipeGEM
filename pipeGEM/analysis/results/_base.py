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
\n-----------\nParameters:\n{self._log}\n-----------\nResult keys:\n{', '.join([i for i in self._result.keys()])}
    """)
        if self._running_time:
            showed_str += f"-----------\nRunning time: \n {self._running_time}"
        return showed_str

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return self._result[item]

    @property
    def result(self) -> dict:
        return self._result

    def add_result(self, result) -> None:
        self._result.update(result)

    def add_running_time(self, t):
        self._running_time = t

    @property
    def log(self):
        return self._log

    def _load_results(self,
                      parent_dir,
                      result_types,
                      fn_to_key=None,
                      ignore_not_in_types=True):
        result_dic = {}
        fn_to_key = fn_to_key or {}
        for fn in Path(parent_dir).iterdir():
            if fn.stem in fn_to_key:
                result_key = fn_to_key[fn.stem]
            else:
                result_key = fn.stem

            if result_key not in result_types and ignore_not_in_types:
                continue

            if fn.name == "other_values.toml":
                result_dic.update(parse_toml_file(fn))
                continue

            if result_types[result_key] == "Analysis":
                result_dic[result_key] = BaseAnalysis.load(str(fn))
                continue

            file_manager = self._fmanagers[result_types[result_key]]()
            kws = {} if result_key not in self._result_loading_params else self._result_loading_params[result_key]
            result_dic[result_key] = file_manager.read(fn, **kws)
        self.add_result(result_dic)

    def _save_results(self, parent_dir, result_types):
        (parent_dir / self._result_folder_name).mkdir()
        singular_values = {}
        for k, v in self._result.items():
            if k not in result_types:
                continue

            if any([isinstance(v, tp) for tp in self._s_val_tps]):
                singular_values[k] = v
                continue

            if result_types[k] == "Analysis":
                v.save(parent_dir / self._result_folder_name / k)
                continue
            print(f"Saving {k}..")
            file_manager = self._fmanagers[result_types[k]]()
            kws = {} if k not in self._result_saving_params else {k: v for k, v in self._result_saving_params[k].items()
                                                                  if k not in ["fm_name"]}
            file_manager.write(v, parent_dir / self._result_folder_name / k, **kws)
            print(f"{k} is saved as a {result_types[k]}")
        save_toml_file(parent_dir / self._result_folder_name / "other_values.toml", singular_values)

    def save(self,
             file_path: str):
        saved_dir = Path(file_path)
        saved_dir.mkdir(parents=True)
        print(f"Created a folder {file_path} to store the result")
        result_types = {}
        for k, v in self._result.items():
            if v is None:
                print(f"Skipped {k} cause it is None")
                continue

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

    @classmethod
    def load_result(cls,
                    file_path,
                    key,
                    result_type):
        file_path = Path(file_path)
        new_analysis = cls(log={})
        new_analysis._load_results(parent_dir=file_path.parent,
                                   result_types={key: result_type},
                                   fn_to_key={file_path.stem: key})
        return new_analysis

    def plot(self, **kwargs):
        raise NotImplementedError()