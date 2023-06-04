from functools import wraps
from time import time
from textwrap import dedent
from pathlib import Path

import pandas as pd

from pipeGEM.utils import save_toml_file, parse_toml_file, ObjectFactory


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

    func.__doc__ = """
    {descriptions}
    
    Parameters
    -------
    {parameters}
    {added_params}
        
    """.format(**default_docs)
    return wrapped


class BaseFileManager:
    def __init__(self, type, default_suffix):
        self._type = type
        self._default_suffix = default_suffix

    @property
    def suffix(self):
        return self._default_suffix

    def is_valid_type(self, obj):
        return isinstance(obj, self._type)

    def write(self, obj, file_name, **kwargs):
        raise NotImplementedError()

    def read(self, file_name, **kwargs):
        raise NotImplementedError()


class CSVFileManager(BaseFileManager):
    def __init__(self):
        super(CSVFileManager, self).__init__(pd.DataFrame, default_suffix=".csv")

    def write(self, obj, file_name, **kwargs):
        if "suffix" in kwargs:
            suffix = kwargs.pop("suffix")
        else:
            suffix = self.suffix

        obj.to_csv(Path(file_name).with_suffix(suffix), **kwargs)

    def read(self, file_name, **kwargs):
        return pd.read_csv(file_name, **kwargs)


class FileManagers(ObjectFactory):
    def __init__(self):
        super().__init__()


_fmanagers = FileManagers()
_fmanagers.register(type(pd.DataFrame), CSVFileManager)


class BaseAnalysis:
    def __init__(self, log):
        self._log = log  # analysis record (parameters used to reproduce the same result)
        self._running_time = None
        self._result = {}
        self._docs = {}
        self._result_saving_params = {}
        self._fmanagers = _fmanagers

    def __repr__(self):
        return self.format_str

    def __str__(self):
        return self.format_str

    def format_str(self) -> str:
        showed_str = f"""{self.__class__.__name__} at {hex(id(self))} \n
        Parameters:
        {self._log}\n
        Result keys:
        {self._result.keys()}
        """
        if self._running_time:
            showed_str += f"Running time: \n {self._running_time}"
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

    def _save_result(self, parent_dir):
        for k, v in self._result.items():
            file_manager = self._fmanagers[type(v)].create()
            kws = {} if k not in self._result_saving_params else self._result_saving_params[k]
            file_manager.write(v, parent_dir / k, **kws)

    def save(self, file_path):
        saved_dir = Path(file_path)
        saved_dir.mkdir(parents=True)
        print(f"Created a folder {file_path} to store the result")
        save_toml_file(saved_dir / "analysis_params.toml", {"running_time": self._running_time,
                                                            "log": self.log,
                                                            "result_types": {k: str(type(v))
                                                                             for k, v in self._result.items()}})
        self._save_result(parent_dir=saved_dir)

    @classmethod
    def load(cls, file_path, **kwargs):
        pass

    def plot(self, **kwargs):
        raise NotImplementedError()