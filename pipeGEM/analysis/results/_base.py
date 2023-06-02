from functools import wraps
from time import time
from textwrap import dedent


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


class BaseAnalysis:
    def __init__(self, log):
        self._log = log  # analysis record (parameter, model, and data name)
        self._running_time = None
        self._result = None
        self._docs = {}

    def __repr__(self):
        return repr(self.result)

    def __str__(self):
        return str(self._log)

    def format_str(self):
        showed_str = f"{self.__class__.__name__} at {hex(id(self))} \n" \
                     f"Parameters:"\
                     f"{self._log}"\ 
                     f"Results:"

        if self._running_time:
            showed_str += f"Running time: \n {self._running_time}"

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return self._result[item]

    @property
    def result(self):
        return self._result

    def add_running_time(self, t):
        self._running_time = t

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