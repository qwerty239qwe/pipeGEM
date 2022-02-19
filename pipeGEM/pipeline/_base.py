from collections import OrderedDict
from functools import partial
import warnings


__all__ = ["Pipeline", "Config"]


class Config:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        pass


class Pipeline:
    def __init__(self, *args, **kwargs):
        self.output = None
        self._pre_hooks = OrderedDict()  # affect every jobs
        self._post_hooks = OrderedDict()
        self._jobs = OrderedDict()
        self._prev_lvl_pl_name = kwargs.get("container_name")
        self._verbosity = kwargs.get("verbosity") or 0
        self._setup_loggers()
        self._debug(f"Init {type(self).__name__} pipeline.")

    def _setup_loggers(self):
        self._debug = partial(self._log_msg, is_warning=False, verbose=self._verbosity > 1)
        self._info = partial(self._log_msg, is_warning=False, verbose=self._verbosity > 0)
        self._warn = partial(self._log_msg, is_warning=True, verbose=self._verbosity > -1)

    @staticmethod
    def _log_msg(val, is_warning, verbose):
        if not verbose:
            return
        if is_warning:
            warnings.warn(val)
        else:
            print(val)

    def __str__(self):
        return self.__class__.__name__ + self._next_layer_tree()

    def _next_layer_tree(self):
        pipelines = [getattr(self, name) for name in dir(self)
                     if name not in ['__weakref__'] and isinstance(getattr(self, name), Pipeline)]
        result = ""
        if len(pipelines) > 1:
            result = "\n├── "
            result += "\n├── ".join(str(pipelines[:-1]))
        if len(pipelines) ==1:
            result = "\n├── "
            result += "\n└── " + str(pipelines[-1])
        elif len(pipelines) > 0:
            result += "\n└── " + str(pipelines[-1])
        return result

    def __call__(self, *args, **kwargs):
        if len(self._pre_hooks) == 0 and len(self._post_hooks) == 0:
            output = self.run(*args, **kwargs)
        else:
            if len(self._pre_hooks) != 0:
                for name, hook in self._pre_hooks.items():
                    output = hook(*args)
                    if output is not None:
                        args = output if isinstance(output, tuple) else (output,)
            output = self.run(*args, **kwargs)
            if len(self._post_hooks) != 0:
                for name, hook in self._post_hooks.items():
                    output = hook(*args)
                    if output is not None:
                        args = output if isinstance(output, tuple) else (output,)
        return output

    def __getattr__(self, item):
        if item in self.__dict__["_jobs"]:
            return self.__dict__["jobs"][item]
        raise AttributeError(f"This object doesn't have this attr: {item}")

    def add_job(self):
        pass

    def add_pre_hook(self, hook):
        pass

    def delete_pre_hook(self):
        pass

    def add_post_hook(self):
        pass

    def delete_post_hook(self):
        pass

    def run(self, *args, **kwargs):
        raise NotImplementedError()