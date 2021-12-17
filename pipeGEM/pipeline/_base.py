from collections import OrderedDict


__all__ = ["Pipeline", "Config"]


class Config:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        pass


class Logger:
    def __init__(self, pipeline, verbose=True, file_name=None):
        self.pipeline = pipeline
        self.verbose = verbose
        self.file_name = file_name

    def __call__(self, *args, **kwargs):
        log = self.pipeline.get_log(*args, **kwargs)
        if self.verbose:
            print(log)
        if self.file_name is not None:
            with open(self.file_name, "w+") as f:
                f.write(log)


class Pipeline:
    def __init__(self, *args, **kwargs):
        self.output = None
        self._pre_hooks = OrderedDict()  # affect on every jobs
        self._post_hooks = OrderedDict()
        self._jobs = OrderedDict()

    def __str__(self):
        return self.__class__.__name__ + self._next_layer_tree()

    def get_log(self, *args, **kwargs):
        raise NotImplementedError()

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