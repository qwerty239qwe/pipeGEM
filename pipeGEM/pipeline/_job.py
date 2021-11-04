from collections import OrderedDict


__all__ = ["Pipeline"]


class Pipeline:
    def __init__(self, *args, **kwargs):
        self.output = None
        self._pre_hooks = OrderedDict()  # affect on every jobs
        self._post_hooks = OrderedDict()
        self._jobs = OrderedDict()

    def __call__(self, *args, **kwargs):
        if len(self._pre_hooks) == 0 and len(self._post_hooks) == 0:
            self.run(*args, **kwargs)
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