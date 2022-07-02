
class BaseData:
    def __init__(self, hook_name):
        self._hook_name = hook_name
        self._hooked_attr = {}

    def __getattr__(self, item):
        if self._hook_name == item:
            return self._hooked_attr[item]
        return getattr(self, item)

    def clean(self):
        self._hooked_attr = {}

    def align(self, model):
        raise NotImplementedError
