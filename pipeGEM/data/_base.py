
class BaseData:
    def __init__(self, hook_name):
        self._hook_name = hook_name
        self._hooked_attr = {}

    def clean(self):
        self._hooked_attr = {}

    def align(self, model, **kwargs):
        raise NotImplementedError()

