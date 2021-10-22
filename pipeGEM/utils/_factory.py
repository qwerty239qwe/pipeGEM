class ObjectFactory:

    def __init__(self):
        self._builders = {}

    def register(self, name, builder):
        self._builders[name] = builder

    def create(self, name, **kwargs):
        builder = self._builders.get(name)
        if not builder:
            raise KeyError(name)
        return builder(**kwargs)