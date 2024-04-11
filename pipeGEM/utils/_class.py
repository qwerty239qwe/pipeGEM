class ObjectFactory:

    def __init__(self):
        self._builders = {}

    def __getitem__(self, item):
        return self._builders[item]

    def register(self, name, builder):
        self._builders[name] = builder

    def create(self, builder_name, **kwargs):
        builder = self._builders.get(builder_name)
        if not builder:
            raise KeyError(builder_name)
        return builder(**kwargs)

    def items(self):
        return self._builders.items()


class ClassPropertyDescriptor:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def is_iter(obj) -> bool:
    try:
        iter(obj)
        return True
    except TypeError:
        return False
