from threading import Lock

from pipeGEM.utils import ObjectFactory
from ._models import NamedModel
from ._groups import ModelGroup, ComplementGroup, Group, Batch
from ._batch import ComplementBatch


class _SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class NameManager(metaclass=_SingletonMeta):
    _model_names: dict = {}
    _group_names: dict = {}
    _batch_names: dict = {}

    @property
    def model_names(self):
        return self._model_names

    @property
    def group_names(self):
        return self._group_names

    def add(self, name, obj):
        if name in self._model_names:
            raise ValueError("This name is already taken")
        else:
            if isinstance(obj, NamedModel):
                self._model_names[name] = id(obj)
            elif isinstance(obj, ModelGroup):
                self._group_names[name] = id(obj)
            elif isinstance(obj, Batch):
                self._batch_names[name] = id(obj)
            else:
                raise TypeError("The obj type is not correct")
            return name

    def delete(self, name, obj):
        if isinstance(obj, NamedModel) and name in self._model_names:
            self._model_names.pop(name)
        elif isinstance(obj, ModelGroup) and name in self._group_names:
            self._group_names.pop(name)
        elif isinstance(obj, Batch) and name in self._batch_names:
            self._batch_names.pop(name)
        else:
            raise ValueError("This model is not in the name list")

    def update(self, name, new_name, obj):
        if isinstance(obj, NamedModel):
            cont = self._model_names
        elif isinstance(obj, ModelGroup):
            cont = self._group_names
        elif isinstance(obj, Batch):
            cont = self._batch_names
        else:
            TypeError("The obj type is not correct")

        if name in cont and new_name not in cont:
            cont[new_name] = cont.pop(name)
        elif new_name in cont:
            raise ValueError("This name is already taken")
        else:
            raise ValueError("This model is not in the name list")


class ModelBuilder:
    _instance = None
    _complement_group = None
    _complement_batch = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._complement_batch = kwargs.get("complement_batch")
            cls._complement_group = kwargs.get("complement_group")
        return cls._instance

    def get(self, *args, **kwargs):
        kw = {k: v for k, v in kwargs.items()}
        kw["complement_batch"] = self._complement_batch
        kw["complement_group"] = self._complement_group
        return NamedModel(*args, **kwargs)


class ModelGroupBuilders(ObjectFactory):
    def get(self, name, **kwargs):
        return self.create(name, **kwargs)


class GroupBuilder:
    _instance = None
    _complement_group = None
    _complement_batch = None

    def __new__(cls, *args, **kwargs):
        assert "complement_batch" in kwargs
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._complement_batch = kwargs.get("complement_batch")
            cls._complement_group = cls._complement_batch.complement_group
        return cls._instance

    def __call__(self, *args, **kwargs):
        return Group(*args, **kwargs)


class BatchBuilders(ObjectFactory):
    def get(self, name, **kwargs):
        return self.create(name, **kwargs)


class ComplementBatchBuilder:
    _instance = None
    _complement_group = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._complement_group = ComplementGroup(**kwargs)
        return cls._instance

    def __call__(self, *args, **kwargs):
        return ComplementBatch(*args, **kwargs)


class ComparableBatchBuilder:
    _instance = None
    _complement_group = None
    _complement_batch = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._complement_batch = kwargs.get("complement_batch")
            cls._complement_group = ComplementGroup(**kwargs)
        return cls._instance

    def __call__(self, *args, **kwargs):
        return Group(*args, **kwargs)  # TODO: change


def get_group(builder, complement_batch, *args, **kwargs):
    """API"""
    kwargs.update({"complement_batch": complement_batch})
    return builder.get(*args, **kwargs)


def get_model(builder, complement_batch, complement_group, *args, **kwargs):
    """API"""
    kwargs.update({"complement_batch": complement_batch, "complement_group": complement_group})
    return builder.get(*args, **kwargs)


def get_batch(builder, *args, **kwargs):
    """API"""
    return builder.get(*args, **kwargs)