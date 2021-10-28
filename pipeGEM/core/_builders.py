from threading import Lock

from pipeGEM.utils import ObjectFactory
from ._models import NamedModel
from ._groups import ModelGroup, ComplementGroup, Group
from ._batch import Batch, ComplementBatch, ComparableBatch


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

    _group_allow_collapse = ["all"]
    _unnamed_format = "unnamed_{obj}_{i}"
    _n_unnamed_models, _n_unnamed_groups, _n_unnamed_batches = 0, 0, 0

    @property
    def model_names(self):
        return self._model_names

    @property
    def group_names(self):
        return self._group_names

    @property
    def batch_names(self):
        return self._batch_names

    def _handle_unnamed(self, obj_type: str) -> str:
        if obj_type == NamedModel.obj_type:
            self._n_unnamed_models += 1
            return self._unnamed_format.format(obj="model", i=self._n_unnamed_models)
        elif obj_type == ModelGroup.obj_type:
            self._n_unnamed_groups += 1
            return self._unnamed_format.format(obj="group", i=self._n_unnamed_groups)
        elif obj_type == Batch.obj_type:
            self._n_unnamed_batches += 1
            return self._unnamed_format.format(obj="batch", i=self._n_unnamed_batches)

    def add(self, name, obj):
        obj_type = obj.obj_type
        if name is None:
            name = self._handle_unnamed(obj_type)

        if name in self._model_names:
            raise ValueError("This name is already taken")
        else:
            if obj_type == NamedModel.obj_type:
                self._model_names[name] = id(obj)
            elif obj_type == ModelGroup.obj_type:
                if name not in self._group_allow_collapse:
                    self._group_names[name] = id(obj)
            elif obj_type == Batch.obj_type:
                self._batch_names[name] = id(obj)
            else:
                raise TypeError("The obj type is not correct")
            return name

    def delete(self, name, obj):
        obj_type = obj.obj_type

        if obj_type == NamedModel.obj_type and name in self._model_names:
            self._model_names.pop(name)
        elif obj_type == ModelGroup.obj_type:
            if name in self._group_names:
                self._group_names.pop(name)
            elif not name in self._group_allow_collapse:
                raise ValueError(f"This {obj_type} is not in the name list")
        elif obj_type == Batch.obj_type and name in self._batch_names:
            self._batch_names.pop(name)
        else:
            raise ValueError(f"This {obj_type} is not in the name list")

    def update(self, name, new_name, obj):
        obj_type = obj.obj_type

        if obj_type == NamedModel.obj_type:
            cont = self._model_names
        elif obj_type == ModelGroup.obj_type:
            if new_name not in self._group_allow_collapse and name not in self._group_allow_collapse:
                cont = self._group_names
            else:
                raise ValueError(f"This name is not mutable or assignable: {name}")
        elif obj_type == Batch.obj_type:
            cont = self._batch_names
        else:
            TypeError("The obj type is not correct")

        if name in cont and new_name not in cont:
            cont[new_name] = cont.pop(name)
        elif new_name in cont:
            raise ValueError("This name is already taken")
        else:
            raise ValueError(f"This {obj_type} is not in the name list")


class ModelBuilders(ObjectFactory):
    def get(self, builder_name, **kwargs):
        return self.create(builder_name, **kwargs)


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

    def __call__(self, *args, **kwargs):
        kw = {k: v for k, v in kwargs.items()}
        kw["complement_batch"] = self._complement_batch
        kw["complement_group"] = self._complement_group
        return NamedModel(*args, **kwargs)


class ModelGroupBuilders(ObjectFactory):
    def get(self, builder_name, **kwargs):
        return self.create(builder_name, **kwargs)


class GroupBuilder:
    _instance = None
    _complement_group = None
    _complement_batch = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, *args, **kwargs):
        return Group(*args, **kwargs)


class ComplementGroupBuilder:
    _instance = None
    _complement_batch = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, *args, **kwargs):
        return ComplementGroup(*args, **kwargs)


class BatchBuilders(ObjectFactory):
    def get(self, builder_name, **kwargs):
        return self.create(builder_name, **kwargs)


class ComplementBatchBuilder:
    _instance = None
    _complement_group = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._complement_group = ComplementGroup(named_models=[],
                                                    batch=None,
                                                    name="_complement",
                                                    complement_batch=cls._instance,
                                                    name_manager=NameManager())
        return cls._instance

    def __call__(self, *args, **kwargs):
        return ComplementBatch(complement_group=self._complement_group,
                               complement_batch=None,
                               groups=[],
                               name="_complement",
                               *args, **kwargs)


class ComparableBatchBuilder:
    _instance = None
    _complement_group = None
    _complement_batch = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._complement_batch = kwargs.get("complement_batch")
            cls._complement_group = ComplementGroup(named_models=[],
                                                    batch=None,
                                                    name="_complement",
                                                    complement_batch=cls._complement_batch,
                                                    name_manager=NameManager())
        return cls._instance

    def __call__(self, *args, **kwargs):
        return ComparableBatch(*args, **kwargs)


def get_batch(builders, **kwargs):
    """API"""
    return builders.get(**kwargs)


def get_group(builders, complement_batch, **kwargs):
    """API"""
    kwargs.update({"complement_batch": complement_batch})
    return builders.get(**kwargs)


def get_model(builders, complement_batch, complement_group, **kwargs):
    """API"""
    kwargs.update({"complement_batch": complement_batch,
                   "complement_group": complement_group})
    return builders.get(**kwargs)