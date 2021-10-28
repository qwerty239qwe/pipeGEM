from enum import Enum
from typing import List

import pandas as pd

from ._models import NamedModel
from pipeGEM.utils import classproperty


ALL_LABEL = "all"


class ReturnedType(Enum):
    List = "list"
    Dict = "dict"
    Default = "dict"
    Set = "set"
    DF = "pandas.DataFrame"


class ModelGroup:
    _obj = "group"

    def __init__(self,
                 named_models: List[NamedModel],
                 name_manager,
                 complement_batch,
                 batch = None,
                 name: str = None):
        self._name_manager = name_manager
        self._name = self._name_manager.add(name, self)
        self._named_models = named_models
        self._model_dict = {mod.name: mod for mod in self._named_models}
        print(self._model_dict, type(self))
        self._batch = batch if batch is not None else complement_batch
        self._complement_batch = complement_batch
        for model in self._named_models:
            model.batch = self._batch

        self._mem_dic = {}

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelGroup [{self.name}] in {self._batch} with core: \n" + \
               ", ".join(list(self._model_dict.keys()))

    def __len__(self):
        return len(self._model_dict)

    def __getitem__(self, item):
        return self._model_dict[item]

    def __contains__(self, item):
        return item in self._model_dict

    def __iter__(self):
        index = 0
        while index < len(self._named_models):
            yield self._named_models[index]
            index += 1

    def __del__(self):
        for model in self._named_models:
            model.leave_group()
        self._name_manager.delete(self.name, self)

    @classproperty
    def obj_type(self):
        return self._obj

    @property
    def name(self):
        return self._name

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch):
        from ._batch import Batch
        assert isinstance(batch, Batch)
        self._batch = batch
        for model in self._named_models:
            model.batch = batch

    def items(self):
        return self._model_dict.items()

    def to_list(self):
        return self._named_models

    def get_all_model_labels(self):
        return list(self._model_dict.keys())

    def _get_model_label(self, query) -> list:
        if query in self._model_dict:
            return [query]
        else:
            raise ValueError("No such model name in this group")

    def get_model_labels(self, queries):
        if queries == ALL_LABEL:
            return self.get_all_model_labels()
        if isinstance(queries, str):
            return self._get_model_label(queries)
        elif isinstance(queries, list):
            labels = []
            for q in queries:
                labels.extend(self._get_model_label(q))
            return labels
        else:
            ValueError("Invalid type")

    def get_models(self):
        return list(self._model_dict.values())

    def rename_model(self, old_label, new_label):
        self._model_dict[new_label].name = new_label
        self._model_dict[new_label] = self._model_dict[old_label].pop()

    def _add_model(self, model):
        assert model.name not in self._model_dict, "Cannot add a model that is already in the group"
        self._named_models.append(model)
        self._model_dict[model.name] = model

    def add_models(self, named_model):
        if isinstance(named_model, NamedModel):
            self._add_model(named_model)
        elif isinstance(named_model, list):
            for mod in named_model:
                self._add_model(mod)
        else:
            raise ValueError("Input should be a list of named model or a named model")

    def _del_model_from_list(self, name):
        for i, mod in enumerate(self._named_models):
            if mod.name == name:
                return self._named_models.pop(i)

    def pop_model(self, name):
        if name in self._model_dict:
            print("found and pop:", name)
            if self._model_dict[name].group == self:
                self._model_dict[name].leave_group()
            del self._model_dict[name]
            return self._del_model_from_list(name)
        raise ValueError(f"The model is not found in this model_group: {type(self)}, {self.name}, {self._model_dict}")

    def pop_models(self, names):
        if isinstance(names, str):
            self.pop_model(names)
        elif isinstance(names, list):
            for name in names:
                self.pop_model(name)
        else:
            raise TypeError("Names should be a list of str or a str.")

    def sort_model(self, key=lambda x: x.name):
        self._named_models = sorted(self._named_models, key=key)

    def leave_batch(self):
        """
        Triggered when the group is removed from a batch
        """
        self.batch = self._complement_batch

    def get_flux_dfs(self, method, constr="default", type=ReturnedType.Default):
        if type.value not in self._mem_dic:
            if type.value == ReturnedType.List.value:
                self._mem_dic[type.value] = [mod.get_analysis(method, constr) for mod in self._named_models]
            if type.value == ReturnedType.Dict.value:
                self._mem_dic[type.value] = {mod.name: mod.get_analysis(method, constr) for mod in self._named_models}
            if type.value == ReturnedType.Set.value:
                self._mem_dic[type.value] = set([mod.get_analysis(method, constr) for mod in self._named_models])
            if type.value == ReturnedType.DF.value:
                self._mem_dic[type.value] = pd.concat([pd.DataFrame(mod.get_analysis(method, constr))
                                                       for mod in self._named_models], axis=1).fillna(0.0)
        return self._mem_dic[type.value]

    def save_analysis(self, file_dir_path):
        for model in self._named_models:
            model.save_analysis(file_dir_path)

    def load_analysis(self, file_dir_path):
        for model in self._named_models:
            model.load_analysis(file_dir_path)


class Group(ModelGroup):
    def __init__(self,
                 named_models: List[NamedModel],
                 name_manager,
                 complement_batch,
                 batch=None,
                 name: str = None,
                 ):
        super().__init__(named_models=named_models,
                         name=name,
                         complement_batch=complement_batch,
                         batch=batch,
                         name_manager=name_manager)
        for model in self._named_models:
            model.group = self

    @ModelGroup.batch.setter
    def batch(self, batch):
        from ._batch import Batch
        assert isinstance(batch, Batch)
        self._batch = batch
        for mod in self._named_models:
            mod.batch = batch

    def _add_model(self, model):
        super(Group, self)._add_model(model)
        model.group = self
        model.batch = self._batch

    def leave_batch(self):
        super().leave_batch()
        for mod in self._named_models:
            mod.leave_batch()


class VirtualGroup(ModelGroup):
    def __init__(self,
                 named_models: List[NamedModel],
                 name_manager,
                 complement_batch,
                 batch=None,
                 name: str = None,
                 ):
        super().__init__(named_models=named_models,
                         name=name,
                         batch=batch,
                         name_manager=name_manager,
                         complement_batch=complement_batch)


class ComplementGroup(ModelGroup):
    def __init__(self,
                 named_models: List[NamedModel],
                 name_manager,
                 complement_batch,
                 batch=None,
                 name: str = None,
                 ):
        super().__init__(named_models=named_models,
                         name=name,
                         batch=batch,
                         name_manager=name_manager,
                         complement_batch=complement_batch)
        for model in self._named_models:
            model.group = self

    @ModelGroup.batch.setter
    def batch(self, batch):
        from ._batch import Batch
        assert isinstance(batch, Batch)
        self._batch = batch
        for mod in self._named_models:
            mod.batch = batch