from typing import List, Union, Dict
from functools import lru_cache

import pandas as pd

from ._groups import Group, VirtualGroup, ComplementGroup
from ._models import NamedModel
from pipeGEM.utils import classproperty


NOT_IN_ANY_GROUP = "_NotInAnyGroup"


class Batch:
    _obj = "batch"

    def __init__(self,
                 groups: Union[List[Group], Dict[str, List[NamedModel]]],
                 complement_group,
                 complement_batch,
                 name_manager,
                 name = None
                 ):

        self._name = name_manager.add(name, self)
        self._name_manager = name_manager
        self._complement_group = complement_group
        self._complement_batch = complement_batch

        self._groups = self._check_groups(groups)
        self._models = VirtualGroup([mod for name, group in self._groups.items() for mod in group],
                                    batch=self, name="all", name_manager=self._name_manager,
                                    complement_batch=self._complement_batch)

    def __len__(self):
        return len(self._models)

    def __getitem__(self, item):
        return self._models[item]

    def __del__(self):
        for name, group in self._groups.items():
            group.batch = self._complement_batch

    @lru_cache(maxsize=128)
    def component(self):
        return pd.DataFrame([{'reactions': len(mod.reactions),
                              'metabolites': len(mod.metabolites),
                              'genes': len(mod.genes)}
                             for mod in self._models],
                            columns=['reactions', 'metabolites', 'genes'],
                            index=[mod.name for mod in self._models])

    def _check_groups(self, groups) -> dict:
        if isinstance(groups, list):
            assert all([isinstance(g, Group) for g in groups]), "all group in groups should be Group objs"
            dic = {group.name: group for group in groups}
        elif isinstance(groups, dict):
            dic = {name: Group(named_models=models, name=name, batch=self,
                               complement_batch=self._complement_batch,
                               name_manager=self._name_manager)
                    for name, models in groups.items()}
        elif isinstance(groups, Group):
            dic = {groups.name: groups}
        else:
            raise TypeError("")
        for name, group in dic.items():
            group.batch = self
        return dic

    def _check_models(self, models) -> dict:
        if isinstance(models, list):
            assert all([isinstance(m, NamedModel) for m in models]), "all group in groups should be Group objs"
            dic = {model.name: model for model in models}
        elif isinstance(models, dict):
            dic = {name: NamedModel(model=mod,
                                    name=name,
                                    batch=self,
                                    complement_batch=self._complement_batch,
                                    complement_group=self._complement_group,
                                    group=self._complement_group,
                                    name_manager=self._name_manager)
                   for name, mod in models.items()}
        elif isinstance(models, NamedModel):
            dic = {models.name: models}
        else:
            raise TypeError("")
        for name, model in dic.items():
            model.batch = self
        return dic

    def items(self):
        return self._models.items()

    @classproperty
    def obj_type(self):
        return self._obj

    @property
    def models(self):
        return self._models

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, group_schema: dict):
        self.init_group_from_dict(group_schema)

    @property
    def complement_group(self):
        return self._complement_group

    def init_group_from_dict(self, group_schema: dict) -> None:
        self._groups = {}
        rest_models = {name: model for name, model in zip(self._models.get_all_model_labels(),
                                                          self._models.get_models())}
        if group_schema is not None:
            for group_name, model_names in group_schema.items():
                for model_name in model_names:
                    assert model_name in self._models, "all model names should be in this batch, " \
                                                       "use get_model_labels to get all valid model names"
                    rest_models.pop(model_name)
                new_group = Group([self._models[model_name] for model_name in model_names],
                                  name=group_name,
                                  batch=self,
                                  complement_batch=self._complement_batch,
                                  name_manager=self._name_manager)
                self._groups[group_name] = new_group
        self._complement_group = ComplementGroup(list(rest_models.values()),
                                                 name=NOT_IN_ANY_GROUP,
                                                 name_manager=self._name_manager,
                                                 complement_batch=self._complement_batch,
                                                 batch=self)

    def _repr_html_(self):
        model_rows = [self._gen_table_row(mod) for mod in self._models]
        tab = f"""<table>
                      <tr>
                          <td>Name</td>
                          <td>Group</td>
                          <td>annotation</td>
                          <td>Reactions</td>
                          <td>Metabolites</td>
                          <td>Genes</td>
                      </tr>
                      <tr>{"</tr><tr>".join(model_rows)}</tr>
                  </table>"""
        return tab

    @staticmethod
    def _gen_table_row(named_model: NamedModel) -> str:
        rxn = named_model.model.reactions
        met = named_model.model.metabolites
        gen = named_model.model.genes
        grp = named_model.group
        annot = named_model.annotation
        return f"""<td>{named_model.name}</td>
                    <td>{grp}</td>
                    <td>{annot}</td>
                    <td>{len(rxn)}</td>
                    <td>{len(met)}</td>
                    <td>{len(gen)}</td>"""

    def group_sizes(self):
        return {name: len(group) for name, group in self._groups.items()}

    def to_frame(self):
        n_rows = max(max(self.group_sizes().values()), len(self._complement_group))
        df = pd.DataFrame({name: models.to_list()+[None for _ in range(n_rows - len(models))]
                           for name, models in self._groups.items()})
        df["not_in_any_group"] = self._complement_group.to_list()+[None for _ in range(n_rows -
                                                                                       len(self._complement_group))]
        return df

    def get_group_names(self) -> List[str]:
        return [name for name in self._groups.keys()]

    def get_group(self, group_name) -> Group:
        return self._groups[group_name]

    def add(self,
            groups = None,
            models = None):
        if groups is None and models is None:
            raise ValueError("input groups or models to add new components")

        if groups is not None:
            if isinstance(groups, dict):
                for g in groups:
                    assert g not in self._groups, "groups are already in the batch"
            added = self._check_groups(groups)
            self._groups.update(added)
            for name, models in added.items():
                self._models.add_models(models)
        if models is not None:
            if isinstance(models, dict):
                for m in models:
                    assert m not in self._models, "models are already in the batch"
            added = self._check_models(models)
            self._models.add_models(list(added.values()))

    def create_group(self, group_name, model_names):
        new_group = Group([self._models[name] for name in model_names],
                          name=group_name,
                          batch=self,
                          complement_batch=self._complement_batch,
                          name_manager=self._name_manager)
        self._groups[group_name] = new_group

    def remove(self,
               groups = None,
               models = None):

        if groups is None and models is None:
            raise ValueError("input groups or models to remove components")

        if groups is not None:
            self._groups[groups].leave_batch()
            if isinstance(groups, list):
                for g in groups:
                    self._groups[g].leave_batch()
        if models is not None:
            self._models[models].leave_batch()
            if isinstance(models, list):
                for m in models:
                    self._models[m].leave_batch()

    def rename(self,
               groups = None,
               models = None):
        if groups is None and models is None:
            raise ValueError("input groups or models to rename components")

        if groups is not None:
            for old, new in groups.items():
                self._groups[old].name = new

        if models is not None:
            for old, new in models.items():
                self._models[old].name = new

    def pop_model(self, name):
        for gname, group in self._groups.items():
            if name in group:
                return group.pop_model(name)

    def pop_models(self, names):
        if isinstance(names, str):
            self.pop_model(names)
        elif isinstance(names, list):
            for name in names:
                self.pop_model(name)
        else:
            raise ValueError("Names should be a list of str or a str.")

    def ungroup_model(self, name) -> None:
        self._models[name].leave_group()

    def get_all_model_labels(self) -> List[str]:
        mods = sorted([model for _, model in self._models.items()], key=lambda x: x.group.name)
        return [mod.name for mod in mods]

    def _get_model_label(self, query) -> list:
        if query in self._groups:
            return self._groups[query].get_all_model_labels()
        elif query in self._models:
            return [query]
        else:
            raise ValueError("No such model name or group name in this supergroup")

    def get_model_labels(self, queries="all"):
        if queries == "all":
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

    def save_analysis(self, file_dir_path):
        self._models.save_analysis(file_dir_path)

    def load_analysis(self, file_dir_path):
        self._models.load_analysis(file_dir_path)


class ComplementBatch(Batch):
    _instance = None

    def __init__(self,
                 name,
                 groups: Union[List[Group], Dict[str, List[NamedModel]]],
                 complement_group,
                 complement_batch,
                 name_manager
                 ):
        super().__init__(name=name,
                         groups=groups,
                         complement_group=complement_group,
                         complement_batch=complement_batch,
                         name_manager=name_manager)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class ComparableBatch(Batch):
    def __init__(self,
                 name,
                 groups: Union[List[Group], Dict[str, List[NamedModel]]],
                 complement_group,
                 complement_batch,
                 name_manager
                 ):
        super().__init__(name=name,
                         groups=groups,
                         complement_group=complement_group,
                         complement_batch=complement_batch,
                         name_manager=name_manager)