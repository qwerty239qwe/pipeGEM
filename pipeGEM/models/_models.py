from typing import List, Union
from enum import Enum

from pathlib import Path

import pandas as pd
import cobra


from pipeGEM.integration.mapping import Expression
from pipeGEM.analysis import FluxAnalyzer


__all__ = ("NamedModel", "ModelGroup", "Batch",
           "ReturnedType",
           "NOT_CATEGORIZED_LABEL", "ALL_LABEL")


NOT_CATEGORIZED_LABEL = "_NotCategorized"
NOT_IN_ANY_GROUP = "_NotInAnyGroup"
NOT_IN_ANY_BATCH = "_NotInAnyBatch"




class NamedModel:
    def __init__(self,
                 name,
                 model,
                 group,
                 batch,
                 complement_group,
                 complement_batch,
                 name_manager):
        self._name_manager = name_manager
        self._name = self._name_manager.add(name, self)
        self._model = model
        self._rxn_ids = [rxn.id for rxn in self._model.reactions]
        self._expression = None
        self._analyzer = None
        self._group = group
        self._batch = batch
        self._complement_group = complement_group
        self._complement_batch = complement_batch

    def __str__(self):
        return f"Named model [{self.name}]\n{self._model}"

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, item):
        return getattr(self._model, item)

    def __del__(self):
        if isinstance(self._group, ModelGroup):
            self._group.pop_models(self.name)
        if isinstance(self._batch, Batch):
            self._batch.pop_models(self.name)
        self._name_manager.delete(self.name, self)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if isinstance(self._group, ModelGroup):
            self._group.rename_model(self._name, new_name)
        self._name_manager.update(self.name, new_name, self)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        assert isinstance(val, cobra.Model)
        self._model = val
        self._rxn_ids = [rxn.id for rxn in self._model.reactions]

    @property
    def rxn_ids(self):
        return self._rxn_ids

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, group):
        assert isinstance(group, ModelGroup)
        # unlink the model from its original group and supergroup
        if isinstance(self._batch, Batch):
            self._batch.pop_models(self._name)
        elif isinstance(self._group, ModelGroup):
            self._group.pop_models(self._name)
        self._group = group
        self._batch = group.batch

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch):
        # assert isinstance(supergroup, ModelSuperGroup) or isinstance(supergroup, cobrave.comparison.ModelComparer)
        if batch is not self._batch:
            if isinstance(self._batch, Batch):
                self._batch.pop_models(self._name)
            elif isinstance(self._group, ModelGroup):
                self._group.pop_models(self._name)
            self._batch = batch

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, data):
        self._expression = Expression(self._model, data)

    def leave_group(self):
        self._group.pop_models(self._name)
        self._group = self._complement_group

    def leave_batch(self):
        self._batch.pop_models(self._name)
        self._batch = self._complement_batch

    def set_analyzer(self, solver: str):
        self._analyzer = FluxAnalyzer(model=self._model,
                                      solver=solver,
                                      rxn_expr_score=self.expression)

    def get_analyzer(self):
        return self._analyzer

    def get_analysis(self, method, constr="default", keep_rc=False):
        return self._analyzer.get_df(method=method, constr=constr, keep_rc=keep_rc)

    def save_analysis(self, file_dir_path):
        path = Path(file_dir_path) / Path(self._name)
        path.mkdir(parents=True, exist_ok=True)
        self._analyzer.save_analysis(file_path=path)

    def load_analysis(self, file_dir_path):
        path = Path(file_dir_path) / Path(self._name)
        self._analyzer.load_analysis(path)


ALL_LABEL = "all"


class ReturnedType(Enum):
    List = "list"
    Dict = "dict"
    Default = "dict"
    Set = "set"
    DF = "pandas.DataFrame"


class ModelGroup:
    def __init__(self,
                 named_models: List[NamedModel],
                 name: str,
                 batch,
                 name_manager):
        self._name_manager = name_manager
        self._name = self._name_manager.add(name, self)
        self._named_models = named_models
        self._batch = batch
        for model in self._named_models:
            model.group = self
            model.batch = self._batch
        self._model_dict = {mod.name: mod for mod in self._named_models}
        self._mem_dic = {}

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelGroup [{self.name}] in {self._batch} with models: \n" + \
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

    @property
    def name(self):
        return self._name

    def items(self):
        return self._model_dict.items()

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
            if isinstance(self._model_dict[name].batch, Batch):
                self._model_dict[name].leave_group()
            del self._model_dict[name]
            return self._del_model_from_list(name)
        raise ValueError("The model is not found in this model_group")

    def pop_models(self, names):
        if isinstance(names, str):
            self.pop_model(names)
        elif isinstance(names, list):
            for name in names:
                self.pop_model(name)
        else:
            raise ValueError("Names should be a list of str or a str.")

    def sort_model(self, key=lambda x: x.name):
        self._named_models = sorted(self._named_models, key=key)

    def leave_batch(self):
        """
        Triggered when the group is removed from a batch
        """
        self._batch.delete_group(self.name)

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


class Batch:
    def __init__(self,
                 groups: List[ModelGroup],
                 complement_group,
                 complement_batch
                 ):
        self._groups = {group.name: group for group in groups}
        self._groups[NOT_IN_ANY_GROUP] = complement_group
        self._complement_group = complement_group
        self._complement_batch = complement_batch
        self._models = ModelGroup([mod for name, group in self._groups.items() for mod in group],
                                  supergroup=self,
                                  is_real_group=False)
        for name, group in self._groups.items():
            group.batch = self

    def __len__(self):
        return len(self._models)

    def __getitem__(self, item):
        return self._models[item]

    def items(self):
        return self._models.items()

    @property
    def models(self):
        return self._models

    @property
    def groups(self):
        return {k: v for k, v in self._groups.items() if k != NOT_IN_ANY_GROUP}

    @groups.setter
    def groups(self, group_schema: dict):
        self.init_group_from_dict(group_schema)

    @property
    def complement_group(self):
        return self._complement_group

    def init_group_from_dict(self, group_schema: dict) -> None:
        self._groups = {}
        rest_models = {name: model for name, model in zip(self._models.get_all_model_labels(), self._models.get_models())}
        if group_schema is not None:
            for group_name, model_names in group_schema.items():
                for model_name in model_names:
                    assert model_name in self._models, "all model names should in this supergroup, " \
                                                       "use get_model_labels to get all valid model names"
                    rest_models.pop(model_name)
                new_group = ModelGroup([self._models[model_name] for model_name in model_names],
                                       name=group_name,
                                       supergroup=self)
                self._groups[group_name] = new_group
        self._groups[NOT_IN_ANY_GROUP] = ModelGroup(list(rest_models.values()), name=NOT_IN_ANY_GROUP, supergroup=self)

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

    def get_group_names(self) -> List[str]:
        return [name for name in self._groups.keys() if name != NOT_IN_ANY_GROUP]

    def get_group(self, group_name) -> ModelGroup:
        return self._groups[group_name]

    def add_group(self, added_group):
        assert added_group.name not in self._groups, "This group is already in the supergroup"
        self._groups[added_group.name] = added_group
        added_group.assign_supergroup_to_models(self)
        self._models.add_models(added_group.get_models())

    def create_group(self, group_name, model_names):
        new_group = ModelGroup([self._models[name] for name in model_names],
                               name=group_name,
                               supergroup=self)
        self._groups[group_name] = new_group

    def delete_group(self, group_name):
        target_group = self._groups[group_name]
        self._models.pop_models(target_group.get_model_labels())
        target_group.batch = self._complement_batch
        target_group.remove_supergroup_from_models()
        del self._groups[group_name]

    def add_models(self, models, model_names):
        if isinstance(models, list):
            if all([isinstance(model, NamedModel) for model in models]):
                self._models.add_models(models)
                for model in models:
                    model.supergroup = self
            else:
                assert len(models) == len(model_names), ""
                self._models.add_models([NamedModel(name=name, model=model)
                                         for name, model in zip(model_names, models)])
        elif isinstance(models, dict):
            self._models.add_models([NamedModel(name=name, model=model)
                                       for name, model in models.items()])
        else:
            raise ValueError("models should be a dict or a list")

    def rename_model(self, new_labels: Union[List[str], str], old_labels: Union[List[str], str]):
        assert type(new_labels) == type(old_labels)
        if isinstance(new_labels, str):
            self._models.rename_model(old_label=old_labels, new_label=new_labels)
        elif isinstance(new_labels, list):
            assert len(new_labels) == len(old_labels)
            for o, n in zip(old_labels, new_labels):
                self._models.rename_model(old_label=o, new_label=n)
        else:
            raise ValueError("New and old labels should be a list or a string")

    def pop_model(self, name):
        for group in self._groups:
            if group.has_model(name):
                popped = group.pop_model(name)
                popped.leave_batch()
        return self._models.pop_model(name)

    def pop_models(self, names):
        if isinstance(names, str):
            self.pop_model(names)
        elif isinstance(names, list):
            for name in names:
                self.pop_model(name)
        else:
            raise ValueError("Names should be a list of str or a str.")

    def ungroup_model(self, name) -> None:
        self._models[name].group = self._groups[NOT_IN_ANY_GROUP]

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

    def save_analysis(self, file_dir_path):
        self._models.save_analysis(file_dir_path)

    def load_analysis(self, file_dir_path):
        self._models.load_analysis(file_dir_path)

