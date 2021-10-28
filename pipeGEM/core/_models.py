from typing import List, Union
from enum import Enum
from pathlib import Path

import cobra

from pipeGEM.integration.mapping import Expression
from pipeGEM.analysis import FluxAnalyzer
from pipeGEM.utils import classproperty


__all__ = ("NamedModel"
           "NOT_CATEGORIZED_LABEL")


NOT_CATEGORIZED_LABEL = "_NotCategorized"

NOT_IN_ANY_BATCH = "_NotInAnyBatch"


class NamedModel:
    _obj = "model"

    def __init__(self,
                 model,
                 name_manager,
                 complement_group,
                 complement_batch,
                 group=None,
                 batch=None,
                 name=None):
        self._name_manager = name_manager
        self._name = self._name_manager.add(name, self)
        self._model = model
        self._rxn_ids = [rxn.id for rxn in self._model.reactions]
        self._expression = None
        self._analyzer = None
        self._group = group if group is not None else complement_group
        self._batch = batch if batch is not None else complement_batch
        self._complement_group = complement_group
        self._complement_batch = complement_batch

    def __str__(self):
        return f"Named model [{self.name}]\n{self._model}"

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, item):
        return getattr(self._model, item)

    def __del__(self):
        pass
        # from ._groups import ModelGroup
        # from ._batch import Batch
        # if isinstance(self._group, ModelGroup):
        #     self._group.pop_models(self.name)
        # if isinstance(self._batch, Batch):
        #     self._batch.pop_models(self.name)
        # self._name_manager.delete(self.name, self)

    @classproperty
    def obj_type(self):
        return self._obj

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        from ._groups import ModelGroup
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
        from ._groups import ModelGroup
        from ._batch import Batch
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
        from ._groups import ModelGroup
        from ._batch import Batch
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
        self.group = self._complement_group

    def leave_batch(self):
        self.batch = self._complement_batch

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
