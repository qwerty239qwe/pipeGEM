from pathlib import Path
from typing import Union, Dict, List
from copy import deepcopy, copy
from itertools import chain


import pandas as pd
import cobra
import numpy as np

from pipeGEM.core._base import GEMComposite
from pipeGEM.utils import save_model, check_rxn_scales
from pipeGEM.data import GeneData, MediumData
from pipeGEM.integration import integrator_factory
from pipeGEM.integration.algo.fastcore import fastcc
from pipeGEM.analysis import flux_analyzers, FastCCAnalysis, TaskAnalysis
from pipeGEM.analysis.tasks import TaskHandler


class Model(GEMComposite):
    _is_leaf = True

    def __init__(self,
                 name_tag: str = None,
                 model = None):
        """
        Main model used to store cobra.Model and its name, omics data, and analyzer

        Parameters
        ----------
        name: str
            The name of this object, it will be used in a pg.Group object
        """
        super(Model, self).__init__(name_tag=name_tag)
        if not isinstance(model, cobra.Model):
            raise ValueError("input model should be a cobra model")
        self._model = model if model is not None else cobra.Model(name=name_tag)
        self._gene_data: Union[Dict[str, GeneData]] = {}
        self._medium_data = {}
        self._analysis = {}
        self._tasks = {}

    def __repr__(self):
        return f"pipeGEM Model {self._name_tag}"

    def __getattr__(self, item):
        if item not in self.__dict__:
            return getattr(self._model, item)
        return getattr(self, item)

    @property
    def size(self):
        return 1

    @property
    def reaction_ids(self):
        return [r.id for r in self.reactions]

    @property
    def gene_ids(self):
        return [g.id for g in self.genes]

    @property
    def metabolite_ids(self):
        return [m.id for m in self.metabolites]

    @property
    def subsystems(self):
        subs = {}
        for r in self.reactions:
            if r.subsystem in subs:
                subs[r.subsystem].append(r.id)
            else:
                subs[r.subsystem] = [r.id]
        return subs

    @property
    def gene_data(self):
        return self._gene_data

    def add_medium_data(self, name, data: Union[MediumData, pd.DataFrame], data_kwargs=None, **kwargs):
        if isinstance(data, pd.DataFrame):
            data_kwargs = {} if data_kwargs is None else data_kwargs
            data = MediumData(data, **data_kwargs)
        elif not isinstance(data, MediumData):
            raise TypeError()

        self._medium_data[name] = data
        self._medium_data[name].align(self, **kwargs)

    def apply_medium(self, name, **kwargs):
        self._medium_data[name].apply(self, **kwargs)

    def add_gene_data(self, name_or_prefix: str, data: Union[GeneData, pd.DataFrame], data_kwargs=None, **kwargs):
        data_kwargs = {} if data_kwargs is None else data_kwargs
        if isinstance(data, pd.DataFrame):
            data_dict = {f"{name_or_prefix}_{c}" if name_or_prefix else c:
                         GeneData(data[c], **data_kwargs) for c in data.columns}
            self._gene_data.update(data_dict)
            for k, v in data_dict.items():
                v.align(self, **kwargs)
            return
        elif isinstance(data, pd.Series):
            data = GeneData(data, **data_kwargs)
        elif not isinstance(data, GeneData):
            raise ValueError()

        self._gene_data[name_or_prefix] = data
        self._gene_data[name_or_prefix].align(self._model, **kwargs)

    def set_gene_data(self, name, data, data_kwargs=None, **kwargs):
        self._gene_data.pop(name)
        self.add_gene_data(name, data, data_kwargs, **kwargs)

    def add_tasks(self, name, tasks):
        self._tasks[name] = tasks

    def test_tasks(self, name, model_compartment_parenthesis="[{}]", **kwargs):
        tester = TaskHandler(model=self._model,
                             tasks_path_or_container=self._tasks[name],
                             model_compartment_parenthesis=model_compartment_parenthesis)
        return tester.test_tasks(**kwargs)

    def calc_ind_task_score(self, data_name, task_analysis: TaskAnalysis, all_na_indicator = -1, **kwargs):

        return {task_id: self._gene_data[data_name].calc_rxn_score_stat(rxn_ids=rxns,
                                                                        return_if_all_na=all_na_indicator,
                                                                        **kwargs)
                for task_id, rxns in task_analysis.task_support_rxns.items()}

    def get_activated_tasks(self, data_name, task_analysis: TaskAnalysis,
                            all_na_indicator=-1,
                            score_threshold=5*np.log10(2), **kwargs):
        ind_task_scores = self.calc_ind_task_score(data_name, task_analysis,
                                                   all_na_indicator=all_na_indicator, **kwargs)
        return [k for k, v in ind_task_scores.items() if v >= score_threshold or v == all_na_indicator]

    def get_activated_task_sup_rxns(self, data_name, task_analysis: TaskAnalysis, score_threshold=5*np.log10(2), include_supp_rxns=True):
        activated_tasks = self.get_activated_tasks(data_name, task_analysis, score_threshold)
        return list(set(chain(*[task_analysis.get_task_support_rxns(task_id=task_id,
                                                                    include_supps=include_supp_rxns)
                                for task_id in activated_tasks])))

    def check_rxn_scales(self, threshold=1e4):
        check_rxn_scales(mod=self._model, threshold=threshold)

    def check_consistency(self, method="fastcc", threshold=1e-6, **kwargs):
        if method == "fastcc":
            cons_obj = FastCCAnalysis(log={"threshold": threshold})
            fastcc_result = fastcc(self._model,
                                   epsilon=threshold,
                                   return_model=True,
                                   return_rxn_ids=True,
                                   return_removed_rxn_ids=True,
                                   **kwargs)
            new_model = self.__class__(model=fastcc_result["model"], name_tag=self._name_tag + "_consist")
            fastcc_result["model"] = new_model
            cons_obj.add_result(result=fastcc_result)
        else:
            raise ValueError()
        return cons_obj

    def do_flux_analysis(self, method, solver="gurobi", **kwargs):
        analyzer = flux_analyzers.create(method, model=self._model, solver=solver, log={"name": self.name_tag})
        return analyzer.analyze(**kwargs)

    def integrate_gene_data(self, data_name, integrator="GIMME", integrator_init_kwargs=None, **kwargs):
        if isinstance(integrator, str):
            integrator_init_kwargs = {} if integrator_init_kwargs is None else integrator_init_kwargs
            integrator = integrator_factory.create(integrator, model=self._model, **integrator_init_kwargs)
        if hasattr(integrator, "apply"):
            integrator.apply(data=self._gene_data[data_name], **kwargs)
        else:
            return integrator.integrate(data=self._gene_data[data_name], **kwargs)

    def get_RAS(self, data_name, method="mean"):
        return self._gene_data[data_name].calc_rxn_score_stat([r.id for r in self._model.reactions])

    def save_model(self, file_name):
        path = Path(file_name)
        save_model(self._model, str(path))


class ReducedModel(Model):
    def __init__(self,
                 name_tag=None,
                 model=None
                 ):
        super().__init__(name_tag=name_tag, model=model)