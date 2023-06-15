from pathlib import Path
from typing import Union, Dict, List
from copy import deepcopy, copy
from itertools import chain


import pandas as pd
import cobra
from cobra.util import linear_reaction_coefficients
import numpy as np
from anndata import AnnData

from pipeGEM.core._base import GEMComposite
from pipeGEM.utils import save_model, load_model, check_rxn_scales, save_toml_file, parse_toml_file
from pipeGEM.analysis import model_scaler_collection
from pipeGEM.data import GeneData, MediumData
from pipeGEM.integration import integrator_factory
from pipeGEM.analysis import flux_analyzers, consistency_testers, TaskAnalysis, ko_analyzers
from pipeGEM.analysis.tasks import TaskHandler, TaskContainer


class Model(GEMComposite):
    """
    Main model used to store cobra.Model, tasks and omics data

    Parameters
    ----------
    name_tag: optional, str
        The name of this object, it will be used in a pg.Group object.
        If None, the model will be named 'Unnamed_model'
    model: optional, cobra.Model
        A cobra model analyzed in this object

    """
    def __init__(self,
                 name_tag: str = None,
                 model = None,
                 gene_data_factor_df: pd.DataFrame = None,
                 **kwargs):
        super(Model, self).__init__(name_tag=name_tag or "Unnamed_model")
        if not isinstance(model, cobra.Model):
            raise ValueError("input model should be a cobra model")
        self._model = model if model is not None else cobra.Model(name=name_tag)
        self._gene_data: Union[Dict[str, GeneData]] = {}
        self._medium_data = {}
        self._tasks = {}
        self._merged_rxn_lu_table = {}
        self._original_objs = {}
        self._empty_merged_rxns = []
        self._annotations = kwargs
        self._gene_data_factor_df = gene_data_factor_df

    def __enter__(self):
        return self._model.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self):
        return f"pipeGEM.Model [{self._name_tag}] (g,m,r)=({self.n_genes}, {self.n_mets}, {self.n_rxns})\n"

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, item):
        if item not in self.__dict__:
            return getattr(self._model, item)
        return self.__dict__[item]

    def rename(self, name_tag=None):
        if name_tag is not None:
            if not isinstance(name_tag, str):
                TypeError("name_tag must be a string.")
            self._name_tag = name_tag

    @property
    def annotation(self) -> dict:
        return self._annotations

    def add_annotation(self, key, value):
        self._annotations[key] = value

    @property
    def size(self):
        return 1

    @property
    def reaction_ids(self) -> List[str]:
        return [r.id for r in self._model.reactions]

    @property
    def gene_ids(self) -> List[str]:
        return [g.id for g in self._model.genes]

    @property
    def metabolite_ids(self) -> List[str]:
        return [m.id for m in self._model.metabolites]

    @property
    def cobra_model(self) -> cobra.Model:
        return self._model

    @property
    def subsystems(self) -> Dict[str, List[str]]:
        subs = {}
        for r in self.reactions:
            if r.subsystem in subs:
                subs[r.subsystem].append(r.id)
            else:
                subs[r.subsystem] = [r.id]
        return subs

    def get_rxn_info(self, attrs):
        return pd.DataFrame([{attr: getattr(r, attr) for attr in attrs}
                             for r in self._model.reactions],
                            index=[r.id for r in self._model.reactions])

    @property
    def gene_data(self) -> Dict[str, GeneData]:
        return self._gene_data

    @property
    def medium_data(self) -> Dict[str, MediumData]:
        return self._medium_data

    @property
    def tasks(self) -> Dict[str, TaskContainer]:
        return self._tasks

    @property
    def aggregated_gene_data(self):
        return GeneData.aggregate(self._gene_data, prop="data",
                                  group_annotation=self._gene_data_factor_df)

    def aggregate_gene_data(self, **kwargs):
        return GeneData.aggregate(data=self._gene_data,
                                  group_annotation=self._gene_data_factor_df,
                                  **kwargs)

    def copy(self,
             copy_gene_data=False,
             copy_medium_data=False,
             copy_tasks=False,
             copy_merging_info=True):
        """
        Create a deep-copied object of this Model

        Parameters
        ----------
        copy_gene_data: bool
            Also copy the gene data in this Model
        copy_medium_data: bool
            Also copy the medium data in this Model
        copy_tasks: bool
            Also copy the tasks in this Model
        copy_merging_info: bool
            Also copy the merged reaction information
        Returns
        -------
        copied_model: pipeGEM.Model

        """
        new = self.__class__(model=self._model.copy(),
                             name_tag=f"copied_{self.name_tag}")
        if copy_gene_data:
            new._gene_data = self._gene_data.copy()
        if copy_medium_data:
            new._medium_data = self._medium_data.copy()
        if copy_tasks:
            new._tasks = self._tasks.copy()
        if copy_merging_info:
            new._merged_rxn_lu_table = self._merged_rxn_lu_table.copy()
            new._original_objs = self._original_objs.copy()

        return new

    def add_medium_data(self,
                        name,
                        data: Union[MediumData, pd.DataFrame],
                        data_kwargs=None,
                        **kwargs):
        if isinstance(data, pd.DataFrame):
            data_kwargs = {} if data_kwargs is None else data_kwargs
            data = MediumData(data, **data_kwargs)
        elif not isinstance(data, MediumData):
            raise TypeError(f"data should be a MediumData object or a pd.DataFrame, got {type(data)} instead")

        self._medium_data[name] = data
        self._medium_data[name].align(self, **kwargs)

    def apply_medium(self, name, **kwargs):
        self._medium_data[name].apply(self, **kwargs)

    def add_gene_data(self,
                      name_or_prefix: str,
                      data: Union[GeneData, pd.DataFrame, pd.Series, AnnData],
                      data_kwargs=None,
                      **kwargs) -> None:
        """
        Add gene data to the internal dictionary of gene data in MyClass.

        Parameters
        ----------
        name_or_prefix : str
            The name or prefix of the gene data. If a prefix is provided, then
            the actual column names in the pd.DataFrame will be suffixed with
            the prefix. If an empty string is provided, then the column names
            will not be modified.
        data : Union[GeneData, pd.DataFrame, pd.Series, anndata.AnnData]
            The gene data to add to the internal dictionary. This can be a
            pd.DataFrame, pd.Series, anndata.AnnData, or GeneData object.
            If a pd.DataFrame is provided, then each column of the DataFrame will be converted into
            a GeneData object with a modified name based on the name_or_prefix
            argument. If a pd.Series is provided, then it will be converted into
            a GeneData object with the name provided by name_or_prefix. If a
            GeneData object is provided, then it will be added to the internal
            dictionary as-is.
        data_kwargs : dict, optional
            Additional keyword arguments to pass to the GeneData constructor
            when converting a pd.DataFrame or pd.Series into GeneData objects.
            The default value is None, which means no additional arguments are
            passed to the GeneData constructor.
            Ignored when the input data is already a GeneData.
        **kwargs
            Additional keyword arguments to pass to the align method of the
            GeneData object(s) after they have been added to the internal
            dictionary.

        Raises
        ------
        ValueError
            If the data argument is not a pd.DataFrame, pd.Series, anndata.AnnData, or GeneData
            object.

        """
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
        elif isinstance(data, AnnData):
            data_dict = {}
            for obs in data.obs.index:
                cell_data = GeneData(data[obs, :],
                                     **data_kwargs)
                data_dict[obs] = cell_data
            self._gene_data.update(data_dict)
            for k, v in data_dict.items():
                v.align(self, **kwargs)
            return
        elif not isinstance(data, GeneData):
            raise ValueError("The 'data' argument should be a pd.DataFrame, pd.Series, or GeneData object.")

        self._gene_data[name_or_prefix] = data
        self._gene_data[name_or_prefix].align(self._model, **kwargs)

    def set_gene_data(self, name, data, data_kwargs=None, **kwargs):
        self._gene_data.pop(name)
        self.add_gene_data(name, data, data_kwargs, **kwargs)

    def add_tasks(self,
                  name: str,
                  tasks: TaskContainer):
        self._tasks[name] = tasks

    def test_tasks(self,
                   name,
                   model_compartment_parenthesis="[{}]",
                   **kwargs):
        tester = TaskHandler(model=self._model,
                             tasks_path_or_container=self._tasks[name],
                             model_compartment_parenthesis=model_compartment_parenthesis)
        return tester.test_tasks(**kwargs)

    def calc_ind_task_score(self,
                            data_name: str,
                            task_analysis: TaskAnalysis,
                            all_na_indicator = -1,
                            **kwargs):

        return {task_id: self._gene_data[data_name].calc_rxn_score_stat(rxn_ids=rxns,
                                                                        return_if_all_na=all_na_indicator,
                                                                        **kwargs)
                for task_id, rxns in task_analysis.task_support_rxns.items()}

    def get_activated_tasks(self,
                            data_name,
                            task_analysis: TaskAnalysis,
                            all_na_indicator=-1,
                            score_threshold=5*np.log10(2),
                            **kwargs):
        passed_task = task_analysis.result_df.query("Passed").index.to_list()
        ind_task_scores = self.calc_ind_task_score(data_name,
                                                   task_analysis,
                                                   all_na_indicator=all_na_indicator,
                                                   **kwargs)
        return [k for k, v in ind_task_scores.items()
                if (v >= score_threshold or v == all_na_indicator) and (k in passed_task)]

    def get_activated_task_sup_rxns(self,
                                    data_name: str,
                                    task_analysis: TaskAnalysis,
                                    score_threshold: float = 5*np.log10(2),
                                    include_supp_rxns: bool = True,
                                    **kwargs):
        activated_tasks = self.get_activated_tasks(data_name, task_analysis,
                                                   score_threshold=score_threshold,
                                                   **kwargs)
        return list(set(chain(*[task_analysis.get_task_support_rxns(task_id=task_id,
                                                                    include_supps=include_supp_rxns)
                                for task_id in activated_tasks])))

    def check_rxn_scales(self,
                         threshold=1e4):
        check_rxn_scales(mod=self._model, threshold=threshold)

    def check_model_scale(self,
                          method="geometric_mean",
                          n_iter=10):
        rescaler = model_scaler_collection[method]()
        return rescaler.rescale_model(model=self, n_iter=n_iter)

    def check_consistency(self,
                          method: str = "FASTCC",
                          tol: float = 1e-6,
                          **kwargs):
        cons_tester = consistency_testers[method](model=self)
        test_result = cons_tester.analyze(tol=tol,
                                          **kwargs)
        test_result.consistent_model.rename(name_tag=f"consistent_{self.name_tag}")
        return test_result

    def do_flux_analysis(self, method, solver="gurobi", **kwargs):
        analyzer = flux_analyzers.create(method, model=self._model, solver=solver, log={"name": self.name_tag})
        return analyzer.analyze(**kwargs)

    def simulate_ko_genes(self, gene_ids, **kwargs):
        dummy_data = GeneData({g.id: 1 if g.id not in gene_ids else 0 for g in self._model.genes})
        dummy_data.align(model=self, **kwargs)
        return dummy_data.rxn_scores

    def do_ko_analysis(self, method="single_KO", solver="gurobi", **kwargs):
        analyzer = ko_analyzers.create(method, model=self, solver=solver, log={"name": self.name_tag})
        return analyzer.analyze(**kwargs)

    def integrate_gene_data(self,
                            data_name,
                            integrator="GIMME",
                            integrator_init_kwargs=None,
                            **kwargs):
        if isinstance(integrator, str):
            integrator_init_kwargs = {} if integrator_init_kwargs is None else integrator_init_kwargs
            integrator = integrator_factory.create(integrator, **integrator_init_kwargs)
        return integrator.integrate(data=self._gene_data[data_name],
                                    model=self._model,
                                    **kwargs)

    def get_RAS(self, data_name, method="mean"):
        return self._gene_data[data_name].calc_rxn_score_stat([r.id for r in self._model.reactions])

    def save_model(self,
                   file_name: str) -> None:
        """
        Save this model at the provided location.
        This is just a workaround for now
        since the io function for all the file types haven't been implemented.
        Besides the model, this function stores annotations and name_tag as a toml file in the same folder of the model.

        Parameters
        ----------
        file_name: str

        Returns
        -------

        """
        path = Path(file_name)
        save_model(self._model, str(path))
        additional = self.annotation
        additional.update({"name_tag": self.name_tag})
        save_toml_file(file_name=Path(path.stem).with_suffix(".toml"),
                       dic=additional)

    @classmethod
    def load_model(cls, file_name):
        model_pth = Path(file_name)
        add_ = parse_toml_file(Path(model_pth.stem).with_suffix(".toml"))
        model = load_model(file_name)
        name = add_.pop("name_tag")
        return cls(name_tag=name, model=model, **add_)

    def separate_merged_rxns(self):
        to_be_restored, to_be_pruned = [], []
        for r in self._model.reactions + self._empty_merged_rxns:
            if hasattr(r, "merged_rxns"):
                to_be_restored.extend(list(r.merged_rxns.keys()))
                to_be_pruned.append(r)
        if len(to_be_restored) == len(to_be_pruned) == 0:
            print("No merged rxns found in this model")
            return
        self._model.add_reactions(to_be_restored)
        self._model.objective = self._original_objs
        self._model.remove_reactions(to_be_pruned, remove_orphans=True)

    def update_merged_rxn(self, merged_rxn):
        if len(self._original_objs) == 0:
            self._original_objs = linear_reaction_coefficients(self._model)

        if len(merged_rxn.metabolites) == 0:
            self._empty_merged_rxns.append(merged_rxn)

        for r, c in merged_rxn.merged_rxns.items():
            self._merged_rxn_lu_table[r.id] = merged_rxn
            if r in self._original_objs:
                merged_rxn.objective_coefficient = self._original_objs[r] / c

    def get_merged_rxn(self, rxn_id):
        return self._merged_rxn_lu_table[rxn_id]

