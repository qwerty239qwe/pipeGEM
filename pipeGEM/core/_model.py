from pathlib import Path
from typing import Union, Dict, List, Optional
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
from pipeGEM.data import GeneData, MediumData, EnzymeData, MetaboliteData
from pipeGEM.integration import integrator_factory
from pipeGEM.analysis import flux_analyzers, consistency_testers, TaskAnalysis, ko_analyzers
from pipeGEM.analysis.tasks import TaskHandler, TaskContainer


class Model(GEMComposite):
    """
    A comprehensive container for metabolic models and associated data.

    This class wraps a `cobra.Model` object and extends it with capabilities
    for managing and integrating various types of biological data, including
    gene expression, enzyme kinetics, metabolite concentrations, and medium
    compositions. It also facilitates task-based analysis and model consistency checks.

    Parameters
    ----------
    name_tag : str, optional
        A unique identifier for this model instance, used within `pipeGEM.Group`.
        Defaults to "Unnamed_model".
    model : cobra.Model, optional
        An existing `cobra.Model` to initialize the `pipeGEM.Model` with.
        If None, an empty `cobra.Model` is created.
    gene_data_factor_df : pd.DataFrame, optional
        A DataFrame specifying how different gene datasets should be grouped or
        factored during aggregation (e.g., by condition, time point).
    **kwargs
        Additional key-value pairs to store as annotations for this model.

    Raises
    ------
    ValueError
        If the provided `model` is not a `cobra.Model` instance.
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
        self._enzyme_data: Optional[EnzymeData] = None  # this is a singleton obj in the model
        self._metabolite_data: Optional[MetaboliteData] = None  # this is a singleton obj in the model
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
        """dict: Arbitrary annotations associated with the model."""
        return self._annotations

    def add_annotation(self, key, value):
        """Add or update an annotation."""
        self._annotations[key] = value

    @property
    def size(self):
        """int: The size of the model (always 1 for a single Model)."""
        return 1

    @property
    def reaction_ids(self) -> List[str]:
        """List[str]: A list of all reaction IDs in the model."""
        return [r.id for r in self._model.reactions]

    @property
    def gene_ids(self) -> List[str]:
        """List[str]: A list of all gene IDs in the model."""
        return [g.id for g in self._model.genes]

    @property
    def metabolite_ids(self) -> List[str]:
        """List[str]: A list of all metabolite IDs in the model."""
        return [m.id for m in self._model.metabolites]

    @property
    def cobra_model(self) -> cobra.Model:
        """cobra.Model: The underlying COBRA model object."""
        return self._model

    @property
    def subsystems(self) -> Dict[str, List[str]]:
        """Dict[str, List[str]]: Reactions grouped by subsystem."""
        subs = {}
        for r in self.reactions:
            if r.subsystem in subs:
                subs[r.subsystem].append(r.id)
            else:
                subs[r.subsystem] = [r.id]
        return subs

    def get_rxn_info(self, attrs) -> pd.DataFrame:
        """Get reaction information for specified attributes."""
        return pd.DataFrame([{attr: getattr(r, attr) for attr in attrs}
                             for r in self._model.reactions],
                            index=[r.id for r in self._model.reactions])

    @property
    def gene_data(self) -> Dict[str, GeneData]:
        """Dict[str, GeneData]: Dictionary of associated gene data objects."""
        return self._gene_data

    @property
    def metabolite_data(self) -> Optional[MetaboliteData]:
        """Optional[MetaboliteData]: Associated metabolite data object."""
        return self._metabolite_data

    @property
    def enzyme_data(self) -> Optional[EnzymeData]:
        """Optional[EnzymeData]: Associated enzyme data object."""
        return self._enzyme_data

    @property
    def medium_data(self) -> Dict[str, MediumData]:
        """Dict[str, MediumData]: Dictionary of associated medium data objects."""
        return self._medium_data

    @property
    def tasks(self) -> Dict[str, TaskContainer]:
        """Dict[str, TaskContainer]: Dictionary of associated task containers."""
        return self._tasks

    @property
    def aggregated_gene_data(self):
        """GeneData: Aggregated gene data based on the factor DataFrame."""
        return GeneData.aggregate(self._gene_data, prop="data",
                                  group_annotation=self._gene_data_factor_df)

    def aggregate_gene_data(self, **kwargs):
        """Aggregate gene data using specified parameters."""
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
                        **kwargs) -> None:
        """
        Add medium data to the model.

        Parameters
        ----------
        name : str
            Name to assign to this medium dataset.
        data : Union[MediumData, pd.DataFrame]
            The medium data, either as a MediumData object or a DataFrame.
            If a DataFrame, it will be converted to MediumData.
        data_kwargs : dict, optional
            Keyword arguments for the MediumData constructor if `data` is a DataFrame.
        **kwargs
            Additional keyword arguments passed to the `align` method of MediumData.
        """
        if isinstance(data, pd.DataFrame):
            data_kwargs = {} if data_kwargs is None else data_kwargs
            data = MediumData(data, **data_kwargs)
        elif not isinstance(data, MediumData):
            raise TypeError(f"data should be a MediumData object or a pd.DataFrame, got {type(data)} instead")

        self._medium_data[name] = data
        self._medium_data[name].align(self, **kwargs)

    def apply_medium(self, name, **kwargs):
        """Apply a defined medium composition to the model's exchange reactions."""
        self._medium_data[name].apply(self, **kwargs)

    def add_gene_data(self,
                      name_or_prefix: str,
                      data: Union[GeneData, pd.DataFrame, pd.Series, AnnData],
                      data_kwargs: dict = None,
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
        """Replace an existing gene dataset."""
        self._gene_data.pop(name)
        self.add_gene_data(name, data, data_kwargs, **kwargs)

    def add_tasks(self,
                  name: str,
                  tasks: TaskContainer):
        """Add a metabolic task container."""
        self._tasks[name] = tasks

    def test_tasks(self,
                   name,
                   model_compartment_parenthesis="[{}]",
                   **kwargs):
        """
        Test the model's ability to perform defined metabolic tasks.

        Parameters
        ----------
        name : str
            The name of the TaskContainer to use for testing.
        model_compartment_parenthesis : str, optional
            String format for compartment identifiers in the model, default "[{}]".
        **kwargs
            Additional arguments passed to `TaskHandler.test_tasks`.

        Returns
        -------
        TaskAnalysis
            An object containing the results of the task analysis.
        """
        tester = TaskHandler(model=self._model,
                             tasks_path_or_container=self._tasks[name],
                             model_compartment_parenthesis=model_compartment_parenthesis)
        return tester.test_tasks(**kwargs)

    def calc_ind_task_score(self,
                            data_name: str,
                            task_analysis: TaskAnalysis,
                            all_na_indicator = -1,
                            **kwargs):
        """
        Calculate scores for individual tasks based on associated gene data.

        Parameters
        ----------
        data_name : str
            Name of the GeneData object to use for scoring.
        task_analysis : TaskAnalysis
            The TaskAnalysis result object containing task definitions and supporting reactions.
        all_na_indicator : numeric, optional
            Value to return if all genes associated with a task's reactions have NA scores. Default is -1.
        **kwargs
            Additional arguments passed to `GeneData.calc_rxn_score_stat`.

        Returns
        -------
        dict
            A dictionary mapping task IDs to their calculated scores.
        """
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
        """
        Identify tasks considered 'activated' based on gene data scores and task analysis results.

        Parameters
        ----------
        data_name : str
            Name of the GeneData object to use for scoring.
        task_analysis : TaskAnalysis
            The TaskAnalysis result object.
        all_na_indicator : numeric, optional
            Indicator value used in `calc_ind_task_score`. Default is -1.
        score_threshold : float, optional
            Minimum score for a task to be considered activated. Default is 5*log10(2).
        **kwargs
            Additional arguments passed to `calc_ind_task_score`.

        Returns
        -------
        list
            A list of task IDs considered activated.
        """
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
        """
        Get supporting reactions for tasks identified as 'activated'.

        Parameters
        ----------
        data_name : str
            Name of the GeneData object to use for scoring.
        task_analysis : TaskAnalysis
            The TaskAnalysis result object.
        score_threshold : float, optional
            Minimum score threshold used in `get_activated_tasks`. Default is 5*log10(2).
        include_supp_rxns : bool, optional
            Whether to include supplementary reactions defined in the tasks. Default is True.
        **kwargs
            Additional arguments passed to `get_activated_tasks`.

        Returns
        -------
        list
            A list of unique reaction IDs supporting the activated tasks.
        """
        activated_tasks = self.get_activated_tasks(data_name, task_analysis,
                                                   score_threshold=score_threshold,
                                                   **kwargs)
        print(f"Found {len(activated_tasks)} activated tasks by mapping {data_name} to this model.")
        return list(set(chain(*[task_analysis.get_task_support_rxns(task_id=task_id,
                                                                    include_supps=include_supp_rxns)
                                for task_id in activated_tasks])))

    def check_rxn_scales(self,
                         threshold=1e4):
        """Check if reaction stoichiometric coefficients exceed a threshold."""
        check_rxn_scales(mod=self._model, threshold=threshold)

    def check_model_scale(self,
                          method="geometric_mean",
                          n_iter=10):
        """
        Check the numerical scale of the model's stoichiometric matrix.

        Parameters
        ----------
        method : str, optional
            Scaling method to use ('geometric_mean', etc.). Default is "geometric_mean".
        n_iter : int, optional
            Number of iterations for the scaling algorithm. Default is 10.

        Returns
        -------
        ScalingResult
            An object containing the results of the scaling analysis.
        """
        rescaler = model_scaler_collection[method]()
        return rescaler.rescale_model(model=self, n_iter=n_iter)

    def scale_model(self,
                    scaling_result):
        """
        Apply a previously calculated scaling to the model.

        Parameters
        ----------
        scaling_result : ScalingResult
            The result object obtained from `check_model_scale`.

        Returns
        -------
        Model
            The rescaled pipeGEM Model object.
        """
        scaler_cls_name = {v().__class__.__name__: k for k, v in model_scaler_collection.items()}
        method_name = scaler_cls_name[scaling_result.log["method"]]
        scaler = model_scaler_collection[method_name]()
        return scaler.rescale_with_previous_result(model=self, scaling_result=scaling_result)

    def check_consistency(self,
                          method: str = "FASTCC",
                          tol: float = 1e-6,
                          **kwargs):
        """
        Check the metabolic consistency of the model.

        Parameters
        ----------
        method : str, optional
            Consistency checking algorithm ('FASTCC', etc.). Default is "FASTCC".
        tol : float, optional
            Numerical tolerance for consistency checks. Default is 1e-6.
        **kwargs
            Additional arguments passed to the consistency checker's `analyze` method.

        Returns
        -------
        ConsistencyAnalysis
            An object containing the results of the consistency check, including a consistent sub-model.
        """
        cons_tester = consistency_testers[method](model=self)
        test_result = cons_tester.analyze(tol=tol,
                                          **kwargs)
        test_result.consistent_model.rename(name_tag=f"consistent_{self.name_tag}")
        return test_result

    def do_flux_analysis(self, method, solver="gurobi", **kwargs):
        """
        Perform flux balance analysis (FBA) or its variants.

        Parameters
        ----------
        method : str
            Flux analysis method ('FBA', 'pFBA', 'FVA', etc.).
        solver : str, optional
            LP solver to use ('gurobi', 'cplex', 'glpk', etc.). Default is "gurobi".
        **kwargs
            Additional arguments passed to the flux analyzer's `analyze` method.

        Returns
        -------
        FluxAnalysisResult
            An object containing the results of the flux analysis.
        """
        analyzer = flux_analyzers.create(method, model=self._model, solver=solver, log={"name": self.name_tag})
        return analyzer.analyze(**kwargs)

    def simulate_ko_genes(self, gene_ids, **kwargs):
        """
        Simulate gene knockouts by setting their associated reaction scores to zero.

        Parameters
        ----------
        gene_ids : list
            List of gene IDs to knock out.
        **kwargs
            Additional arguments passed to `GeneData.align`.

        Returns
        -------
        pd.Series
            Reaction scores reflecting the simulated knockouts.
        """
        dummy_data = GeneData({g.id: 1 if g.id not in gene_ids else 0 for g in self._model.genes})
        dummy_data.align(model=self, **kwargs)
        return dummy_data.rxn_scores

    def do_ko_analysis(self, method="single_KO", solver="gurobi", **kwargs):
        """
        Perform gene knockout analysis.

        Parameters
        ----------
        method : str, optional
            Knockout analysis method ('single_KO', etc.). Default is "single_KO".
        solver : str, optional
            LP solver to use. Default is "gurobi".
        **kwargs
            Additional arguments passed to the knockout analyzer's `analyze` method.

        Returns
        -------
        KOAnalysisResult
            An object containing the results of the knockout analysis.
        """
        analyzer = ko_analyzers.create(method, model=self, solver=solver, log={"name": self.name_tag})
        return analyzer.analyze(**kwargs)

    def integrate_enzyme_data(self,
                              prot_abund_data_name,
                              integrator: str = "",
                              integrator_init_kwargs=None,):
        """Integrate enzyme abundance data (Placeholder)."""
        pass

    def integrate_gene_data(self,
                            data_name,
                            integrator="GIMME",
                            integrator_init_kwargs=None,
                            rxn_scaling_coefs=None,
                            predefined_threshold=None,
                            protected_rxns=None,
                            **kwargs):
        """
        Integrate gene data with this model.

        Parameters
        ----------
        data_name: str
            Name of the gene data to be integrated with the model
        integrator: str or Integrator
            Name of the used integrator (algorithm name)
            Possible choices: GIMME, CORDA, rFASTCORMICS, mCADRE, RIPTiDe, and Eflux (for now).
        integrator_init_kwargs: optional, dict
            Keyword arguments for initializing the integrator
        rxn_scaling_coefs: optional, dict
            Reaction scaling coefficient for the integrator if the model was rescaled before.
        predefined_threshold: ThresholdAnalysis or a dict
            Threshold analysis object contains expression threshold needed,
            or a dict contains an expression threshold with a key named exp_th
            and a non-expression threshold with a key named non_exp_th
        protected_rxns: list of str
            Protected reaction IDs contained in a list

        kwargs: dict
            Keyword arguments for integrating the data.

        Returns
        -------
        integrating_result: BaseAnalysis
            Result object containing gene data-integrated model (context-specific model).
        """
        if isinstance(integrator, str):
            integrator_init_kwargs = {} if integrator_init_kwargs is None else integrator_init_kwargs
            integrator = integrator_factory.create(integrator, **integrator_init_kwargs)
        return integrator.integrate(data=self._gene_data[data_name],
                                    model=self._model,
                                    rxn_scaling_coefs=rxn_scaling_coefs,
                                    predefined_threshold=predefined_threshold,
                                    protected_rxns=protected_rxns,
                                    **kwargs)

    def get_RAS(self, data_name, method="mean"):
        return self._gene_data[data_name].calc_rxn_score_stat([r.id for r in self._model.reactions],
                                                              method=method)

    def save_model(self,
                   file_name: str) -> None:
        """
        Save the pipeGEM model and its annotations.

        Saves the underlying `cobra.Model` to the specified `file_name` (e.g.,
        'model.json', 'model.xml'). Additionally, saves model annotations
        (including `name_tag`) to a corresponding TOML file (e.g.,
        'model_annotations.toml') in the same directory.
        This is just a workaround for now
        since the io function for all the file types haven't been implemented.
        Besides the model, this function stores annotations and name_tag as a toml file in the same folder of the model.

        Parameters
        ----------
        file_name: str

        Returns
        -------
        None

        """
        path = Path(file_name)
        save_model(self._model, str(path))
        additional = self.annotation
        additional.update({"name_tag": self.name_tag})
        save_toml_file(file_name=path.with_name(f"{path.stem}_annotations.toml"),
                       dic=additional)

    @classmethod
    def load_model(cls,
                   file_name: str):
        """
        Load a pipeGEM model from a model file (json, sbml, mat..) and a toml file storing the metadata of the model

        Parameters
        ----------
        file_name: str
            Model file name.
            In the same directory, here should be a toml file having the same file name and a .toml suffix
            For example, a valid model file called 'model.json' is stored in a folder called 'folder'.
            Then the files in the folder should be:
            folder
            |- model.json
            |- model.toml
            ...
        Returns
        -------
        model: pipeGEM.Model
        """
        model_pth = Path(file_name)
        add_ = parse_toml_file(model_pth.with_name(f"{model_pth.stem}_annotations.toml"))
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
        self._merged_rxn_lu_table = {k: v for k, v in self._merged_rxn_lu_table.items()
                                     if k not in [r.id for r in to_be_restored]}
        self._empty_merged_rxns = [] # Clear empty merged reactions after separation
        self._original_objs = {} # Clear original objectives after separation

    def update_merged_rxn(self, merged_rxn):
        """
        Update internal state when a reaction is merged.

        Stores the original objective coefficients if not already done,
        adds the merged reaction to the lookup table, and handles empty merged reactions.

        Parameters
        ----------
        merged_rxn : cobra.Reaction
            The reaction object representing the merged reaction. It should have
            a `merged_rxns` attribute (dict mapping original reactions to coefficients).
        """
        if len(self._original_objs) == 0:
            self._original_objs = linear_reaction_coefficients(self._model)

        if len(merged_rxn.metabolites) == 0:
            self._empty_merged_rxns.append(merged_rxn)

        for r, c in merged_rxn.merged_rxns.items():
            self._merged_rxn_lu_table[r.id] = merged_rxn
            if r in self._original_objs:
                # Adjust objective coefficient based on original reaction's coefficient and merging factor
                merged_rxn.objective_coefficient += self._original_objs[r] / c # Use += in case multiple original objectives map here

    def get_merged_rxn(self, rxn_id):
        """
        Retrieve the merged reaction object corresponding to an original reaction ID.

        Parameters
        ----------
        rxn_id : str
            The ID of the original reaction before merging.

        Returns
        -------
        cobra.Reaction or None
            The merged reaction object if the original reaction was merged, otherwise None.
        """
        return self._merged_rxn_lu_table.get(rxn_id)
