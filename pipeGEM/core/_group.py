from typing import List, Union, Optional, Tuple, Literal, Dict
from functools import reduce
import itertools
from warnings import warn

import numpy as np
import pandas as pd
import cobra
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

from pipeGEM.core._base import GEMComposite
from pipeGEM.core._model import Model
from pipeGEM.analysis import ComponentComparisonAnalysis, \
    ComponentNumberAnalysis, prepare_PCA_dfs, PCA_Analysis, FBA_Analysis
from pipeGEM.data import GeneData
from pipeGEM.plotting import plot_clustermap
from pipeGEM.utils import is_iter, calc_jaccard_index


class Group(GEMComposite):
    """
    A container for managing and comparing multiple `pipeGEM.Model` objects.

    This class facilitates comparative analyses across a collection of metabolic
    models, such as comparing component numbers, calculating similarity indices
    (e.g., Jaccard), performing dimensionality reduction (PCA), and aggregating
    analysis results.

    Parameters
    ----------
    group : Union[List[Model], Dict[str, cobra.Model], Dict[str, List[Model]], Dict[str, Dict[str, cobra.Model]]]
        The collection of models to include in the group. Can be provided as:
        - A list of `pipeGEM.Model` objects.
        - A dictionary mapping desired name tags to `cobra.Model` objects.
        - A dictionary mapping subgroup names to lists of `pipeGEM.Model` objects.
        - A dictionary mapping subgroup names to dictionaries mapping model names to `cobra.Model` objects.
    name_tag : str, optional
        An identifier for this group. Defaults to "Unnamed_group".
    factors : pd.DataFrame, optional
        A DataFrame providing annotations for the models in the group.
        Index should correspond to model name tags, columns are annotation keys.
    **kwargs
        Additional annotations provided as key-value pairs, where keys are
        annotation names and values are dictionaries mapping model name tags
        to annotation values (e.g., `condition={'model1': 'control', 'model2': 'treated'}`).

    Raises
    ------
    ValueError
        If input models have non-unique name tags or if the input `group`
        format is invalid.
    TypeError
        If elements within the input `group` are not of the expected types.
    KeyError
        If annotation dictionaries or factor DataFrames refer to model names
        not present in the group.
    """
    _is_leaf = False
    agg_methods = ["concat", "mean", "sum", "min", "max", "weighted_mean", "absmin", "absmax"]

    def __init__(self,
                 group,
                 name_tag: str = None,
                 factors: pd.DataFrame = None,
                 **kwargs):

        super().__init__(name_tag=name_tag or "Unnamed_group")
        self._group_annotation = {}
        self._group = self._form_group(group, factors, **kwargs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Group [{self._name_tag}] containing {len(self._group)} models"

    def __len__(self):
        return len(self._group)

    def items(self):
        """Return an iterator over the group's (name_tag, model) items."""
        return self._group.items()

    def __iter__(self):
        """Return an iterator over the models in the group."""
        for k, v in self.items():
            yield v

    def __contains__(self, item):
        """Check if a model name_tag is in the group."""
        return item in self._group

    def __getitem__(self, item):
        """
        Access models within the group by name tag or list of name tags.

        Parameters
        ----------
        item : Union[str, List[str], np.ndarray]
            A single model name tag or a list/array of name tags.

        Returns
        -------
        Union[Model, Group]
            The corresponding `pipeGEM.Model` if `item` is a string, or a new
            `pipeGEM.Group` containing the specified subset of models if `item`
            is a list or array.
        """
        if isinstance(item, str):
            return self._group[item]
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            # Create a new Group with the subset of models and their annotations
            subset_group = [self._group[i] for i in item]
            subset_annotations = {attrn: {modn: attr for modn, attr in v.items() if modn in item}
                                  for attrn, v in self._get_converted_grp_annot().items()}
            return self.__class__(group=subset_group, **subset_annotations)

    def __setitem__(self, key, value):
        """
        Add or replace a model in the group.

        Parameters
        ----------
        key : str
            The name tag to assign to the model.
        value : Union[cobra.Model, Model]
            The model to add. If a `cobra.Model`, it will be wrapped in a
            `pipeGEM.Model`. If a `pipeGEM.Model` or `pipeGEM.Group`, its
            `name_tag` will be updated to `key`.

        Raises
        ------
        TypeError
            If `value` is not a `cobra.Model` or `pipeGEM.GEMComposite`.
        """
        if isinstance(value, cobra.Model):
            self._group[key] = Model(model=value, name_tag=key)
            self._group_annotation[key] = {} # Initialize annotation dict
        elif isinstance(value, GEMComposite):
            value._name_tag = key
            value.rename(name_tag=key) # Ensure name_tag matches the key
            self._group[key] = value
            if key not in self._group_annotation: # Initialize if new
                 self._group_annotation[key] = {}
        else:
            raise TypeError("Inputted value should be a pipeGEM.Model, pipeGEM.Group, or a cobra.model")

    @property
    def annotation(self) -> pd.DataFrame:
        """pd.DataFrame: Combined annotations from models and the group level."""
        return self._handle_annotation(self._group, replacing_annot=self._group_annotation)

    def add_annotation(self, added, store_in_model=False):
        """
        Add annotations to the models in the group.

        Parameters
        ----------
        added : Dict
            A dictionary where keys are annotation names and values are
            dictionaries mapping model name tags to annotation values.
            Example: `{'condition': {'model1': 'A', 'model2': 'B'}}`
        store_in_model : bool, optional
            If True, add annotations directly to the individual `pipeGEM.Model`
            objects. If False (default), store them at the `Group` level.
        """
        self._ck_annotations(group=self._group, new_annot=added, store_in_model=store_in_model)

    def _get_converted_grp_annot(self):
        """Convert group annotation format for internal use."""
        original_annot = {}
        for mod_name, annot in self._group_annotation.items():
            for key, val in annot.items():
                if key not in original_annot:
                    original_annot[key] = {mod_name: val}
                else:
                    original_annot[key].update({mod_name: val})
        return original_annot

    def index(self, item, raise_err=True):
        """
        Get the numerical index of a model within the group's internal order (dict).

        Note: Dictionary order is guaranteed in Python 3.7+.

        Parameters
        ----------
        item : str
            The name tag of the model.
        raise_err : bool, optional
            If True (default), raise KeyError if the item is not found.
            If False, return None.

        Returns
        -------
        int or None
            The index of the model, or None if not found and `raise_err` is False.

        Raises
        ------
        KeyError
            If `item` is not found and `raise_err` is True.
        """
        for i, name_tag in enumerate(self._group.keys()):
            if name_tag == item:
                return i
        if raise_err:
            if raise_err:
                raise KeyError(f"'{item}' is not in this group")
            return None # Return None if not found and raise_err is False

    @property
    def reaction_ids(self) -> List[str]:
        """List[str]: A list of unique reaction IDs across all models in the group."""
        return list(reduce(set.union, [set(g.reaction_ids) for _, g in self._group.items()]))

    @property
    def metabolite_ids(self) -> List[str]:
        """List[str]: A list of unique metabolite IDs across all models in the group."""
        return list(reduce(set.union, [set(g.metabolite_ids) for _, g in self._group.items()]))

    @property
    def gene_ids(self) -> List[str]:
        """List[str]: A list of unique gene IDs across all models in the group."""
        return list(reduce(set.union, [set(g.gene_ids) for _, g in self._group.items()]))

    @property
    def subsystems(self) -> Dict[str, set]:
        """Dict[str, set]: Unique reaction IDs grouped by subsystem across all models."""
        subs = {}
        for _, g in self._group.items():
            for s_name, rxns in g.subsystems.items():
                if s_name in subs:
                    subs[s_name] += rxns
                else:
                    subs[s_name] = rxns
        for g, rxns in subs.items():
            subs[s_name] = set(rxns)
        return subs

    @property
    def gene_data(self) -> GeneData:
        """GeneData: Aggregated gene data from all models in the group."""
        gene_data = {name: m.gene_data for name, m in self._group.items()}
        # Note: Default aggregation might need refinement depending on desired behavior.
        return GeneData.aggregate(gene_data, prop="data")

    def get_RAS(self, data_name, method="mean"):
        """
        Calculate aggregated Reaction Activity Scores (RAS) across the group.

        Parameters
        ----------
        data_name : str
            The name of the gene data set within each model to use.
            (Currently assumes this name exists in all models, might need error handling).
        method : str, optional
            Aggregation method for RAS (e.g., 'mean', 'median'). Default is 'mean'.

        Returns
        -------
        pd.Series or pd.DataFrame
            Aggregated RAS scores. Structure depends on GeneData.aggregate implementation.
        """
        # This assumes each model 'm' has gene_data accessible like this.
        # Might need adjustment based on how gene_data is structured/aggregated.
        gene_data_sets = {name: m.gene_data[data_name] for name, m in self._group.items() if data_name in m.gene_data}
        if not gene_data_sets:
            warn(f"No gene data found with name '{data_name}' in any model of the group.")
            return pd.Series(dtype=float) # Or appropriate empty structure

        # Aggregation logic might need more parameters passed to GeneData.aggregate
        # depending on how aggregation across models vs. within models is handled.
        # The current implementation seems to aggregate *within* each model first,
        # then aggregates across models. Revisit if needed.
        aggregated_data = GeneData.aggregate(gene_data_sets, prop="data") # Aggregates the GeneData objects first

        # Now calculate RAS on the aggregated GeneData object
        # This assumes the aggregated object has a suitable calc_rxn_score_stat method
        # and access to relevant reaction IDs (potentially the union across the group).
        all_rxn_ids = self.reaction_ids
        return aggregated_data.calc_rxn_score_stat(all_rxn_ids, method=method)


    def _get_group_model(self, group_by):
        """Helper to get model names grouped by an annotation."""
        if group_by is None:
            return {"all_models": list(self._group.keys())} # Return all models under a single key

        model_annot = self.annotation # Use the property which handles merging
        if group_by not in model_annot.columns:
             raise KeyError(f"Annotation key '{group_by}' not found for grouping.")
        gb = model_annot.groupby(group_by).apply(lambda x: list(x.index))
        return gb.to_dict() # Returns dict {group_value: [model_name1, ...]}

    def aggregate_models(self, group_by):
        """
        Create new Group objects based on an annotation key.

        Parameters
        ----------
        group_by : str
            The annotation key to group models by.

        Returns
        -------
        Dict[str, Group]
            A dictionary where keys are the unique values of the `group_by`
            annotation, and values are new `Group` objects containing the
            corresponding models. Returns {self.name_tag: self} if group_by is None.
        """
        if group_by is None:
            return {self.name_tag: self}

        model_groups = self._get_group_model(group_by)
        # Create new Group instances for each value in the group_by annotation
        return {gp_name: self[mod_names].rename(gp_name) # Use __getitem__ and rename
                for gp_name, mod_names in model_groups.items()}

    def get_rxn_info(self,
                     models: Optional[Union[str, list]] = None, # Allow selecting models
                     attrs: list = None, # Specify attributes
                     drop_duplicates=True) -> pd.DataFrame:
        """
        Get reaction information across specified models in the group.

        Parameters
        ----------
        models : Union[str, list], optional
            A single model name tag or a list of name tags to include.
            If None (default), includes all models in the group.
        attrs : list, optional
            A list of reaction attributes to retrieve (e.g., ['name', 'subsystem', 'gene_reaction_rule']).
            If None, behavior might depend on `Model.get_rxn_info`.
        drop_duplicates : bool, optional
            If True (default), remove duplicate rows from the combined DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the requested reaction information, indexed by reaction ID (potentially duplicated if drop_duplicates=False).
        """
        if models is None or models == "all":
            selected_models_group = self
        elif isinstance(models, str):
             selected_models_group = [self[models]] # Get single model
        else: # Assume list
            selected_models_group = self[models] # Get subset Group

        if attrs is None:
            # Define default attributes if none are provided, e.g.:
            attrs = ['name', 'reaction', 'gene_reaction_rule', 'subsystem', 'lower_bound', 'upper_bound']
            warn(f"No attributes specified for get_rxn_info, using defaults: {attrs}")


        rxn_info_dfs = [mod.get_rxn_info(attrs) for mod in selected_models_group]
        if not rxn_info_dfs:
            return pd.DataFrame() # Return empty DataFrame if no models selected or no info found

        mg_info = pd.concat(rxn_info_dfs, axis=0)
        if drop_duplicates:
            # Drop duplicates based on index (reaction ID) and all attribute columns
            mg_info = mg_info[~mg_info.index.duplicated(keep='first')] if mg_info.index.has_duplicates else mg_info
            # Alternative: drop based on all columns if index isn't reliable
            # mg_info = mg_info.drop_duplicates()
        return mg_info

    def rename(self, name_tag, inplace=False):
        """
        Rename the group.

        Parameters
        ----------
        name_tag : str
            The new name tag for the group.
        inplace : bool, optional
            If True, modify the current group's name tag directly.
            If False (default), return a new Group instance with the new name.

        Returns
        -------
        Group or None
            The renamed Group instance if `inplace` is False, otherwise None.
        """
        if inplace:
            self._name_tag = name_tag
            return None # Explicitly return None for inplace modification
        else:
            # Create a new group instance with the same models but a new name
            # Pass annotations correctly
            new_group = self.__class__(group=list(self._group.values()), # Pass list of models
                                       name_tag=name_tag,
                                       **self._get_converted_grp_annot()) # Pass existing annotations
            print(f"Created a new group '{name_tag}' containing models: {list(self._group.keys())}")
            return new_group


    def do_flux_analysis(self,
                         method: str,
                         aggregate_method: str = "concat",
                         solver: str = "gurobi",
                         group_by: str = None,
                         **kwargs):
        """
        Do flux analysis on the models contained in this group.

        Parameters
        ----------
        method: str
            Analysis performed on the models.
        aggregate_method: str
            Aggregation method performed on the flux result.
        solver: str
            Solver used to do the analysis.
        group_by: str
            Used to determine the groups for the aggregate_method.
        kwargs: dict
            Keyword arguments used in the model.do_flux_analysis()

        Returns
        -------
        flux_result: FluxAnalysis

        """
        results = []
        for name, c in self._group.items():
            result = c.do_flux_analysis(method=method, solver=solver, **kwargs)
            result.add_categorical(c.name_tag, col_name="model")
            if c.name_tag in self._group_annotation:
                for ftr_name, ftr in self._group_annotation[c.name_tag].items():
                    result.add_categorical(ftr,
                                           col_name=ftr_name)
            results.append(result)
        gp_annot = pd.DataFrame({group_by: self.annotation[group_by].unique()},
                                index=self.annotation[group_by].unique()) if group_by is not None else self.annotation

        agg_result = results[0].__class__.aggregate(results, method=aggregate_method,
                                                    log={"name": self.name_tag,
                                                         "group_by": group_by if group_by is not None else "model",
                                                         "group": self._get_group_model(group_by)},
                                                    group_annotation=gp_annot,
                                                    rxn_annotation=self.get_rxn_info(models="all", attrs=["subsystem"]))

        return agg_result

    @staticmethod
    def _check_rxn_id(comp: GEMComposite, index, subsystems):
        if index is None and subsystems is None:
            return []
        if index is not None:
            return [comp.reaction_ids[r] for r in index]
        if subsystems is not None:
            all = []
            for s in subsystems:
                all += comp.subsystems[s]
            return all

    @staticmethod
    def _handle_annotation(group, included_attrs="n_comp", replacing_annot=None) -> pd.DataFrame:
        annot = {m_name: m.annotation for m_name, m in group.items()}
        if included_attrs == "n_comp":
            included_attrs = ["n_rxns", "n_genes", "n_mets"]
        elif included_attrs is None:
            included_attrs = []
        elif not isinstance(included_attrs, list):
            raise TypeError(f"included_attrs should be either None, a list, or 'n_comp'. "
                            f"Got {type(included_attrs)} instead.")
        if replacing_annot is None:
            replacing_annot = {}

        for ia in included_attrs:
            for m_name, mod in group.items():
                annot[m_name].update({ia: getattr(mod, ia) if hasattr(mod, ia) else None})
        for m, a in replacing_annot.items():
            annot[m].update(a)

        return pd.DataFrame(annot).T

    @staticmethod
    def _form_group_chk_list(group_list, existing_gp=None):
        if not all([isinstance(c, GEMComposite) for c in group_list]):
            raise ValueError("The input list should be a list of pipeGEM.Model, the list contains types: ",
                             set([type(c) for c in group_list]))
        if not len(group_list) == len(set([c.name_tag for c in group_list])):
            raise ValueError("At least two models have non-unique name. "
                             "Please consider rename the input models or drop the models with the same names.")
        if existing_gp is not None:
            if any([c.name_tag in existing_gp for c in group_list]):
                raise ValueError(f"At least two models have non-unique name. "
                                 "Please consider rename the input models or drop the models with the same names.")

    def _ck_annotations(self, group, new_annot, factor_df=None, store_in_model=False):
        if factor_df is not None:
            factors = factor_df.to_dict(orient='dict')
            if not all([mn in group for key, mg in factors.items() for mn in mg]):
                raise KeyError("Some names in the factors are not in the model group",
                               f"{[mn for key, mg in factors.items() for mn in mg if mn not in group]}")
            for key, annot_dic in factors.items():
                for mod_name, val in annot_dic.items():
                    if store_in_model:
                        group[mod_name].add_annotation(key=key, value=val)
                    else:
                        self._group_annotation[mod_name][key] = val

        for key, annot_dic in new_annot.items():
            if all([isinstance(mod_names, list) or isinstance(mod_names, np.ndarray)
                    for val, mod_names in annot_dic.items()]):
                for val, mod_names in annot_dic.items():
                    if not all([mn in group for mn in mod_names]):
                        raise KeyError("Some names in the new annotation dict are not in the model group",
                                       f"{[mn for mn in annot_dic if mn not in group]}")

                for val, mod_names in annot_dic.items():
                    for mod_name in mod_names:
                        if store_in_model:
                            group[mod_name].add_annotation(key=key, value=val)
                        else:
                            self._group_annotation[mod_name][key] = val
            else:
                if not all([mn in group for mn in annot_dic]):
                    raise KeyError("Some names in the new annotation dict are not in the model group:",
                                   f"{[mn for mn in annot_dic if mn not in group]}")
                for mod_name, val in annot_dic.items():
                    if store_in_model:
                        group[mod_name].add_annotation(key=key, value=val)
                    else:
                        self._group_annotation[mod_name][key] = val

    def _form_group(self, group_dict, factors=None, store_in_model=False, **kwargs) -> dict:
        groups = {}
        if isinstance(group_dict, list) or isinstance(group_dict, np.ndarray):
            self._form_group_chk_list(group_dict)
            groups.update({c.name_tag: c for c in group_dict})
            for c_name, c in groups.items():
                self._group_annotation[c_name] = {}
        elif isinstance(group_dict, dict):
            for name, comp in group_dict.items():
                if isinstance(comp, cobra.Model):
                    groups[name] = Model(model=comp, name_tag=name)
                    self._group_annotation[name] = {}
                elif isinstance(comp, Model):
                    groups[name] = comp
                    self._group_annotation[name] = {}
                    if name != comp.name_tag:
                        warn(f"Assigned name ({name}) is not equal to model's name_tag ({comp.name_tag})")
                elif isinstance(comp, list):
                    self._form_group_chk_list(comp, existing_gp=groups)
                    for c in comp:
                        self._group_annotation[c.name_tag] = {"group_name": name}
                        groups[c.name_tag] = c
                elif isinstance(comp, dict):
                    pg_mod_comp = {c_name: Model(model=c, name_tag=c_name) for c_name, c in comp.items()}
                    self._form_group_chk_list([c for _, c in pg_mod_comp.items()], existing_gp=groups)
                    for c_name, c in pg_mod_comp.items():
                        self._group_annotation[c_name] = {"group_name": name}
                        groups[c_name] = c
        elif group_dict is None:
            groups = {}
        else:
            raise ValueError("Group doesn't support object type: ", type(group_dict))

        self._ck_annotations(group=groups,
                             new_annot=kwargs,
                             factor_df=factors,
                             store_in_model=store_in_model)
        return groups

    def get_info(self,
                 models=None,
                 features=None) -> pd.DataFrame:
        """
        Get a information table by traversing the object structure

        Parameters
        ----------
        models: optional, str
            The name tag of the selected models, if None, use all models.
        features: optional, list of str, str
            The features to be obtained while the traverse
        Returns
        -------
        information_table: pd.DataFrame

        """
        if models is None:
            group = self._group
        else:
            group = [m for n, m in self._group.items() if n in models]
        return self._handle_annotation(group,
                                       included_attrs=features,
                                       replacing_annot=self._group_annotation)

    def _compare_components_jaccard(self,
                                    models: 'Group', # Type hint
                                    group_by: Optional[str] = None,
                                    components: Union[str, List[str]] = "all") -> ComponentComparisonAnalysis:
        """Calculate Jaccard similarity between models or aggregated groups."""
        # Aggregate models first if group_by is specified
        if group_by:
            aggregated_groups = models.aggregate_models(group_by=group_by)
            group_list = list(aggregated_groups.values())
            if not group_list:
                 warn(f"No groups formed for group_by='{group_by}'. Cannot calculate Jaccard index.")
                 return ComponentComparisonAnalysis(log={"components": components, "group_by": group_by}) # Return empty result
        else:
            group_list = list(models) # Use iterator directly

        if len(group_list) < 2:
            warn("Need at least two models/groups to compare.")
            return ComponentComparisonAnalysis(log={"components": components, "group_by": group_by}) # Return empty result


        label_index = {grp.name_tag: ind for ind, grp in enumerate(group_list)}
        jaccard_index = {f'{A.name_tag}_to_{B.name_tag}': calc_jaccard_index(A, B, components)
                         for A, B in itertools.combinations(group_list, 2)}
        jaccard_index.update({f'{A.name_tag}_to_{A.name_tag}': 1 for A in group_list})
        group_names = [A.name_tag for A in group_list]
        comp_arr = np.array([[jaccard_index[f'{sorted([A, B], key=lambda x: label_index[x.name_tag])[0].name_tag}'
                                          f'_to_'
                                          f'{sorted([A, B], key=lambda x: label_index[x.name_tag])[1].name_tag}'] # Use name_tag
                              for A in group_list] # Iterate through group_list directly
                             for B in group_list]) # Iterate through group_list directly

        comp_arr = pd.DataFrame(comp_arr, index=[g.name_tag for g in group_list], columns=[g.name_tag for g in group_list])
        result = ComponentComparisonAnalysis(log={"components": components,
                                                  "group_by": group_by,
                                                  "method": "jaccard"}) # Add method to log

        # Get appropriate annotation for the compared entities (original models or aggregated groups)
        if group_by:
            # Create annotation for the aggregated groups
            group_annot_df = pd.DataFrame(index=[g.name_tag for g in group_list])
            # Optionally, try to aggregate original annotations if meaningful
            # Example: take the first model's annotation from each group
            first_model_annots = {}
            original_annot = models.annotation # Get original annotations
            for gp_name, model_list in aggregated_groups.items():
                 first_model_name = model_list[0].name_tag # Assuming model_list is Group of Models
                 if first_model_name in original_annot.index:
                      first_model_annots[gp_name] = original_annot.loc[first_model_name]
            group_annot_df = pd.DataFrame.from_dict(first_model_annots, orient='index')

        else:
            # Use the annotation of the input models
            group_annot_df = models.annotation

        result.add_result({"comparison_df": comp_arr,
                           "group_annotation": group_annot_df
                           })
        return result

    def _compare_component_num(self,
                               models: 'Group', # Type hint
                               group_by: Optional[str] = None,
                               components: Union[str, List[str]] = "all") -> ComponentNumberAnalysis:
        """Compare the number of components (genes, reactions, metabolites) across models."""
        if components == "all":
            components_list = ['genes', 'reactions', 'metabolites']
            features_list = ["n_genes", "n_rxns", "n_mets"]
        elif isinstance(components, str):
            components_list = [components]
            features_map = {"genes": "n_genes", "reactions": "n_rxns", "metabolites": "n_mets"}
            features_list = [features_map.get(components)]
            if features_list[0] is None:
                 raise ValueError(f"Invalid component specified: {components}. Choose from 'genes', 'reactions', 'metabolites'.")
        else: # Assume list
             components_list = components
             features_map = {"genes": "n_genes", "reactions": "n_rxns", "metabolites": "n_mets"}
             features_list = [features_map.get(c) for c in components_list]
             if None in features_list:
                  invalid = [c for c, f in zip(components_list, features_list) if f is None]
                  raise ValueError(f"Invalid component(s) specified: {invalid}. Choose from 'genes', 'reactions', 'metabolites'.")


        # Get component counts and annotations
        comp_df = models.get_info(features=features_list) # Use the property

        # Add group_by column if specified
        id_vars = ["model"] # Start with model name tag
        if group_by:
            if group_by not in comp_df.columns:
                 warn(f"Group_by key '{group_by}' not found in model annotations. Ignoring grouping.")
            else:
                 id_vars.append(group_by)
        comp_df = comp_df.reset_index().rename(columns={"index": "model"}) # Ensure 'model' column exists


        # Melt the DataFrame for plotting/analysis
        melted_dfs = []
        for feature, component_name in zip(features_list, components_list):
             if feature in comp_df.columns:
                  melted_dfs.append(pd.melt(comp_df,
                                            id_vars=id_vars,
                                            value_vars=[feature], # Melt one feature at a time
                                            var_name="component_type_temp", # Temporary column name
                                            value_name="number"))
                  melted_dfs[-1]['component'] = component_name # Assign correct component name
                  melted_dfs[-1].drop(columns=['component_type_temp'], inplace=True) # Drop temp column
             else:
                  warn(f"Feature '{feature}' not found in model info DataFrame. Skipping.")


        if not melted_dfs:
             warn("No component number data could be generated.")
             return ComponentNumberAnalysis(log={"components": components_list, 'group_by': group_by})


        new_comp_df = pd.concat(melted_dfs, ignore_index=True)
        new_comp_df["number"] = pd.to_numeric(new_comp_df["number"], errors='coerce').fillna(0).astype(int) # Ensure integer type

        result = ComponentNumberAnalysis(log={"components": components_list, 'group_by': group_by, "method": "num"})
        result.add_result({"comp_df": new_comp_df})
        return result

    def _compare_component_PCA_helper(self, comp_id_dic, model, comp_name, name):
        """Helper function to collect component IDs for PCA."""
        try:
            ids = getattr(model, comp_name)
            if name not in comp_id_dic:
                comp_id_dic[name] = list(ids) # Ensure it's a list
            else:
                comp_id_dic[name].extend(list(ids)) # Ensure extension is with list
        except AttributeError:
             warn(f"Model '{name}' does not have attribute '{comp_name}'. Skipping for PCA.")


    def _compare_component_PCA(self,
                               models: 'Group', # Type hint
                               group_by: Optional[str] = None,
                               components: Union[str, List[str]] = "all",
                               n_components: int = 2,
                               incremental: bool = False,
                               **kwargs) -> PCA_Analysis:
        """Perform PCA on component presence/absence across models or groups."""
        # Map user input to internal attribute names
        comp_map = {'genes': 'gene_ids', 'reactions': 'reaction_ids', 'metabolites': 'metabolite_ids'}
        if components == "all":
            components_attrs = list(comp_map.values())
        elif isinstance(components, str):
            attr = comp_map.get(components)
            if attr is None:
                 raise ValueError(f"Invalid component specified: {components}. Choose from 'genes', 'reactions', 'metabolites'.")
            components_attrs = [attr]
        else: # Assume list
             components_attrs = [comp_map.get(c) for c in components]
             if None in components_attrs:
                  invalid = [c for c, attr in zip(components, components_attrs) if attr is None]
                  raise ValueError(f"Invalid component(s) specified: {invalid}. Choose from 'genes', 'reactions', 'metabolites'.")


        comp_id_dic = {}
        # Aggregate models first if group_by is specified
        if group_by:
            aggregated_groups = models.aggregate_models(group_by=group_by)
            group_list = list(aggregated_groups.values())
            if not group_list:
                 warn(f"No groups formed for group_by='{group_by}'. Cannot perform PCA.")
                 return PCA_Analysis(log={"group_name_tag": models.name_tag, "dr_method": "PCA", "group_by": group_by})
        else:
            group_list = list(models) # Use iterator directly

        if not group_list:
             warn("No models/groups to perform PCA on.")
             return PCA_Analysis(log={"group_name_tag": models.name_tag, "dr_method": "PCA", "group_by": group_by})


        # Collect component IDs for each model/group
        for c_attr in components_attrs:
            for g in group_list:
                self._compare_component_PCA_helper(comp_id_dic, g, c_attr, g.name_tag)

        if not comp_id_dic:
             warn("No component IDs collected. Cannot perform PCA.")
             return PCA_Analysis(log={"group_name_tag": models.name_tag, "dr_method": "PCA", "group_by": group_by})


        # Create presence/absence matrix
        comp_dfs = {}
        all_unique_ids = set(itertools.chain.from_iterable(comp_id_dic.values()))
        for m_name, comp_ids in comp_id_dic.items():
            # Create a Series indicating presence (1) or absence (0) for all unique IDs
            presence_series = pd.Series(0, index=list(all_unique_ids), dtype=int)
            presence_series.loc[list(set(comp_ids))] = 1 # Use set for efficiency
            comp_dfs[m_name] = presence_series

        component_df = pd.DataFrame(comp_dfs) # Rows are components, columns are models/groups

        # Perform PCA
        try:
            pca_fitted_df, pca_expvar_df, pca_comp_df = prepare_PCA_dfs(component_df, # Transpose: rows=samples (models), cols=features (components)
                                                                        n_components=n_components,
                                                                        incremental=incremental,
                                                                        **kwargs)
        except Exception as e:
             warn(f"PCA calculation failed: {e}")
             return PCA_Analysis(log={"group_name_tag": models.name_tag, "dr_method": "PCA", "group_by": group_by})


        result = PCA_Analysis(log={"group_name_tag": models.name_tag,
                                   "dr_method": "PCA",
                                   "components_used": components,
                                   "group_by": group_by})

        # Get appropriate annotation
        if group_by:
            # Create annotation for the aggregated groups (similar to _compare_components_jaccard)
            group_annot_df = pd.DataFrame(index=[g.name_tag for g in group_list])
            first_model_annots = {}
            original_annot = models.annotation
            for gp_name, model_list in aggregated_groups.items():
                 first_model_name = model_list[0].name_tag
                 if first_model_name in original_annot.index:
                      first_model_annots[gp_name] = original_annot.loc[first_model_name]
            group_annot_df = pd.DataFrame.from_dict(first_model_annots, orient='index')
        else:
            group_annot_df = models.annotation


        result.add_result({"PC": pca_fitted_df,
                           "exp_var": pca_expvar_df,
                           "components": pca_comp_df, # PCA components (loadings)
                           "group_annotation": group_annot_df})
        return result

    def compare(self,
                models: Optional[Union[str, list, np.ndarray]] = None,
                group_by: Optional[str] = None, # Made optional, default handled internally
                method: Literal["jaccard", "PCA", "num"] = "jaccard",
                **kwargs
                ):
        if method not in ["jaccard", "PCA", "num"]:
            raise ValueError("Method should be 'jaccard', 'PCA', or 'num'")

        models: Group = self[models] if models is not None else self
        if method == "jaccard":
            return self._compare_components_jaccard(models=models, group_by=group_by, **kwargs)
        elif method == "num":
            return self._compare_component_num(models=models, group_by=group_by, **kwargs)
        elif method == "PCA":
            return self._compare_component_PCA(models=models, group_by=group_by, **kwargs)
