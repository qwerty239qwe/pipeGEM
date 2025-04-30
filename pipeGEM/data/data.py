import warnings
from typing import Dict, Union, Literal, List, Optional
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pint import UnitRegistry

from ._base import BaseData
from pipeGEM.analysis import RxnMapper
from pipeGEM.analysis import threshold_finders, ALL_THRESHOLD_ANALYSES
from pipeGEM.analysis import DataAggregation

HPA_scores = {'High': 20,
              'Medium': 15,
              'Low': 10,
              'None': -8,
              'Strong': 20,
              'Moderate': 15,
              'Weak': 10,
              'Negative': -8,
              'Not detected': -8}

dis_trans = {"HPA": HPA_scores}


class GeneData(BaseData):
    """
    A GeneData object stores gene data using a dict. It can also calculate rxn scores for a given model.

    Parameters
    ----------
    data: pd.Series, ad.AnnData, or dict
        Data contains pairs of gene IDs and expression values.
    convert_to_str: bool
        Convert the gene names into strings
    expression_threshold: float
        The absent_expression will be assigned to the expression values below this threshold
    absent_expression: float
        The value assigned to the low-expressed genes
    data_transform: callable
        Transformation applied to the rxn_scores and the transformed_gene_data. e.g. np.log2
    discrete_transform: str, dict, or callable
        Discrete data transformation applied to the rxn_scores.
    ordered_thresholds: list
        Ascending thresholds indicating how the gene level will be transformed discretely

    Examples
    ----------
    """

    def __init__(self,
                 data: Union[ad.AnnData, pd.Series, dict],
                 convert_to_str: bool = True,
                 expression_threshold: float = 1e-4,
                 absent_expression: float = 0,
                 data_transform = None,
                 discrete_transform = None,
                 ordered_thresholds: list = None):

        super().__init__("genes")
        self.convert_to_str = convert_to_str
        discrete_transform = self._parse_discrete_transform(discrete_transform, ordered_thresholds)
        self.data_transform = (lambda x: x) if data_transform is None else data_transform
        if isinstance(self.data_transform, str):
            self.data_transform = getattr(np, self.data_transform)

        if isinstance(data, pd.Series):
            data = data.copy().apply(lambda x: discrete_transform(x)) if discrete_transform is not None else data.copy()
            self.gene_data = {str(gene_id) if convert_to_str else gene_id:
                              exp if exp > expression_threshold else absent_expression
                              for gene_id, exp in zip(data.index, data)}
        elif isinstance(data, dict):
            data = {k: discrete_transform(v) if discrete_transform is not None else v for k, v in data.items()}
            self.gene_data = {str(gene_id) if convert_to_str else gene_id:
                              exp if exp > expression_threshold else absent_expression
                              for gene_id, exp in data.items()}
        elif isinstance(data, ad.AnnData):
            if data.obs.shape[0] > 1:
                raise ValueError(f"Input anndata should contain only one observation, found {data.obs.shape[0]} instead.")
            self.gene_data = {str(gene_id) if convert_to_str else gene_id:
                              exp if exp > expression_threshold else absent_expression
                              for gene_id, exp in zip(data.var.index, data[0, :].X.toarray().ravel())}
        else:
            raise ValueError("Expression data should be a dict, an AnnData or a pandas series.")
        self.genes = list(self.gene_data.keys())
        self._digitize_data(ordered_thresholds)
        self.rxn_mapper = None

    def __getitem__(self, item):
        """Retrieve gene expression value by gene ID."""
        return self.gene_data[item]

    def align(self, model, **kwargs):
        """
        Calculate rxn_scores using a metabolic model.

        Parameters
        ----------
        model: pipeGEM.Model or cobra.Model
            The model with the genes and reactions to be mapped onto
        kwargs:
            Keyword arguments used to create a RxnMapper object, including:
            threshold: float or int, default = 0
                The absent_value will be assigned to the rxn_scores below this threshold.
            absent_value: float or int, default = 0
                The value assigned to the reactions with score lower than the threshold.
            missing_value: any, default = np.nan
                The value assigned to the genes not included in the gene_data.
            and_operation: str, default = 'nanmin',
                The operation name used to calculate the 'and' gene-reaction relationships.

                Valid operations include:
                nanmin: return minimum while ignoring all the nan values
                nanmax: return maximum while ignoring all the nan values
                nansum: return the expression sums while ignoring all the nan values
                nanmean: calculate the expression means while ignoring all the nan values
            or_operation: str, default = 'nanmax'
                The operation name used to calculate the 'or' gene-reaction relationships.
            plus_operation: str, default = 'nansum'
                The operation name used to calculate the 'plus' gene-reaction relationships.

        Returns
        -------
        None
        """
        self.rxn_mapper = RxnMapper(self, model, **kwargs)

    def transformed_rxn_scores(self, func) -> dict:
        """
        Get the transformed reaction activity scores.

        Parameters
        ----------
        func: callable
            Function used to transform the reaction score

        Returns
        -------
        transformed_rxn_scores: dict
            A dict contains reaction ids as keys and transformed reaction scores as values.

        """
        return {k: func(v) for k, v in self.rxn_scores.items()}

    @property
    def transformed_gene_data(self) -> Dict[str, float]:
        """
        Gene data after applying the specified `data_transform`.

        Returns
        -------
        dict[str, float]
            Dictionary mapping gene IDs to their transformed expression values.
        """
        return {k: self.data_transform(v) for k, v in self.gene_data.items()}

    @property
    def rxn_scores(self) -> Dict[str, float]:
        """
        Reaction scores calculated by a RxnMapper.
        A RxnMapper assigns a reaction score to each reaction in the aligned model based on its gene-reaction relationship.
        By default, 'or' relationships will be converted into max() formula,
        and 'and' relationships will be converted into min() formula to represent isozymes and protein subunits, respectively.
        However, users can determine which formula to use to replace the relationships.

        Returns
        -------
        rxn_scores: dict[str, float]

        """
        if self.rxn_mapper is None:
            raise AttributeError("The rxn mapper is not initialized. "
                                 "Please call .align(model) first "
                                 "or add this GeneData to a pg.Model using Model.add_gene_data(this object)")
        return {k: self.data_transform(v) for k, v in self.rxn_mapper.rxn_scores.items()}

    def _digitize_data(self, ordered_thresholds):
        """
        Discretizes gene data based on provided ordered thresholds.

        Modifies `self.gene_data` in place. Assigns integer bins centered around zero
        based on where each gene's value falls within the `ordered_thresholds`.

        Parameters
        ----------
        ordered_thresholds : list or None
            A list of ascending threshold values. If None, no digitization occurs.
        """
        if ordered_thresholds is not None:
            n_bins = len(ordered_thresholds)
            ranges = np.array([i for i in range(-((n_bins - 1) // 2), n_bins // 2 + 1)])
            data = np.array(list(self.gene_data.values()))
            disc_data = ranges[np.digitize(data, ordered_thresholds)]
            self.gene_data = dict(zip(self.gene_data.keys(), disc_data))

    @staticmethod
    def _parse_discrete_transform(discrete_transform, ordered_thresholds):
        """
        Parses the discrete_transform input into a callable function.

        Handles string shortcuts (like "HPA"), dictionary mappings, or existing callables.

        Parameters
        ----------
        discrete_transform : str, dict, callable, or None
            The transformation rule to parse.
        ordered_thresholds : list or None
            Thresholds used if digitization is part of the transform (currently not implemented here).

        Returns
        -------
        callable or None
            A function that takes a single value and returns its transformed discrete value,
            or None if no transform is specified.

        Raises
        ------
        ValueError
            If `discrete_transform` is a string but not a recognized key (e.g., "HPA")
            or if it's not a str, dict, callable, or None.
        """
        if isinstance(discrete_transform, str):
            if discrete_transform in dis_trans:
                return lambda x: dis_trans[discrete_transform][x]
        if isinstance(discrete_transform, dict):
            return lambda x: discrete_transform[x]
        if callable(discrete_transform) or discrete_transform is None:
            return discrete_transform
        raise ValueError("Discrete transform is not a valid value, choose a dict or a str as input")

    def calc_rxn_score_stat(self,
                            rxn_ids,
                            ignore_na=True,
                            na_value=0,
                            return_if_all_na=-1,
                            method="mean") -> float:
        """
        Calculate a statistic (mean or median) for reaction scores of specified reactions.

        Parameters
        ----------
        rxn_ids : list or set
            IDs of the reactions to include in the calculation.
        ignore_na : bool, default=True
            If True, ignore NaN scores during calculation.
        na_value : float, default=0
            Value to replace NaN scores with if `ignore_na` is False.
        return_if_all_na : float, default=-1
            Value to return if all selected reaction scores are NaN.
        method : {"mean", "median"}, default="mean"
            The statistic to calculate.

        Returns
        -------
        float
            The calculated statistic.

        Raises
        ------
        ValueError
            If `method` is not "mean" or "median".
        AttributeError
            If reaction scores have not been calculated yet (call `.align()` first).
        """
        scores = np.array([v for k, v in self.rxn_scores.items() if k in rxn_ids])
        if all([np.isnan(i) for i in scores]):
            return return_if_all_na

        if not ignore_na:
            scores[np.isnan(scores)] = na_value
        if method == "mean":
            return np.nanmean(scores)
        if method == "median":
            return np.nanmedian(scores)

        raise ValueError("The method is not supported")

    def apply(self, func):
        """
        Apply a function to each reaction score.

        Parameters
        ----------
        func : callable
            A function that takes a single reaction score (float) as input.

        Returns
        -------
        dict[str, float]
            A dictionary mapping reaction IDs to the results of applying `func`
            to their scores.

        Raises
        ------
        AttributeError
            If reaction scores have not been calculated yet (call `.align()` first).
        """
        return {k: func(v) for k, v in self.rxn_mapper.rxn_scores.items()}

    def get_threshold(self,
                      name: str,
                      transform: bool = True,
                      **kwargs) -> ALL_THRESHOLD_ANALYSES:
        """
        Calculate thresholds for classifying expressed and non-expressed reactions.

        Parameters
        ----------
        name: str
            Thresholding method applied on the gene data.
        transform: bool
            To transform the data before finding thresholds (if True) or not (if False).
        kwargs: dict
            Keyword arguments for applying thresholding methods.

        Returns
        -------

        """
        tf = threshold_finders.create(name)
        print("transform:", transform)
        return tf.find_threshold({gn: self.data_transform(gv) if transform else gv
                                  for gn, gv in self.gene_data.items()}, **kwargs)

    def assign_local_threshold(self,
                               local_threshold_result,
                               transform: bool = True,
                               method: Literal["binary", "ratio", "diff", "rdiff"] = "binary",
                               group: str = None,
                               **kwargs) -> None:
        """
        Assign local threshold result to this object.
        This will change the gene data based on the passed method.
            binary: gene data will be changed to either True or False,
                indicating if the gene data is above (True) or below (False) the threshold.
            ratio: gene data will be changed to a fraction calculated by dividing data by threshold.
            diff: gene data will be changed to data - threshold.
            rdiff: gene data will be changed to threshold - data.

        Parameters
        ----------
        local_threshold_result: LocalThresholdAnalysis
            Assigned threshold object
        transform: bool
            If true, apply data_transform to gene data and thresholds.
        method: str
            Method to assign the new gene data
        group: str
            Group to select from exp_ths's columns of the local_threshold_result.

        Returns
        -------

        """
        assert method in ["binary", "ratio", "diff", "rdiff"]
        group = group if group is not None else 'exp_th'
        gene_exp_ths = local_threshold_result.exp_ths[group]
        data_and_ths = pd.concat([self.data_transform(gene_exp_ths),
                                  pd.DataFrame({"data": {gn: self.data_transform(gv) if transform else gv
                                                         for gn, gv in self.gene_data.items()}})], axis=1)
        if method == "binary":  # returns True / False
            self.gene_data = (data_and_ths["data"] > data_and_ths[group]).astype(int).to_dict()
        elif method == "ratio":  # returns data/threshold
            self.gene_data = (data_and_ths["data"] / data_and_ths[group]).to_dict()
        elif method == "diff":  # returns threshold - data (higher values mean lower expression)
            self.gene_data = (data_and_ths[group] - data_and_ths["data"]).to_dict()
        elif method == "rdiff":  # returns data - threshold (higher values mean higher expression)
            self.gene_data = (data_and_ths["data"] - data_and_ths[group]).to_dict()

    @classmethod
    def aggregate(cls,
                  data: Dict[str, Dict[str, Union[Dict[str, 'GeneData'], 'GeneData']]],
                  method: str = "concat",
                  prop: Literal["data", "score"] = "data",
                  absent_expression: float = 0,
                  group_annotation: pd.DataFrame = None) -> DataAggregation:
        """
        Aggregate data from multiple sources.

        Parameters
        ----------
        data (Dict[str, Dict[str, Union[Dict[str, Any], pd.DataFrame]]]):
            A dictionary containing data to aggregate.
            Outer keys represent different sources, and inner keys represent different datasets within each source.
            Values can either be dictionaries containing 'gene_data' and 'rxn_scores', or Pandas DataFrames.
        method (str, optional):
            The method to use for aggregation. Defaults to "concat".
        prop (Literal["data", "score"], optional):
            The property to aggregate. Should be either "data" or "score". Defaults to "data".
        absent_expression (float, optional):
            Value to fill NaN entries with. Defaults to 0.
        group_annotation (pd.DataFrame, optional):
            DataFrame containing group annotations. Defaults to None.


        Returns
        -------
        aggregated_data (DataAggregation):
            An object containing the aggregated data along with relevant metadata.

        Raises
        -------
        AssertionError: If `prop` is not either "data" or "score".
        ValueError: If `group_annotation` does not match the aggregated data.

        """
        assert prop in ["data", "score"], "prop should be either data or score"
        obj_prop = {"data": "gene_data", "score": "rxn_scores"}

        if all([isinstance(v, dict) for k, v in data.items()]):
            mg_d = pd.concat([pd.DataFrame({name+":"+d_name: getattr(gene_data, obj_prop[prop])})
                              for name, d in data.items() for d_name, gene_data in d.items()], axis=1).fillna(absent_expression)
        else:
            mg_d = pd.concat([pd.DataFrame({name: getattr(gene_data, obj_prop[prop])})
                              for name, gene_data in data.items()], axis=1).fillna(absent_expression)
        if method != "concat":
            mg_d = getattr(mg_d, method)(axis=1).to_frame()
        result = DataAggregation(log={"method": method,
                                      "prop": prop,
                                      "absent_expression": absent_expression,
                                      "group": _data_parse_group_models(data)})
        if group_annotation is not None and (method == "concat") and \
            len(set(group_annotation.index) & set(mg_d.columns)) == 0:
            raise ValueError("Group annotation does not match aggregated data")

        result.add_result(dict(agg_data=mg_d,
                               group_annotation=group_annotation))
        return result


def _data_parse_group_models(data_group) -> dict:
    group_struct = {}
    for k, v in data_group.items():
        group_struct[k] = []
        if isinstance(v, dict):
            for vk, vv in v.items():
                group_struct[k].append(vk)
        else:
            group_struct[k].append(k)
    return group_struct


def find_local_threshold(data_df, **kwargs) -> ALL_THRESHOLD_ANALYSES:
    tf = threshold_finders.create("local")
    return tf.find_threshold(data_df, **kwargs)


class ThermalData(BaseData):
    def __init__(self):
        super().__init__("metabolites")


class MediumData(BaseData):
    """
    Stores and processes medium composition data for constraining metabolic models.

    This class handles loading medium data (metabolite concentrations), aligning
    it with exchange reactions in a metabolic model, and applying these
    concentrations as constraints on reaction bounds.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing medium composition data. Must include columns for
        metabolite IDs and concentrations. Optionally includes metabolite names.
    conc_col_label : str, default="mmol/L"
        The label of the column containing metabolite concentrations in `data`.
    conc_unit : str, default="mmol/L"
        The unit of the concentrations provided in `conc_col_label`. Uses `pint`
        for unit handling.
    id_index : bool, default=False
        If True, assumes the DataFrame index contains the metabolite IDs.
        If False, uses the column specified by `id_col_label`.
    name_index : bool, default=True
        If True, assumes the DataFrame index contains the metabolite names.
        If False, uses the column specified by `name_col_label`.
    id_col_label : str, default="human_1"
        The label of the column containing metabolite IDs, used if `id_index` is False.
    name_col_label : str, optional
        The label of the column containing metabolite names, used if `name_index` is False.
        If None and `name_index` is False, names will not be stored.

    Attributes
    ----------
    data_dict : dict
        Dictionary mapping metabolite IDs to their concentrations.
    rxn_dict : dict
        Dictionary mapping exchange reaction IDs to corresponding metabolite
        concentrations after alignment with a model.
    name_dict : dict
        Dictionary mapping metabolite IDs to their names (if available).
    conc_unit : pint.Quantity
        The concentration unit parsed by `pint`.
    """
    def __init__(self,
                 data,
                 conc_col_label="mmol/L",
                 conc_unit="mmol/L",
                 id_index=False,
                 name_index=True,
                 id_col_label="human_1",
                 name_col_label=None
                 ):
        super().__init__("metabolites")
        # Ensure necessary columns exist if not using index
        if not id_index and id_col_label not in data.columns:
            raise KeyError(f"Metabolite ID column '{id_col_label}' not found in data.")
        if conc_col_label not in data.columns:
             raise KeyError(f"Concentration column '{conc_col_label}' not found in data.")
        if not name_index and name_col_label is not None and name_col_label not in data.columns:
             warnings.warn(f"Metabolite name column '{name_col_label}' not found. Names will not be stored.")
             name_col_label = None # Prevent error later

        self.data_dict = dict(zip(data[id_col_label] if not id_index else data.index, data[conc_col_label]))
        self.rxn_dict = {}
        # Handle name mapping based on index/column presence
        if name_index:
            self.name_dict = dict(zip(data[id_col_label] if not id_index else data.index, data.index))
        elif name_col_label is not None:
             self.name_dict = dict(zip(data[id_col_label] if not id_index else data.index, data[name_col_label]))
        else:
             self.name_dict = {} # Initialize empty if no names provided/found

        self._u = UnitRegistry()
        try:
            self.conc_unit = self._u.Quantity(conc_unit)
        except Exception as e: # Catch potential pint errors
            raise ValueError(f"Invalid concentration unit '{conc_unit}': {e}")


    @staticmethod
    def _find_simple_rxn(rxns):
        """
        Finds the 'simplest' exchange reaction among a list for a given metabolite.

        'Simplest' is defined first by the minimum number of metabolites involved
        (ideally 1 for an exchange reaction), and secondarily by the minimum sum
        of absolute stoichiometric coefficients.

        Parameters
        ----------
        rxns : list of cobra.Reaction
            A list of reactions associated with a metabolite.

        Returns
        -------
        cobra.Reaction
            The identified 'simplest' reaction from the list.

        Raises
        ------
        IndexError
            If the input `rxns` list is empty.
        """
        if not rxns:
            raise IndexError("Input reaction list cannot be empty.")

        cur_simp_ix, best_c, best_nm = -1, float('inf'), float('inf') # Use inf for comparison
        for i, r in enumerate(rxns):
            nm = len(r.metabolites)
            # Ensure coefficients are numeric before summing
            try:
                c = sum([abs(float(coeff)) for coeff in r.metabolites.values()])
            except (ValueError, TypeError):
                 warnings.warn(f"Non-numeric coefficient found in reaction {r.id}. Skipping coefficient sum calculation for this reaction.")
                 c = float('inf') # Penalize reactions with non-numeric coeffs

            # Prioritize fewer metabolites, then smaller total coefficient sum
            if nm < best_nm or (nm == best_nm and c < best_c):
                cur_simp_ix = i
                best_nm = nm
                best_c = c

        if cur_simp_ix == -1:
             # This might happen if all reactions had issues
             raise ValueError("Could not determine the simplest reaction from the provided list.")

        return rxns[cur_simp_ix]

    def align(self,
              model,
              external_comp_name="e",
              met_id_format="{met_id}{comp}",
              raise_err=False):
        """
        Aligns medium metabolite data with exchange reactions in a metabolic model.

        Iterates through the metabolites in `data_dict` and attempts to find
        corresponding exchange reactions in the `model`. Populates `rxn_dict`
        with mappings from reaction IDs to metabolite concentrations.

        Parameters
        ----------
        model : cobra.Model or pipeGEM.Model
            The metabolic model to align against.
        external_comp_name : str, default="e"
            The identifier for the external compartment in the model.
        met_id_format : str, default="{met_id}{comp}"
            A format string to construct the full metabolite ID in the model,
            using the metabolite ID from the data (`met_id`) and the
            `external_comp_name` (`comp`).
        raise_err : bool, default=False
            If True, raises a KeyError if a metabolite from the data cannot be
            found in the model's external compartment. If False, issues a warning.

        Returns
        -------
        None
        """
        # Pre-fetch exchange reaction IDs and metabolite IDs for efficiency
        try:
            exs = {r.id for r in model.exchanges}
            mets_in_model = {m.id for m in model.metabolites}
        except AttributeError as e:
             raise TypeError(f"Input 'model' does not appear to be a valid cobra.Model or pipeGEM.Model object: {e}")


        for mid, conc in self.data_dict.items():
            try:
                met_id = met_id_format.format(met_id=mid, comp=external_comp_name)
            except KeyError:
                 raise ValueError(f"Invalid 'met_id_format' string: '{met_id_format}'. Ensure it contains '{{met_id}}' and '{{comp}}'.")

            if met_id not in mets_in_model:
                msg = f"Metabolite '{met_id}' (from data ID '{mid}') not found in the model."
                if raise_err:
                    raise KeyError(msg)
                warnings.warn(msg + " Ignoring this metabolite.")
                continue

            try:
                m = model.metabolites.get_by_id(met_id)
                # Find reactions associated with the metabolite that are also exchange reactions
                m_related_exchanges = [r for r in m.reactions if r.id in exs]

                if len(m_related_exchanges) == 1:
                    self.rxn_dict[m_related_exchanges[0].id] = conc
                elif len(m_related_exchanges) > 1:
                    # If multiple exchanges, find the simplest one
                    simplest_rxn = self._find_simple_rxn(m_related_exchanges)
                    self.rxn_dict[simplest_rxn.id] = conc
                    warnings.warn(f"Metabolite '{met_id}' associated with multiple exchange reactions ({[r.id for r in m_related_exchanges]}). Using the 'simplest' one: {simplest_rxn.id}")
                else:
                    # No exchange reaction found for this metabolite
                    warnings.warn(f"Metabolite '{met_id}' found in the model but has no associated exchange reaction. Ignoring.")

            except Exception as e: # Catch potential errors during reaction processing
                 warnings.warn(f"Error processing metabolite '{met_id}': {e}. Ignoring.")


    def apply(self,
              model,
              cell_dgw=1e-12,
              n_cells_per_l = 1e9,
              time_hr=96,
              flux_unit="mmol/g/hr",
              threshold=1e-6
              ):
        """
        Applies the medium constraints to the bounds of model reactions.

        Calculates the maximum possible influx rate for each metabolite based on
        its concentration, cell density, dry weight, and time. Updates the
        lower or upper bounds of the corresponding exchange, sink, or demand
        reactions in the model.

        Parameters
        ----------
        model : cobra.Model or pipeGEM.Model
            The metabolic model whose reaction bounds will be modified.
        cell_dgw : float, default=1e-12
            Cell dry weight in grams.
        n_cells_per_l : float, default=1e9
            Number of cells per liter of medium.
        time_hr : float, default=96
            Duration of the experiment or simulation in hours.
        flux_unit : str, default="mmol/g/hr"
            The desired unit for reaction fluxes in the model. The calculated
            influx bounds will be converted to this unit.
        threshold : float, default=1e-6
            A minimum absolute value for the calculated bound. Bounds smaller
            than this threshold will be set to this value (or its negative).
            Helps avoid numerical issues with zero bounds.

        Returns
        -------
        None

        Notes
        -----
        - Modifies the `model` object in place.
        - Assumes exchange reactions consuming the metabolite have negative stoichiometry.
        - Sets bounds for unconstrained inorganic exchanges or sinks/demands to 0
          if they only produce/consume metabolites, respectively. Issues warnings
          for others.
        """
        # Input validation and unit setup
        try:
            cell_dgw_q = cell_dgw * self._u.gram
            n_cells_per_l_q = n_cells_per_l / self._u.liter
            time_hr_q = time_hr * self._u.hour
            target_flux_unit = self._u.Quantity(flux_unit)
        except Exception as e:
            raise ValueError(f"Invalid unit or value provided for constraint calculation: {e}")

        # Identify relevant reactions (exchanges, sinks, demands)
        try:
            relevant_rxns = {r.id: r for r in model.exchanges + model.sinks + model.demands}
        except AttributeError:
             raise TypeError("Input 'model' lacks expected reaction lists (exchanges, sinks, demands).")


        # Apply constraints based on aligned reactions in rxn_dict
        for r_id, conc in self.rxn_dict.items():
            if r_id not in relevant_rxns:
                warnings.warn(f"Reaction ID '{r_id}' from alignment not found among exchanges/sinks/demands in the current model state. Skipping.")
                continue

            r = relevant_rxns[r_id]
            try:
                # Calculate max influx rate based on concentration and parameters
                # Formula: (Concentration [mmol/L]) / (Cells/L * CellWeight [g/cell] * Time [hr])
                # Units: (mmol/L) / (1/L * g * hr) = mmol / (g * hr)
                influx_ub_q = (conc * self.conc_unit / (n_cells_per_l_q * cell_dgw_q * time_hr_q))
                # Convert to the target flux unit
                influx_ub = influx_ub_q.to(target_flux_unit).magnitude
            except Exception as e: # Catch potential unit conversion errors
                warnings.warn(f"Could not calculate or convert flux bound for reaction '{r_id}' with concentration {conc} {self.conc_unit}: {e}. Skipping.")
                continue

            # Determine if the reaction represents uptake (negative net stoichiometry)
            # or secretion/demand (positive net stoichiometry)
            try:
                total_stoich = sum(r.metabolites.values())
            except (TypeError, AttributeError):
                 warnings.warn(f"Could not determine stoichiometry for reaction '{r_id}'. Skipping bound update.")
                 continue

            # Apply the calculated bound, ensuring it meets the threshold
            # For uptake reactions (typically negative total_stoich for exchanges): set lower bound
            # For secretion/demand reactions (typically positive total_stoich): set upper bound
            applied_bound = max(abs(influx_ub), threshold) # Use absolute value for threshold comparison

            current_lb, current_ub = r.bounds
            if total_stoich < 0: # Assume uptake
                new_lb = -applied_bound
                # Only update if the new bound is more restrictive (less negative)
                if new_lb > current_lb:
                     r.lower_bound = new_lb
                # else: keep the potentially more negative original bound
            elif total_stoich > 0: # Assume secretion/demand
                 new_ub = applied_bound
                 # Only update if the new bound is more restrictive (less positive)
                 if new_ub < current_ub:
                     r.upper_bound = new_ub
                 # else: keep the potentially more positive original bound
            else: # total_stoich == 0 (should not happen for exchanges/sinks/demands)
                 warnings.warn(f"Reaction '{r_id}' has zero net stoichiometry. Bounds not modified by medium data.")


        # Handle reactions *not* in rxn_dict (i.e., not in the medium data)
        constrained_rxn_ids = set(self.rxn_dict.keys())
        for r_id, r in relevant_rxns.items():
            if r_id in constrained_rxn_ids:
                continue # Already handled

            # Check if it's an inorganic exchange (often unconstrained)
            # Heuristic: check if *any* metabolite contains 'C' in its formula
            try:
                is_organic = any(hasattr(m, 'formula') and m.formula is not None and 'C' in m.formula.upper()
                                 for m in r.metabolites)
                if not is_organic:
                    # print(f"Reaction '{r_id}' appears inorganic; bounds not modified by medium absence.")
                    pass # Keep original bounds for inorganics
                else:
                    # For organic metabolites not in the medium, constrain uptake/demand
                    total_stoich = sum(r.metabolites.values())
                    if total_stoich < 0: # Uptake reaction
                        if r.lower_bound < 0: # Only constrain if it allows uptake
                            r.lower_bound = 0
                            # print(f"Constraining uptake for '{r_id}' (absent from medium) to 0.")
                    elif total_stoich > 0: # Demand/Secretion reaction
                         if r.upper_bound > 0: # Only constrain if it allows production
                             r.upper_bound = 0
                             # print(f"Constraining demand/secretion for '{r_id}' (absent from medium) to 0.")
                    # else: zero stoichiometry, do nothing

            except Exception as e:
                 warnings.warn(f"Error processing unconstrained reaction '{r_id}': {e}. Bounds not modified.")


    @classmethod
    def from_file(cls, file_name="DMEM", csv_kw=None, **kwargs):
        """
        Loads medium data from a file.

        Supports TSV and CSV formats. Looks for the file in the standard
        `medium/` directory relative to the package structure first. If not
        found there, attempts to load from the provided `file_name` path directly.

        Parameters
        ----------
        file_name : str or Path, default="DMEM"
            The base name of the medium file (e.g., "DMEM", "Hams") or a full
            path to a custom medium file. The method will try appending ".tsv"
            first, then assume CSV if not found or if `csv_kw` is provided.
        csv_kw : dict, optional
            Keyword arguments to pass directly to `pandas.read_csv`. If provided,
            CSV reading is prioritized. Example: `{'sep': ',', 'index_col': 0}`.
        **kwargs :
            Additional keyword arguments passed directly to the `MediumData`
            constructor (`__init__`), such as `conc_col_label`, `id_col_label`, etc.

        Returns
        -------
        MediumData
            An instance of the MediumData class initialized with the loaded data.

        Raises
        ------
        FileNotFoundError
            If the specified file cannot be found either in the default directory
            or at the provided path.
        Exception
            Propagates exceptions from `pandas.read_csv` or `MediumData.__init__`.
        """
        medium_file_dir = Path(__file__).parent.parent.parent / "medium"
        potential_tsv_path = (medium_file_dir / file_name).with_suffix(".tsv")
        potential_csv_path = (medium_file_dir / file_name).with_suffix(".csv") # Also check for .csv in default dir
        direct_path = Path(file_name) # Treat file_name as a potential direct path

        data = None
        used_path = None

        # Prioritize TSV in default directory
        if potential_tsv_path.is_file():
            try:
                data = pd.read_csv(potential_tsv_path, sep='\t') # Assume no index col by default for tsv
                used_path = potential_tsv_path
                # Allow overriding sep/index with csv_kw if explicitly provided for tsv
                if csv_kw:
                     data = pd.read_csv(potential_tsv_path, **csv_kw)
            except Exception as e:
                 raise IOError(f"Error reading TSV file '{potential_tsv_path}': {e}")

        # Else, try CSV in default directory (especially if csv_kw is given)
        elif potential_csv_path.is_file() or csv_kw:
             path_to_try = potential_csv_path if potential_csv_path.is_file() else None
             if path_to_try:
                 try:
                     csv_kw = csv_kw or {} # Ensure csv_kw is a dict
                     data = pd.read_csv(path_to_try, **csv_kw)
                     used_path = path_to_try
                 except Exception as e:
                     raise IOError(f"Error reading CSV file '{path_to_try}': {e}")

        # Else, try the direct path provided in file_name
        elif direct_path.is_file():
             try:
                 csv_kw = csv_kw or {} # Ensure csv_kw is a dict
                 # Determine separator based on extension if not in csv_kw
                 if 'sep' not in csv_kw:
                     if direct_path.suffix.lower() == '.tsv':
                         csv_kw['sep'] = '\t'
                     # else assume comma or let pandas detect
                 data = pd.read_csv(direct_path, **csv_kw)
                 used_path = direct_path
             except Exception as e:
                 raise IOError(f"Error reading file '{direct_path}': {e}")

        # If data is still None, file not found
        if data is None:
            raise FileNotFoundError(f"Medium file '{file_name}' not found in default directory "
                                    f"'{medium_file_dir}' (as .tsv or .csv) or as a direct path.")

        print(f"Loaded medium data from: {used_path}")
        # Pass loaded data and any extra kwargs to constructor
        return cls(data, **kwargs)


class MetaboliteData(BaseData):
    def __init__(self,
                 data: Union[pd.DataFrame],
                 met_id_col: Optional[str] = None,
                 smiles_col="SMILES"):
        super().__init__("metabolites")
        self._add_met_df = data.copy()
        self.smiles_col = smiles_col
        if met_id_col is not None:
            self._add_met_df.index = self._add_met_df[met_id_col]

        if self.smiles_col not in self._add_met_df.columns:
            raise KeyError(f"A column named {self.smiles_col} containing SMILES is required")

    def get_smiles(self, ids):
        if isinstance(ids, str):
            return self._add_met_df.loc[ids, self.smiles_col]
        return self._add_met_df.loc[ids, self.smiles_col].values

    def __getitem__(self, item):
        if isinstance(item, str):
            item = [item]
            smiles = [self.get_smiles(item)]
        else:
            smiles = self.get_smiles(item)
        return dict(zip(item, smiles))


class ProteinAbundanceData(BaseData):
    def __init__(self,
                 data: Union[pd.DataFrame],
                 prot_id_col: Optional[str] = None,
                 abundance_col: str = "abundance",):
        super().__init__("genes")
        self._prot_abund_df = data.copy()
        if prot_id_col is not None:
            self._prot_abund_df.index = self._prot_abund_df[prot_id_col]
        if abundance_col not in self._prot_abund_df:
            raise KeyError(f"abundance_col {abundance_col} cannot be found in the data, "
                           f"possible column names = {self._prot_abund_df.columns}")

    def calc_f_coef(self):
        pass


class EnzymeData(BaseData):
    def __init__(self,
                 data: Union[pd.DataFrame],
                 gene_id_col: Optional[str] = None,
                 prot_id_col: Optional[str] = None,
                 rxn_id_col: Optional[str] = None,
                 met_id_col: Optional[str] = None,
                 mw_col: str = "MW",
                 kcat_col: str = "Kcat",
                 alt_kcat_col: str = "DLKcat",
                 prot_seq_col: str = "Sequence",
                 ec_num_col: str = "EC",
                 sa_col: str = "SA",
                 ) -> None: # Added return type annotation
        super().__init__("genes")
        self._enzyme_df = data.copy()
        if gene_id_col is not None:
            self._enzyme_df.index = self._enzyme_df[gene_id_col]
        else:
            print("Using dataframe's index as the gene ID")

        if prot_id_col is None:
            warnings.warn("No prot_id_col is provided, Gene ID will be used in the following process.")
        self._rxn_id_col = rxn_id_col
        self._met_id_col = met_id_col
        self.prot_id_col = prot_id_col
        self.mw_col = mw_col
        self.kcat_col = kcat_col
        self.alt_kcat_col = alt_kcat_col
        self.prot_seq_col = prot_seq_col

        if self.mw_col not in self._enzyme_df.columns:
            print(f"Inferring molecular weight via protein sequence (column '{prot_seq_col}')")
            self._enzyme_df[self.mw_col] = self._enzyme_df[self.prot_seq_col].apply(self.calc_molecular_weight)
        self.ec_num_col = ec_num_col
        self.sa_col = sa_col
        self._best_matched_df = None
        self._rxn_df_to_be_analyzed = None

    @staticmethod
    def calc_molecular_weight(seq: str) -> float:
        aa_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        aa_MWs = [71.08, 114.60, 103.14, 115.09, 129.11, 147.17, 57.05, 137.14,
                  113.16, 113.16, 128.17, 113.16, 131.20, 114.10, 255.31, 97.12,
                  128.13, 156.19, 87.08, 101.10, 150.04, 99.13, 186.21, 126.50,
                  163.17, 128.62]

        aa_mw_dic = dict(zip(aa_codes, aa_MWs))
        # Handle potential non-standard characters or empty strings gracefully
        try:
            # Add mass of H2O (18.0153 Da) for terminal groups
            return sum(aa_mw_dic[s.upper()] for s in seq if s.upper() in aa_mw_dic) + 18.0153
        except KeyError as e:
            warnings.warn(f"Unknown amino acid code '{e.args[0]}' encountered in sequence. Ignoring.")
            # Recalculate excluding the unknown character
            return sum(aa_mw_dic[s.upper()] for s in seq if s.upper() in aa_mw_dic) + 18.0153
        except TypeError: # Handle case where seq might not be a string
             warnings.warn(f"Input sequence is not a string: {seq}. Returning 0 MW.")
             return 0.0


    def check_gene_rxn_pair(self,
                            ref_model, # Add type hint: Union[cobra.Model, 'pipeGEM.Model'] - requires forward ref or import
                            raise_err: bool = True) -> None:
        """
        Checks if the gene-reaction pairs in the enzyme data exist in the reference model.

        Iterates through the enzyme DataFrame and verifies that for each row,
        the gene (index or `gene_id_col`) is associated with the reaction
        specified in `_rxn_id_col` within the `ref_model`.

        Parameters
        ----------
        ref_model : cobra.Model or pipeGEM.Model
            The metabolic model used as a reference.
        raise_err : bool, default=True
            If True, raises a ValueError upon finding a mismatch.
            If False, issues a warning instead.

        Raises
        ------
        ValueError
            If `raise_err` is True and a mismatch is found.
        AttributeError
            If `_rxn_id_col` is None or not set.
        KeyError
            If a reaction ID from the data is not found in the model.
        """
        if self._rxn_id_col is None:
            raise AttributeError("'_rxn_id_col' must be set before checking gene-reaction pairs.")

        # Create a mapping for faster lookup (handle potential KeyErrors for genes/reactions not in model)
        ref_rxn_gene_map: Dict[str, List[str]] = {}
        for r in ref_model.reactions:
            try:
                ref_rxn_gene_map[r.id] = [g.id for g in r.genes]
            except AttributeError: # Handle reactions without genes
                 ref_rxn_gene_map[r.id] = []


        for gene_id, row in self._enzyme_df.iterrows():
            rxn_id = row[self._rxn_id_col]
            if rxn_id not in ref_rxn_gene_map:
                 err_msg = f"Reaction ID '{rxn_id}' from enzyme data not found in the reference model."
                 if raise_err:
                     raise KeyError(err_msg)
                 else:
                     warnings.warn(err_msg)
                     continue # Skip to next row if reaction not in model

            if gene_id not in ref_rxn_gene_map[rxn_id]:
                err_msg = f"Mismatch found: Gene '{gene_id}' is not associated with reaction '{rxn_id}' in the reference model."
                if raise_err:
                    raise ValueError(err_msg)
                else:
                    warnings.warn(err_msg)

    def rxn_items(self) -> Dict[str, Dict[str, Union[str, float]]]: # More specific return type
        """
        Returns a dictionary mapping reaction IDs to their best-matched enzyme data.

        Requires the `.align()` method to be called first to populate the
        `_best_matched_df`.

        Returns
        -------
        Dict[str, Dict[str, Union[str, float]]]
            A dictionary where keys are reaction IDs and values are dictionaries
            containing 'protein_to_use' (protein ID), 'best_kcat' (kcat value),
            and 'best_mw' (molecular weight).

        Raises
        ------
        AttributeError
            If `.align()` has not been called yet (`_best_matched_df` is None).
        """
        if self._best_matched_df is None:
            raise AttributeError("Please call .align(model) first before getting the rxn_items")

        # Ensure required columns exist in _best_matched_df
        required_cols = ["rxn", "protein", "kcat", "mw"]
        if not all(col in self._best_matched_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in self._best_matched_df.columns]
             raise ValueError(f"Missing required columns in _best_matched_df: {missing}. Alignment might be incomplete.")

        rxn_dic = {row["rxn"]: {"protein_to_use": row["protein"],
                                "best_kcat": row["kcat"],
                                "best_mw": row["mw"]}
                   for i, row in self._best_matched_df.iterrows()}
        # Use .items() directly on the created dictionary
        return rxn_dic # No need to call .items() here, return the dict itself

    def run_DLKcat(self,
                   met_data: MetaboliteData, # Added type hint
                   device: str = "cpu") -> None: # Added return type hint
        """
        Runs the DLKcat tool to predict kcat values.

        (Placeholder - Requires implementation of DLKcat integration)

        Parameters
        ----------
        met_data : MetaboliteData
            Metabolite data object containing SMILES information needed by DLKcat.
        device : str, default="cpu"
            Device to run DLKcat on ('cpu' or 'cuda' if available).

        Notes
        -----
        - This method currently only imports the DLKcat function.
        - Actual prediction logic needs to be implemented.
        - It likely needs to prepare input data (protein sequences, metabolite SMILES)
          from `self._enzyme_df` and `met_data`.
        - Predicted kcat values should probably be stored, potentially in the
          `alt_kcat_col` of `_enzyme_df`.
        """
        try:
            from pipeGEM.extensions.DLKcat import predict_Kcat
            # --- Implementation needed ---
            # 1. Prepare input data for predict_Kcat from self._enzyme_df
            #    (sequences, potentially EC numbers, reaction/metabolite info)
            #    and met_data (SMILES).
            # 2. Call predict_Kcat(prepared_data, device=device)
            # 3. Store results back into self._enzyme_df[self.alt_kcat_col]
            warnings.warn("DLKcat prediction logic is not yet implemented in run_DLKcat.")
            pass
        except ImportError:
            warnings.warn("DLKcat extension not found. Cannot run kcat prediction. "
                          "Please ensure it's installed correctly.")


    def align(self,
              model,
              check_and_raise=True,
              run_DLKcat=True,
              device="cpu"):
        if self._rxn_id_col is not None:
            warnings.warn("Trying to compare the previous rxn ID and the current model's rxn IDs")
            self.check_gene_rxn_pair(ref_model=model,
                                     raise_err=check_and_raise)
            if self._met_id_col is None:
                self._met_id_col = "Metabolite" if "Metabolite" not in self._enzyme_df.columns else "_Metabolite"
                self._enzyme_df[self._met_id_col] = self._enzyme_df[self._rxn_id_col].apply(lambda x:
                                                                                            [m.id for m in
                                                                                             model.reactions.get_by_id(
                                                                                                 x).metabolites])
        elif self._met_id_col is not None:
            raise NotImplementedError()

        else:
            self._rxn_id_col = "Reaction" if "Reaction" not in self._enzyme_df.columns else "_Reaction"
            self._met_id_col = "Metabolite" if "Metabolite" not in self._enzyme_df.columns else "_Metabolite"
            self._enzyme_df[self._rxn_id_col] = self._enzyme_df.index.to_series().apply(lambda x:
                                                                                        [r.id for r in
                                                                                         model.genes.get_by_id(
                                                                                             x).reactions])
            self._enzyme_df[self._met_id_col] = self._enzyme_df[self._rxn_id_col].apply(lambda x:
                                                                                        [m.id for m in
                                                                                         model.reactions.get_by_id(
                                                                                             x).metabolites])
            self._enzyme_df = self._enzyme_df.explode([self._rxn_id_col, self._met_id_col])

        if run_DLKcat:
            if not hasattr(model, 'metabolite_data') or model.metabolite_data is None:
                warnings.warn("Cannot run DLKcat: 'model.metabolite_data' is missing or None. "
                              "Add metabolite data with SMILES using model.add_metabolite_data(). Skipping DLKcat.")
            elif self.prot_seq_col not in self._enzyme_df.columns:
                 warnings.warn(f"Cannot run DLKcat: Protein sequence column '{self.prot_seq_col}' not found in enzyme data. Skipping DLKcat.")
            else:
                # Ensure the alternative kcat column exists
                if self.alt_kcat_col not in self._enzyme_df.columns:
                    self._enzyme_df[self.alt_kcat_col] = np.nan
                self.run_DLKcat(model.metabolite_data, device=device)
