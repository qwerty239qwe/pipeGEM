import pandas as pd
import numpy as np
from ._base import *


def measure_efficacy(kept_rxn_ids,
                     removed_rxn_ids,
                     core_rxn_ids,
                     non_core_rxn_ids,
                     method="F1_score"):
    FP = len(set(kept_rxn_ids) & set(non_core_rxn_ids))
    TP = len(set(kept_rxn_ids) & set(core_rxn_ids))
    TN = len(set(removed_rxn_ids) & set(non_core_rxn_ids))
    FN = len(set(removed_rxn_ids) & set(core_rxn_ids))
    print("# Kept core reactions:", TP)
    print("# Removed core reactions:", FN)
    print("# Kept non-core reactions:", FP)
    print("# Removed non-core reactions:", TN)
    print("Percentage of kept core rxns:", TP / (TP+FN))
    print("Percentage of removed non-core rxns:", TN / (TN + FP))
    if method == "MCC":
        return (TN * TP - FN * FP) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if method == "F1_score":
        return 2 * (precision * recall) / (precision + recall)
    if method == "precision":
        return precision
    if method == "recall":
        return recall


class EFluxAnalysis(BaseAnalysis):
    """
    An object containing EFlux result.
    This should contain results including:
        rxn_bounds: dict[str, tuple[float, float]]
        rxn_scores: dict[str, float]
        flux_result: pd.DataFrame, optional
    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """
    def __init__(self, log):

        super().__init__(log)


class GIMMEAnalysis(BaseAnalysis):
    """
    An object containing GIMME result.
    This should contain results including:
        rxn_coefficents: dict[str, float]
        rxn_scores: dict[str, float]
        flux_result: pd.DataFrame, optional
        result_model: pd.Model or cobra.Model, optional
    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """
    def __init__(self, log):

        super().__init__(log)


# Not finished
class SPOTAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._rxn_scores = None
        self._fluxes = None
        self._model = None

    @property
    def result_model(self):
        return self._model

    @property
    def flux_result(self):
        return self._fluxes

    def add_result(self, rxn_scores, fluxes=None, model=None):
        self._rxn_scores = rxn_scores
        self._fluxes = fluxes if fluxes is not None else self._fluxes
        self._model = model


class RIPTiDePruningAnalysis(BaseAnalysis):
    """Analysis result object for the RIPTiDe pruning step.

    Encapsulates the results from the `apply_RIPTiDe_pruning` function,
    including the pruned model and the objective weights used.

    Parameters
    ----------
    log : dict
        A dictionary storing parameters used during the RIPTiDe pruning
        execution, such as `max_gw`, `obj_frac`, and `threshold`.

    Attributes
    ----------
    result_model : cobra.Model
        The pruned context-specific metabolic model resulting from the
        pFBA-based removal of low-flux reactions.
    removed_rxn_ids : list[str]
        A list of string IDs for the reactions that were removed from the
        original model during the pruning process.
    obj_dict : dict[str, float]
        A dictionary mapping reaction IDs to the calculated objective weights
        used in the parsimonious FBA (pFBA) step. Weights are derived from
        reaction expression scores (RALs).
    """
    def __init__(self, log):
        super().__init__(log)
        # Note: removed_rxn_ids is stored as a list of strings here,
        # unlike CORDA which stores Reaction objects.
        # No specific file manager needed if saving as a simple list.


class RIPTiDeSamplingAnalysis(BaseAnalysis):
    """Analysis result object for the RIPTiDe sampling step.

    Encapsulates the results from the `apply_RIPTiDe_sampling` function,
    primarily the flux sampling data if generated.

    Parameters
    ----------
    log : dict
        A dictionary storing parameters used during the RIPTiDe sampling
        execution, such as `max_gw`, `obj_frac`, `sampling_obj_frac`,
        `sampling_method`, etc.

    Attributes
    ----------
    sampling_result : SamplingAnalysis or None
        An object containing the results of the flux sampling process (e.g.,
        flux distributions stored in `sampling_result.flux_df`). This is `None`
        if `do_sampling` was set to `False` in the `apply_RIPTiDe_sampling` call.
    flux_result : pandas.DataFrame or None
        Property providing direct access to the flux sampling dataframe stored
        within `sampling_result`. Returns `None` if no sampling was performed.
    """
    def __init__(self, log):
        super().__init__(log)

    @property
    def flux_result(self):
        """Flux sampling results as a pandas DataFrame, if available."""
        sampling_res = self._result.get("sampling_result")
        return sampling_res.flux_df if sampling_res else None


class FASTCOREAnalysis(BaseAnalysis):
    """Analysis result object for the FASTCORE algorithm.

    Encapsulates the results from the `apply_FASTCORE` function, which extracts
    a flux-consistent subnetwork from a larger metabolic model based on core
    and non-penalty reaction sets.

    Parameters
    ----------
    log : dict
        A dictionary storing parameters used during the FASTCORE execution,
        notably the flux tolerance `epsilon` (which might be a single float or
        a pandas Series if `rxn_scaling_coefs` were used).

    Attributes
    ----------
    result_model : cobra.Model or None
        The extracted, flux-consistent subnetwork as a `cobra.Model` object.
        This attribute is `None` if `return_model` was set to `False` during
        the `apply_FASTCORE` call.
    removed_rxn_ids : numpy.ndarray
        A NumPy array containing the string IDs of reactions that were present
        in the original model but removed to create the `result_model`.
    kept_rxn_ids : numpy.ndarray
        A NumPy array containing the string IDs of reactions from the original
        model that were retained in the final `result_model`.
    algo_efficacy : dict or None
        A dictionary containing efficacy metrics (e.g., 'precision', 'recall',
        'F1_score', 'MCC') evaluating the performance of the algorithm by
        comparing the `kept_rxn_ids` and `removed_rxn_ids` against the
        initial core (C) and non-core sets (derived from reactions not in C
        or nonP). This attribute is `None` if `calc_efficacy` was `False`
        during the `apply_FASTCORE` call.
    """
    def __init__(self, log):
        super().__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}
        self._result_saving_params["kept_rxn_ids"] = {"fm_name": "NDArrayStr"}


class rFASTCORMICSAnalysis(BaseAnalysis):
    """Analysis result object for the rFASTCORMICS algorithm.

    Stores the outputs generated by the `apply_rFASTCORMICS` function.

    Parameters
    ----------
    log : dict
        Dictionary storing parameters used to perform this analysis.

    Attributes
    ----------
    fastcore_result : FASTCOREAnalysis
        The result object from the underlying FASTCORE run. Contains:
        - result_model (cobra.Model): The final context-specific model.
        - removed_rxn_ids (np.ndarray): IDs of reactions removed.
        - kept_rxn_ids (np.ndarray): IDs of reactions kept.
    threshold_analysis : rFASTCORMICSThresholdAnalysis
        Threshold analysis object defining core/non-core reactions.
    core_rxns : set[str]
        Set of identified core reaction IDs.
    noncore_rxns : set[str]
        Set of identified non-core reaction IDs.
    nonP_rxns : set[str]
        Set of identified non-penalty reaction IDs.
    result_model : cobra.Model
        Property accessing the final context-specific model from `fastcore_result`.
    kept_rxn_ids : np.ndarray
        Property accessing the kept reaction IDs from `fastcore_result`.
    removed_rxn_ids : np.ndarray
        Property accessing the removed reaction IDs from `fastcore_result`.
    """
    def __init__(self, log):
        super().__init__(log)

    # Properties docstrings are now part of the class docstring Attributes section
    @property
    def result_model(self):
        return self.fastcore_result.result_model

    @property
    def kept_rxn_ids(self):
        return self.fastcore_result.kept_rxn_ids

    @property
    def removed_rxn_ids(self):
        return self.fastcore_result.removed_rxn_ids


class CORDA_Analysis(BaseAnalysis):
    """Analysis result object for the CORDA algorithm.

    Encapsulates the results from the `apply_CORDA` function, which builds a
    context-specific metabolic model based on reaction confidence scores,
    typically derived from experimental data like transcriptomics.

    Parameters
    ----------
    log : dict
        A dictionary storing parameters used during the CORDA execution,
        including penalty factors (`penalty_factor`, `penalty_increase_factor`),
        support thresholds (`keep_if_support`), flux thresholds (`threshold`,
        `support_flux_value`), and bounds (`upper_bound`).

    Attributes
    ----------
    result_model : cobra.Model
        The final context-specific metabolic model generated by the CORDA
        algorithm, containing reactions deemed active in the specific context.
    conf_scores : dict[str, float]
        A dictionary mapping reaction variable IDs (including both forward and
        reverse directions, e.g., 'ATPS4r' and 'ATPS4r_reverse') to their
        final confidence scores after the CORDA refinement process. Scores
        typically range from -1 (low confidence, likely removed) to 3 (high
        confidence/core reaction, kept).
    threshold_analysis : ThresholdAnalysis
        An object containing details about the thresholding strategy used to
        convert continuous input data (e.g., gene expression) into the initial
        discrete confidence scores used by CORDA. The specific type of this
        object (e.g., `rFASTCORMICSThresholdAnalysis`) depends on the
        `predefined_threshold` or strategy used in `apply_CORDA`.
    removed_rxn_ids : numpy.ndarray
        A NumPy array containing the `cobra.Reaction` objects (not just their IDs)
        that were present in the original model but were removed during the
        CORDA model building process based on confidence scores and dependency
        assessments.
    algo_efficacy : dict or None
        A dictionary containing efficacy metrics (e.g., 'precision', 'recall',
        'F1_score', 'MCC') evaluating the performance of the algorithm. It
        compares the reactions present in the `result_model` against the
        initial high-confidence (core) and low-confidence (non-core) sets
        derived from the input `conf_scores`.
    """
    def __init__(self, log):
        super().__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}


class MBA_Analysis(BaseAnalysis):
    """Analysis result object for the Model Building Algorithm (MBA).

    Encapsulates the results from the `apply_MBA` function, representing a
    context-specific model built by iteratively removing reactions based on
    confidence levels (high, medium, none).

    Parameters
    ----------
    log : dict
        A dictionary storing parameters used during the MBA execution,
        such as `tolerance`, `epsilon`, and `random_state`.

    Attributes
    ----------
    result_model : cobra.Model
        The final context-specific metabolic model generated by the MBA algorithm
        after iterative removal of no-confidence reactions.
    removed_rxn_ids : numpy.ndarray
        A NumPy array containing the string IDs of reactions removed from the
        original model during the MBA process.
    threshold_analysis : ThresholdAnalysis or None
        An object containing details about the thresholding strategy used to
        derive the initial high- and medium-confidence reaction sets from
        continuous data (like gene expression). This is `None` if confidence
        sets were provided directly instead of using `data`. The specific type
        depends on the `predefined_threshold` used.
    algo_efficacy : float or None
        An efficacy score (e.g., F1-score) comparing the reactions present in
        the `result_model` against the initial high-confidence and
        no-confidence reaction sets.
    """
    def __init__(self, log):

        super().__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}


class mCADRE_Analysis(BaseAnalysis):
    """Analysis result object for the mCADRE algorithm.

    Encapsulates the results from the `apply_mCADRE` function, representing a
    context-specific model built by evaluating reactions based on expression,
    connectivity, evidence, and metabolic task performance.

    Parameters
    ----------
    log : dict
        A dictionary storing parameters used during the mCADRE execution,
        such as `exp_cutoff`, `absent_value`, `eta`, and `tol`.

    Attributes
    ----------
    result_model : cobra.Model
        The final context-specific metabolic model generated by mCADRE after
        iterative reaction removal.
    removed_rxn_ids : numpy.ndarray
        A NumPy array containing the string IDs of reactions removed from the
        original model during the pruning process.
    core_rxn_ids : numpy.ndarray
        A NumPy array containing the string IDs of reactions initially
        classified as 'core' based on the `exp_cutoff` threshold applied to
        mapped expression scores.
    non_expressed_rxn_ids : numpy.ndarray
        A NumPy array containing the string IDs of reactions initially
        classified as 'non-expressed' based on the `absent_value_indicator`.
    score_df : pandas.DataFrame
        A DataFrame containing the calculated scores for each reaction, indexed
        by reaction ID, with columns for 'expression' (mapped score),
        'connectivity' (based on neighboring reaction scores), and 'evidence'
        (user-provided or default zero). This DataFrame is sorted to guide the
        removal process.
    func_test_result : TaskAnalysis or None
        An object containing the results of the functional metabolic task tests
        performed on the model during the pruning process. `None` if no
        functional tests were provided or run.
    salvage_test_result : TaskAnalysis or None
        An object containing the results of the salvage pathway task tests
        performed on the model during the pruning process. `None` if no
        salvage tests were provided or run.
    threshold_analysis : ThresholdAnalysis
        An object containing details about the thresholding strategy used to
        convert continuous input data (e.g., gene expression) into the initial
        scores used for core/non-expressed classification. The specific type
        depends on the `predefined_threshold` used.
    algo_efficacy : float or None
        An efficacy score (e.g., F1-score) comparing the reactions present in
        the `result_model` against the initial `core_rxn_ids` and non-core IDs.
    """
    def __init__(self, log):
        super().__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}
        self._result_saving_params["core_rxn_ids"] = {"fm_name": "NDArrayStr"}
        self._result_saving_params["non_expressed_rxn_ids"] = {"fm_name": "NDArrayStr"}



class iMAT_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}


class INIT_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}
