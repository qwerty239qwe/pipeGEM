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
    """
    An object containing RIPTiDe pruning part result.
    This should contain results including:
        obj_dict: dict[str, float]
        flux_result: pd.DataFrame, optional
        result_model: pd.Model or cobra.Model, optional

    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """

    def __init__(self, log):

        super().__init__(log)


class RIPTiDeSamplingAnalysis(BaseAnalysis):
    """
    An object containing RIPTiDe sampling part result.
    This should contain results including:
        sampling_result: SamplingAnalysis

    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """

    def __init__(self, log):
        super().__init__(log)

    @property
    def flux_result(self):
        return self._result["sampling_result"].flux_df


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
    """
    This object contains:
        result_model: pg.Model or cobra.Model
            A model containing the most core reactions and the least non-core reactions.
        threshold_analysis: rFASTCORMICSThresholdAnalysis
            A threshold analysis object containing the thresholds to calculate the confidence scores
        removed_rxn_ids: np.ndarray
            An array contains the ids of removed reactions
    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """
    def __init__(self, log):

        super().__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}


class mCADRE_Analysis(BaseAnalysis):
    """
    This object contains:
        result_model: pg.Model or cobra.Model
            A model containing the most core reactions and the least non-core reactions.
        threshold_analysis: rFASTCORMICSThresholdAnalysis
            A threshold analysis object containing the thresholds to calculate the confidence scores
        removed_rxn_ids: np.ndarray
            An array contains the ids of removed reactions
        core_rxn_ids: np.ndarray
        non_expressed_rxn_ids: np.ndarray
        score_df: pd.DataFrame
        func_test_result: TaskAnalysis
        salvage_test_result: TaskAnalysis
        threshold_analysis: ThresholdAnalysis
    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
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
