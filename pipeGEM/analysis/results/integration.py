import pandas as pd

from ._base import *


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
    """
    An object containing FASTCORE result.
    This should contain results including:
        result_model: pg.Model or cobra.Model
            A model containing the most core reactions and the least non-core reactions.
        removed_rxn_ids: np.ndarray
            An array contains the ids of removed reactions
        kept_rxn_ids: np.ndarray
            An array contains the ids of remaining reactions
    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """

    def __init__(self, log):
        super().__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}
        self._result_saving_params["kept_rxn_ids"] = {"fm_name": "NDArrayStr"}


class rFASTCORMICSAnalysis(BaseAnalysis):
    """
    An object containing rFASTCORMICS result.
    This should contain results including:
        fastcore_result: FASTCOREAnalysis
            This analysis object contains:
            result_model: pg.Model or cobra.Model
                A model containing the most core reactions and the least non-core reactions.
            removed_rxn_ids: np.ndarray
                An array contains the ids of removed reactions
            kept_rxn_ids: np.ndarray
                An array contains the ids of remaining reactions
        threshold_analysis: rFASTCORMICSThresholdAnalysis
            A threshold analysis object containing the thresholds to define the core and non-core reactions
        core_rxns: set[str]
            A set of identified core reactions' IDs
        noncore_rxns: set[str]
            A set of identified non-core reactions' IDs
        nonP_rxns: set[str]
            A set of identified undefined reactions' IDs
    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """

    def __init__(self, log):
        super().__init__(log)

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
    """
    An object containing CORDA result.
    This should contain results including:
        result_model: pg.Model or cobra.Model
            A model containing the most core reactions and the least non-core reactions.
        threshold_analysis: rFASTCORMICSThresholdAnalysis
            A threshold analysis object containing the thresholds to calculate the confidence scores
        conf_scores: dict[str, float]
            A dict containing the confidence scores used to define the importance of reactions
    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """
    def __init__(self, log):

        super().__init__(log)


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