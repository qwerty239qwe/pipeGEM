from ._base import *


class ConsistencyAnalysis(BaseAnalysis):
    def __init__(self, log):
        super(ConsistencyAnalysis, self).__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}
        self._result_saving_params["kept_rxn_ids"] = {"fm_name": "NDArrayStr"}


class FastCCAnalysis(ConsistencyAnalysis):
    """
    FASTCC analysis result containing consistent_model, removed_rxn_ids, and kept_rxn_ids:
    `consistent_model`: pg.Model or cobra.Model
        A model without inconsistent reactions.
        An inconsistent reaction cannot produce non-zero flux at any circumstance.
    `removed_rxn_ids`: np.ndarray
        An array contains the ids of removed reactions
    `kept_rxn_ids`: np.ndarray
        An array contains the ids of remaining reactions

    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """
    def __init__(self, log):
        super(FastCCAnalysis, self).__init__(log)


class FVAConsistencyAnalysis(ConsistencyAnalysis):
    """
    FVA analysis result containing consistent_model, removed_rxn_ids, and kept_rxn_ids:
    `consistent_model`: pg.Model or cobra.Model
        A model without inconsistent reactions.
        An inconsistent reaction cannot produce non-zero flux at any circumstance.
    `removed_rxn_ids`: np.ndarray
        An array contains the ids of removed reactions
    `kept_rxn_ids`: np.ndarray
        An array contains the ids of remaining reactions

    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """
    def __init__(self, log):
        super(FVAConsistencyAnalysis, self).__init__(log)