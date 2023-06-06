from ._base import *


class ConsistencyAnalysis(BaseAnalysis):
    def __init__(self, log):
        super(ConsistencyAnalysis, self).__init__(log)
        self._result_saving_params["removed_rxn_ids"] = {"fm_name": "NDArrayStr"}
        self._result_saving_params["kept_rxn_ids"] = {"fm_name": "NDArrayStr"}


class FastCCAnalysis(ConsistencyAnalysis):
    def __init__(self, log):
        """
        An object containing FASTCC result.
        This should contain results including:
            consistent_model: pg.Model or cobra.Model
                A model without inconsistent reactions.
                An inconsistent reaction cannot produce non-zero flux at any circumstance.
            removed_rxn_ids: np.ndarray
                An array contains the ids of removed reactions
            kept_rxn_ids: np.ndarray
                An array contains the ids of remaining reactions
        Parameters
        ----------
        log: dict
            A dict storing parameters used to perform this analysis
        """
        super(FastCCAnalysis, self).__init__(log)


class FVAConsistencyAnalysis(ConsistencyAnalysis):
    def __init__(self, log):
        super(FVAConsistencyAnalysis, self).__init__(log)