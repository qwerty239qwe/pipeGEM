from ._base import *


class FastCCAnalysis(BaseAnalysis):
    def __init__(self, log):
        self._consist_model = None
        self._removed_rxn_ids = None
        super(FastCCAnalysis, self).__init__(log)

    def add_result(self, result):
        self._consist_model = result.get("model")
        self._removed_rxn_ids = result.get("removed_rxn_ids")

    @property
    def consist_model(self):
        return self._consist_model

    @property
    def removed_rxn_ids(self):
        return self._removed_rxn_ids

