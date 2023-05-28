from ._base import *


class ConsistencyAnalysis(BaseAnalysis):
    def __init__(self, log):
        self._consist_model = None
        self._removed_rxn_ids = None
        self._consistent_rxn_ids = None
        super(ConsistencyAnalysis, self).__init__(log)

    def add_result(self,
                   model=None,
                   removed_rxn_ids=None,
                   rxn_ids=None):
        self._consist_model = model if model is not None else self._consist_model
        self._removed_rxn_ids = removed_rxn_ids if removed_rxn_ids is not None else self._removed_rxn_ids
        self._consistent_rxn_ids = rxn_ids  if rxn_ids is not None else self._consistent_rxn_ids

    @property
    def consist_model(self):
        return self._consist_model

    @property
    def removed_rxn_ids(self):
        return self._removed_rxn_ids

    @property
    def consistent_rxn_ids(self):
        return self._consistent_rxn_ids


class FastCCAnalysis(ConsistencyAnalysis):
    def __init__(self, log):
        super(FastCCAnalysis, self).__init__(log)


class FVAAnalysis(ConsistencyAnalysis):
    def __init__(self, log):
        super(FVAAnalysis, self).__init__(log)