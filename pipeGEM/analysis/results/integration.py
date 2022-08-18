from ._base import *


class EFluxAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._rxn_bounds = None
        self._rxn_scores = None

    @property
    def rxn_bounds(self):
        return self._rxn_bounds

    def add_result(self, rxn_bounds, rxn_scores):
        self._rxn_bounds = rxn_bounds
        self._rxn_scores = rxn_scores


class GIMMEAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._rxn_coefficents = None

    @property
    def rxn_coefficents(self):
        return self._rxn_coefficents

    def add_result(self, rxn_coefficents, rxn_scores):
        self._rxn_coefficents = rxn_coefficents
        self._rxn_scores =rxn_scores


class RIPTiDePruningAnalysis(BaseAnalysis):
    def  __init__(self, log):
        super().__init__(log)
        self._model = None
        self._removed_rxns = None
        self._obj_dict = None

    @property
    def model(self):
        return self._model

    def add_result(self, model, removed_rxns, obj_dict):
        self._model = model
        self._removed_rxns = removed_rxns
        self._obj_dict = obj_dict


class RIPTiDeSamplingAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._sampling_result = None

    def add_result(self, sampling_result):
        self._sampling_result = sampling_result


class rFastCormicAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._rxn_ids = None
        self._removed_rxn_ids = None
        self._core_rxns = None
        self._noncore_rxns = None
        self._nonP_rxns = None
        self._threshold_analysis = None

    @property
    def threshold_analysis(self):
        return self._threshold_analysis

    @property
    def result_model(self):
        return self._model

    def add_result(self, fastcore_result, core_rxns, noncore_rxns, nonP_rxns, threshold_analysis):
        self._model = fastcore_result.get("model")
        self._rxn_ids = fastcore_result.get("rxn_ids")
        self._removed_rxn_ids = fastcore_result.get("removed_rxn_ids")
        self._core_rxns = core_rxns
        self._noncore_rxns = noncore_rxns
        self._nonP_rxns = nonP_rxns
        self._threshold_analysis = threshold_analysis