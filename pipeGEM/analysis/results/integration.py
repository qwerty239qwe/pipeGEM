import pandas as pd

from ._base import *


class EFluxAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._rxn_bounds = None
        self._rxn_scores = None
        self._fluxes = None

    @property
    def rxn_bounds(self):
        return self._rxn_bounds

    @property
    def flux_result(self):
        return self._fluxes

    def add_result(self, rxn_bounds, rxn_scores, fluxes=None):
        self._rxn_bounds = rxn_bounds
        self._rxn_scores = rxn_scores
        self._fluxes = fluxes


class GIMMEAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._rxn_coefficents = None
        self._rxn_scores = None
        self._fluxes = None
        self._model = None

    @property
    def result_model(self):
        return self._model

    @property
    def rxn_coefficents(self):
        return self._rxn_coefficents

    @property
    def flux_result(self):
        return self._fluxes

    def add_result(self, rxn_coefficents, rxn_scores, fluxes=None, model=None):
        self._rxn_coefficents = rxn_coefficents
        self._rxn_scores = rxn_scores
        self._fluxes = fluxes if fluxes is not None else self._fluxes
        self._model = model


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
    def  __init__(self, log):
        super().__init__(log)
        self._model = None
        self._removed_rxns = None
        self._obj_dict = None

    @property
    def result_model(self):
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

    @property
    def flux_result(self):
        results = []
        for i, result_df in self._sampling_result.result.items():
            result_df = result_df.rename(columns={"flux": i})
            result_df.index = result_df["rxn_id"]
            result_df = result_df[i].to_frame().T
            results.append(result_df)
        return pd.concat(results, axis=0)


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


class CORDA_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._conf_scores = None
        self._threshold_analysis = None

    @property
    def threshold_analysis(self):
        return self._threshold_analysis

    @property
    def result_model(self):
        return self._model

    def add_result(self, model, conf_scores, threshold_analysis):
        self._model = model
        self._conf_scores = conf_scores
        self._threshold_analysis = threshold_analysis

