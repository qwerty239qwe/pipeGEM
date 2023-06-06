import pandas as pd

from ._base import *


class EFluxAnalysis(BaseAnalysis):
    def __init__(self, log):
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
        super().__init__(log)


class GIMMEAnalysis(BaseAnalysis):

    def __init__(self, log):
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
    def  __init__(self, log):
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
        super().__init__(log)


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


class FASTCOREAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._rxn_ids = None
        self._removed_rxn_ids = None

    def add_result(self, model, rxn_ids, removed_rxn_ids):
        self._model = model
        self._rxn_ids = rxn_ids
        self._removed_rxn_ids = removed_rxn_ids

    @property
    def result_model(self):
        return self._model

    @property
    def removed_rxn_ids(self):
        return self._removed_rxn_ids

    @property
    def rxn_ids(self):
        return self._rxn_ids


class rFastCormicAnalysis(BaseAnalysis):
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


class MBA_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._removed_rxns = None
        self._threshold_analysis = None

    @property
    def threshold_analysis(self):
        return self._threshold_analysis

    @property
    def result_model(self):
        return self._model

    def add_result(self, model, removed_rxns, threshold_analysis):
        self._model = model
        self._removed_rxns = removed_rxns
        self._threshold_analysis = threshold_analysis


class mCADRE_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._removed_rxns = None
        self._threshold_analysis = None
        self._score_df = None
        self._core_rxns = None
        self._non_expressed_rxns = None
        self._func_test_result = None
        self._salvage_test_result = None

    @property
    def threshold_analysis(self):
        return self._threshold_analysis

    @property
    def result_model(self):
        return self._model

    def add_result(self, model, removed_rxns, score_df, core_rxns, non_expressed_rxns,
                   func_test_result, salvage_test_result):
        self._model = model
        self._removed_rxns = removed_rxns
        self._score_df = score_df
        self._core_rxns = core_rxns
        self._non_expressed_rxns = non_expressed_rxns
        self._func_test_result = func_test_result
        self._salvage_test_result = salvage_test_result


class iMAT_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._removed_rxns = None
        self._threshold_analysis = None

    @property
    def threshold_analysis(self):
        return self._threshold_analysis

    @property
    def result_model(self):
        return self._model

    def add_result(self, model, removed_rxns, threshold_analysis):
        self._model = model
        self._removed_rxns = removed_rxns
        self._threshold_analysis = threshold_analysis


class INIT_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._model = None
        self._removed_rxns = None
        self._threshold_analysis = None
        self._weight_dic = None

    @property
    def threshold_analysis(self):
        return self._threshold_analysis

    @property
    def result_model(self):
        return self._model

    @property
    def weight_dic(self):
        return self._weight_dic

    def add_result(self, model, removed_rxns, threshold_analysis, weight_dic):
        self._model = model
        self._removed_rxns = removed_rxns
        self._threshold_analysis = threshold_analysis
        self._weight_dic = weight_dic