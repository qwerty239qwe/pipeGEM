from ._base import *
from pipeGEM.plotting import DimReductionPlotter


class PCA_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._method = "PCA"

    def add_result(self, result):
        self._result = result

    def plot(self,
             dpi=150,
             prefix="Dim_reduction_",
             **kwargs):

        pltr = DimReductionPlotter(dpi, prefix)
        pltr.plot(flux_df=self._result,
                  groups=self.log["group"],
                  method=self._method,
                  **kwargs)


class EmbeddingAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._method = None

    def add_result(self, result, method):
        self._result = result
        self._method = method

    def plot(self,
             dpi=150,
             prefix="Dim_reduction_",
             **kwargs):
        pltr = DimReductionPlotter(dpi, prefix)
        pltr.plot(flux_df=self._result,
                  groups=self.log["group"],
                  method=self._method,
                  **kwargs)