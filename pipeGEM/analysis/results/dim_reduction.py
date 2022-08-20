from ._base import *
from pipeGEM.plotting import DimReductionPlotter


class PCA_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._result = None

    def add_result(self, result):
        self._result = result

    def plot(self,
             dpi=150,
             prefix="Dim_reduction_",
             method="PCA",
             **kwargs):

        pltr = DimReductionPlotter(dpi, prefix)
        pltr.plot(data=self._result,
                  group=self.log["group"],
                  method=method,
                  **kwargs)


class EmbeddingAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._result = None

    def add_result(self, result):
        self._result = result

    def plot(self,
             dpi=150,
             prefix="Dim_reduction_",
             method="PCA",
             **kwargs):
        pltr = DimReductionPlotter(dpi, prefix)
        pltr.plot(data=self._result,
                  group=self.log["group"],
                  method=method,
                  **kwargs)