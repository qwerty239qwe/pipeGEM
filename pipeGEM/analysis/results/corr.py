from ._base import *
from pipeGEM.plotting import CorrelationPlotter


class CorrelationAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)

    def plot(self,
             dpi=150,
             prefix="Dim_reduction_",
             **kwargs):
        pltr = CorrelationPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(result=self._result["correlation_result"],
                  **kwargs)