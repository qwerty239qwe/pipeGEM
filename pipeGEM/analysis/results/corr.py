from ._base import *
from pipeGEM.plotting import CorrelationPlotter


class CorrelationAnalysis(BaseAnalysis):
    """
    Correlation result containing a result dict with a key named correlation_result and a pd.DataFrame as value

    Parameters
    ----------
    log: dict
        A dict storing parameters used to perform this analysis
    """
    def __init__(self, log):
        super().__init__(log)

    def plot(self,
             dpi=150,
             prefix="Dim_reduction_",
             **kwargs):
        pltr = CorrelationPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(result=self._result["correlation_result"],
                  **kwargs)