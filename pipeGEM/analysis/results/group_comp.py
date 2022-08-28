from ._base import *
from pipeGEM.plotting import ComponentComparisonPlotter


class ComparisonAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)


class ComponentComparisonAnalysis(ComparisonAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._result = None

    def add_result(self, result):
        self._result = result

    def plot(self,
             dpi=150,
             prefix="",
             *args,
             **kwargs):
        pltr = ComponentComparisonPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(result=self._result,
                  *args,
                  **kwargs)


class FluxCorrAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._result = None

    def add_result(self, result):
        self._result = result

    def plot(self,
             dpi=150,
             prefix="",
             *args,
             **kwargs
             ):
        pass