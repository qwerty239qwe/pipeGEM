from ._base import *
from pipeGEM.plotting import ComponentComparisonPlotter, ComponentNumberPlotter


class ComparisonAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)


class ComponentComparisonAnalysis(ComparisonAnalysis):
    def __init__(self, log):
        super().__init__(log)

    def plot(self,
             dpi=150,
             prefix="",
             *args,
             **kwargs):
        pltr = ComponentComparisonPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(result=self._result["comparison_df"],
                  *args,
                  **kwargs)


class ComponentNumberAnalysis(ComparisonAnalysis):
    def __init__(self, log):
        super().__init__(log)

    def plot(self,
             dpi=150,
             prefix="",
             group="group_name",
             name_order="default",
             *args,
             **kwargs):
        pltr = ComponentNumberPlotter(dpi=dpi, prefix=prefix)
        if name_order == "default":
            name_order = sorted(self._result[group].unique())
        pltr.plot(result=self._result,
                  name_order=name_order,
                  group=group,
                  *args,
                  **kwargs)


class FluxCorrAnalysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)

    def add_result(self, result):
        self._result = result

    def plot(self,
             dpi=150,
             prefix="",
             *args,
             **kwargs
             ):
        pass