from ._base import *
from pipeGEM.plotting import ComponentComparisonPlotter, ComponentNumberPlotter


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


class ComponentNumberAnalysis(ComparisonAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._result = None
        self.name_order = None
        self._group_name = None

    def add_result(self, result, name_order=None, group="group"):
        self._result = result
        self.name_order = name_order
        self._group_name = group

    def plot(self,
             dpi=150,
             prefix="",
             group=None,
             name_order=None,
             *args,
             **kwargs):
        pltr = ComponentNumberPlotter(dpi=dpi, prefix=prefix)
        group = self._group_name if group is None else group
        if name_order == "default":
            name_order = sorted(self._result[group].unique())
        pltr.plot(result=self._result,
                  name_order=self.name_order if name_order is None else name_order,
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