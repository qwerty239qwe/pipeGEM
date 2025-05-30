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
                  row_groups=self._result["group_annotation"],
                  col_groups=self._result["group_annotation"],
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
            name_order = sorted(self._result["comp_df"][group].unique())
        pltr.plot(result=self._result["comp_df"],
                  name_order=name_order,
                  group=group,
                  *args,
                  **kwargs)