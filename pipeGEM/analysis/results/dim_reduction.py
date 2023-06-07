from ._base import *
from pipeGEM.plotting import DimReductionPlotter


class PCA_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)

    def plot(self,
             dpi=150,
             color_by="group_name",
             prefix="Dim_reduction_",
             **kwargs):
        if color_by is None:
            groups = {m: [m] for m in self.log["group_annotation"].index.to_list()}
        else:
            gb = self.log["group_annotation"].groupby(color_by).apply(lambda x: list(x.index))
            groups = {i: row for i, row in gb.items()}

        pltr = DimReductionPlotter(dpi, prefix)
        pltr.plot(result_df=self._result,
                  groups=groups,
                  method=self._log["method"],
                  **kwargs)


class EmbeddingAnalysis(PCA_Analysis):
    def __init__(self, log):
        super().__init__(log)