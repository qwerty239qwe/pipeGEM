from typing import Optional, Union

from ._base import *
from .corr import CorrelationAnalysis
from .dim_reduction import PCA_Analysis, EmbeddingAnalysis
from pipeGEM.analysis._dim_reduction import prepare_PCA_dfs, prepare_embedding_dfs


class DataAggregation(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)

    def __getitem__(self, item):
        return self._result[item]

    def add_result(self, result):
        self._result = result

    def corr(self,
             by="sample"):
        if by not in ["sample", "feature"]:
            raise ValueError("argument 'by' should be 'sample' or 'feature'")
        corr_result = self._result.fillna(0).corr().fillna(0.) if by == "sample" else self._result.T.fillna(0).corr().fillna(0.)
        result = CorrelationAnalysis(log={"by": by})
        result.add_result(corr_result)
        return result

    def dim_reduction(self,
                      method="PCA",
                      **kwargs
                      ) -> Union[PCA_Analysis, EmbeddingAnalysis]:

        if method == "PCA":
            final_df, exp_var_df, component_df = prepare_PCA_dfs(self._result,
                                                                 **kwargs)
            result = PCA_Analysis(log={**kwargs, **self.log})
            result.add_result({"PC": final_df, "exp_var": exp_var_df, "components": component_df})
            return result
        else:
            emb_df = prepare_embedding_dfs(self._result,
                                           reducer=method,
                                           **kwargs)
            result = EmbeddingAnalysis(log={**kwargs, **self.log})
            result.add_result(emb_df, method=method)
            return result
