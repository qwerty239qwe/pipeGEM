from typing import Optional, Union

from ._base import *
from .corr import CorrelationAnalysis
from .dim_reduction import PCA_Analysis, EmbeddingAnalysis
from pipeGEM.analysis._dim_reduction import prepare_PCA_dfs, prepare_embedding_dfs
from pipeGEM.analysis._threshold import threshold_finders


class DataAggregation(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)

    def __getitem__(self, item):
        return self._result["agg_data"][item]

    def find_local_threshold(self, **kwargs):
        tf = threshold_finders.create("local")
        return tf.find_threshold(self._result["agg_data"], **kwargs)

    def corr(self,
             by="sample",
             method='pearson'):
        if by not in ["sample", "feature"]:
            raise ValueError("argument 'by' should be 'sample' or 'feature'")
        corr_result = self._result["agg_data"].fillna(0).corr(method=method).fillna(0.) \
            if by == "sample" else self._result["agg_data"].T.fillna(0).corr(method=method).fillna(0.)
        result = CorrelationAnalysis(log={"by": by})
        result.add_result(dict(correlation_result=corr_result))
        return result

    def dim_reduction(self,
                      method="PCA",
                      **kwargs
                      ) -> Union[PCA_Analysis, EmbeddingAnalysis]:

        if method == "PCA":
            final_df, exp_var_df, component_df = prepare_PCA_dfs(self._result["agg_data"],
                                                                 **kwargs)
            result = PCA_Analysis(log={"method": "PCA", **kwargs, **self.log})
            result.add_result({"PC": final_df,
                               "exp_var": exp_var_df,
                               "components": component_df,
                               "group_annotation": self._result["group_annotation"]})
            return result
        else:
            emb_df = prepare_embedding_dfs(self._result["agg_data"],
                                           reducer=method,
                                           **kwargs)
            result = EmbeddingAnalysis(log={"method": method,
                                            **kwargs, **self.log})
            result.add_result({"embeddings": emb_df,
                               "group_annotation": self._result["group_annotation"]})
            return result
