from typing import Optional, Union

from ._base import *
from .corr import CorrelationAnalysis
from .dim_reduction import PCA_Analysis, EmbeddingAnalysis
from pipeGEM.analysis._dim_reduction import prepare_PCA_dfs, prepare_embedding_dfs
from pipeGEM.analysis._threshold import threshold_finders
from pipeGEM.plotting import DataCatPlotter


class DataAggregation(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)

    def __getitem__(self, item):
        return self._result["agg_data"][item]

    def plot(self,
             ids,
             vertical,
             value_name="value",
             id_name="model",
             hue=None,
             dpi=150,
             prefix="",
             *args,
             **kwargs):
        long_data = self._result["agg_data"].reset_index().rename(
            columns={"index": id_name}).melt(
            value_name=value_name)
        if hue is not None:
            long_data = long_data.merge(self._result["group_annotation"], left_on=id_name, right_index=True)
            if hue not in long_data.columns:
                raise KeyError(f"{hue} is not in group_annotation, "
                               f"possible choices: {self._result['group_annotation'].columns}")

        pltr = DataCatPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(data=long_data,
                  ids=ids,
                  vertical=vertical,
                  id_col=self.log["prop"],
                  value_col=value_name,
                  hue=hue)

    def find_local_threshold(self, group_name=None, **kwargs):
        tf = threshold_finders.create("local")
        if group_name is not None:
            try:
                groups = self._result["group_annotation"].loc[:, group_name]
            except KeyError:
                raise KeyError(f"group_name: {group_name} is not in the group_annotation, possible choices are"
                      f"{self._result['group_annotation'].columns.to_list()}")
        else:
            groups = None
        return tf.find_threshold(self._result["agg_data"], groups=groups, **kwargs)

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
