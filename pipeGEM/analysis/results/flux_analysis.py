import pandas as pd

from typing import Optional, Union
from ._base import *
from pipeGEM.plotting import FBAPlotter, FVAPlotter, SamplingPlotter, HeatmapPlotter
from pipeGEM.analysis._dim_reduction import prepare_PCA_dfs, prepare_embedding_dfs
from .corr import CorrelationAnalysis
from .dim_reduction import PCA_Analysis, EmbeddingAnalysis


class NotAggregatedError(Exception):
    pass


class FluxAnalysis(BaseAnalysis):
    def __init__(self, log):
        super(FluxAnalysis, self).__init__(log)
        self._sol = None
        self._df: Optional[pd.DataFrame] = None

    def add_name(self, name, col_name="name"):
        self._df[col_name] = name
        self._df[col_name] = pd.Categorical(self._df[col_name])

    @classmethod
    def aggregate(cls, analyses, method, log, **kwargs):
        """
        Returns an aggregated dataframe,
        if concat method is used, return a df with 'name' column representing the model name

        Parameters
        ----------
        analyses: list of FluxAnalysis
        method: str
        log: dict
        kwargs

        Returns
        -------

        """
        new = cls(log=log)
        if method == "concat":
            dfs = []
            for a in analyses:
                one_df = a.result.reset_index().rename(columns={"index": "Reaction"}) \
                    if "Reaction" not in a.result.columns else a.result
                dfs.append(one_df)
            new._df = pd.concat(dfs, axis=0).reset_index(drop=True)
        else:
            dfs = []
            for a in analyses:
                one_df = a.result["fluxes"]
                dfs.append(one_df)
            new._df = pd.concat(dfs, axis=1)
            new._df = getattr(new._df, method)(axis=1).to_frame()
            new._df.columns = ["fluxes"]
        return new

    def add_result(self, sol):
        raise NotImplementedError()

    def plot(self, **kwargs):
        raise NotImplementedError()


class FBA_Analysis(FluxAnalysis):
    def __init__(self, log):
        super().__init__(log)

    @property
    def result(self):
        return self._df

    @property
    def solution(self):
        return self._sol

    def add_result(self, result, rxn_annotations=None):
        self._sol = result
        self._df = self._sol.to_frame()
        self._rxn_annotations = rxn_annotations

    def plot(self,
             dpi=150,
             prefix="FBA_",
             *args,
             **kwargs):
        pltr = FBAPlotter(dpi, prefix)
        pltr.plot(flux_df=self._df,
                  *args,
                  **kwargs)

    def plot_heatmap(self,
                     rxn_group_by,
                     sample_group_by,
                     dpi=150,
                     prefix="FBA_",
                     *args,
                     **kwargs
                     ):
        pltr = HeatmapPlotter(dpi=dpi, prefix=prefix)


    def dim_reduction(self,
                      method="PCA",
                      **kwargs
                      ) -> Union[PCA_Analysis, EmbeddingAnalysis]:
        if "group" not in self._log:
            raise NotAggregatedError("This analysis result contains only 1 model's fluxes, "
                                     "please use Group.do_flux_analysis to get a proper result for dim reduction")

        flux_df = self._df.pivot_table(index="Reaction", columns="name", values="fluxes", aggfunc="mean").fillna(0).T
        if method == "PCA":
            final_df, exp_var_df, component_df = prepare_PCA_dfs(flux_df.T,
                                                                 **kwargs)
            result = PCA_Analysis(log={**kwargs, **self.log})
            result.add_result({"PC": final_df, "exp_var": exp_var_df, "components": component_df})
            return result
        else:
            emb_df = prepare_embedding_dfs(flux_df.T,
                                           reducer=method,
                                           **kwargs)
            result = EmbeddingAnalysis(log={**kwargs, **self.log})
            result.add_result(emb_df, method=method)
            return result

    def hclust(self):
        pass

    def corr(self,
             by="name"):
        if "name" not in self._df:
            raise NotAggregatedError("This analysis result contains only 1 model's fluxes, "
                                     "please use Group.do_flux_analysis to get a proper result for dim reduction")
        if by not in ["name", "reaction"]:
            raise ValueError("argument 'by' should be either 'name' or 'reaction'")

        flux_df = self._df.pivot_table(index="Reaction", columns="name", values="fluxes", aggfunc="mean")
        if by == "reaction":
            flux_df = flux_df.T
        corr_result = flux_df.fillna(0).corr().fillna(0.)
        result = CorrelationAnalysis(log={"by": by})
        result.add_result(corr_result)
        return result


class FVA_Analysis(FluxAnalysis):
    def __init__(self, log):
        super().__init__(log)
        self._df = None

    @classmethod
    def aggregate(cls, analyses, method, log, **kwargs):
        new = cls(log=log)
        if method == "concat":
            dfs = []
            for a in analyses:
                one_df = a.result.reset_index().rename(columns={"index": "Reaction"})
                one_df["name"] = a.log["name"]
                dfs.append(one_df)
            new._df = pd.concat(dfs, axis=0)
            new._df["name"] = pd.Categorical(new._df["name"])
        else:
            min_dfs, max_dfs = [], []
            for a in analyses:
                min_df, max_df = a.result["minimum"], a.result["maximum"]
                min_df.name, max_df.name = f"minimum_{a.log['name']}", f"maximum_{a.log['name']}"
                min_dfs.append(min_df)
                max_dfs.append(max_df)
            new_min_df, new_max_df = pd.concat(min_dfs, axis=1), pd.concat(max_dfs, axis=1)
            new_min_df, new_max_df = getattr(new_min_df, method)(axis=1).to_frame(), getattr(new_max_df, method)(axis=1).to_frame()
            new._df = {"minimum": new_min_df, "maximum": new_max_df}
        return new

    def add_result(self, result):
        self._df = result

    @property
    def result(self):
        return self._df

    def plot(self,
             rxn_ids,
             dpi=150,
             prefix="FVA_",
             *args,
             **kwargs):
        pltr = FVAPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(rxn_ids=rxn_ids,
                  flux_df=self._df,
                  *args,
                  **kwargs)


class SamplingAnalysis(FluxAnalysis):
    def __init__(self, log):
        super(SamplingAnalysis, self).__init__(log)
        self._df_dic = {}

    @classmethod
    def aggregate(cls, analyses, method, log, **kwargs):
        new = cls(log=log)
        if method == "concat":
            for a in analyses:
                new._df_dic.update(a._df_dic)
        return new

    @property
    def result(self):
        return self._df_dic

    def add_name(self, name):
        for i in self._df_dic:
            self._df_dic[i]["name"] = name

    def add_result(self, result: pd.DataFrame):
        for i in range(result.shape[0]):
            one_sample = result.loc[i, :].to_frame()
            one_sample = one_sample.reset_index()
            one_sample.columns = ["rxn_id", "flux"]
            self._df_dic[i] = one_sample

    def plot(self,
             dpi=150,
             prefix="sampling_",
             *args,
             **kwargs):
        pltr = SamplingPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(flux_df_dic=self._df_dic,
                  *args,
                  **kwargs)


def combine(analyses, method, log, **kwargs):
    if len(analyses) < 2:
        raise ValueError("Analyses should be a container with more than 2 analysis objects")
    if isinstance(analyses[0], FluxAnalysis):
        return analyses[0].__class__.aggregate(analyses, method, log, **kwargs)

    raise ValueError("These analysis objects have no combining function.")