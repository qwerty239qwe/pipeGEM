import pandas as pd

from ._base import *
from pipeGEM.plotting import FBAPlotter, FVAPlotter, SamplingPlotter


class FluxAnalysis(BaseAnalysis):
    def __init__(self, log):
        super(FluxAnalysis, self).__init__(log)
        self._sol = None
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
            dfs = []
            for a in analyses:
                one_df = a.result["fluxes"]
                one_df.name = f"fluxes_{a.log['name']}"
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

    def add_result(self, result):
        self._sol = result
        self._df = self._sol.to_frame()

    def plot(self,
             dpi=150,
             prefix="FBA_",
             *args,
             **kwargs):
        pltr = FBAPlotter(dpi, prefix)
        pltr.plot(flux_df=self._df,
                  *args,
                  **kwargs)


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

    def add_result(self, result):
        self._df_dic = result.melt(var_name="rxn_id", value_name="flux")

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