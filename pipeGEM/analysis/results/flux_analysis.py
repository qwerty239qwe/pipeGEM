import pandas as pd
import numpy as np
import pingouin
from tqdm import tqdm
import warnings
import re
from typing import Optional, Union, List

from ._base import *
from pipeGEM.plotting import FBAPlotter, FVAPlotter, SamplingPlotter, HeatmapPlotter
from pipeGEM.analysis._dim_reduction import prepare_PCA_dfs, prepare_embedding_dfs
from .corr import CorrelationAnalysis
from .dim_reduction import PCA_Analysis, EmbeddingAnalysis
from .stat import PairwiseTestResult, MultiGroupComparisonTestResult
from dask.distributed import Client
from dask.distributed import progress


def separate_operands(input_string) -> List[str]:
    # Split the string using regular expression to capture operands and other substrings
    separated_list = re.findall(r'[+-/*\^]|[^+-/*\^]+', input_string)
    return separated_list


class NotAggregatedError(Exception):
    pass


class FluxAnalysis(BaseAnalysis):
    def __init__(self, log):

        super(FluxAnalysis, self).__init__(log)

    def add_categorical(self,
                        value: str,
                        col_name: str = "name") -> None:
        """
        Add a categorical column to flux_df in the result

        Parameters
        ----------
        value: str
            The values filled in the categorical column
        col_name: str
            The columns name of the categorical column

        Returns
        -------
        None
        """
        self._result["flux_df"][col_name] = value
        self._result["flux_df"][col_name] = pd.Categorical(self._result["flux_df"][col_name])
        if "categorical" in self._log:
            self._log["categorical"].add(col_name)
        else:
            self._log["categorical"] = set(col_name) if not isinstance(col_name, str) else set([col_name])

    @classmethod
    def aggregate(cls,
                  analyses: List["FluxAnalysis"],
                  method: str,
                  log: Optional[dict] = None,
                  **kwargs):
        """
        Returns an aggregated dataframe,
        if concat method is used, return a df with 'name' column representing the model name

        Parameters
        ----------
        analyses: list of FluxAnalysis
            FluxAnalysis objects used to be aggregated
        method: str
            A string represents the aggregation method
            Possible choices are: concat, sum, mean, and median
        log: dict, optional
            A dict contains new analysis results' information
        kwargs: dict
            Additional keyword arguments added to the result dict
        Returns
        -------
        aggregated_flux_analysis: FluxAnalysis

        """
        log = log if log is not None else {}
        cat_log = set()
        if method == "concat":
            dfs = []
            for a in analyses:
                cat_log |= a.log["categorical"]
                one_df = a.result["flux_df"].reset_index().rename(columns={"index": "Reaction"}) \
                    if "Reaction" not in a.result["flux_df"].columns else a.result["flux_df"]
                dfs.append(one_df)
            new_df = pd.concat(dfs, axis=0).reset_index(drop=True)
        else:
            dfs = []
            for a in analyses:
                cat_log |= a.log["categorical"]
                one_df = a.result["flux_df"]["fluxes"]
                dfs.append(one_df)
            new_df = pd.concat(dfs, axis=1)
            new_df = getattr(new_df, method)(axis=1).to_frame()
            new_df.columns = ["fluxes"]
        new = cls(log={"categorical": cat_log, **log})
        new.add_result({**kwargs, **dict(flux_df=new_df)})
        return new

    def _sel_data_for_diff_test(self, data, rxn_id, between):
        raise NotImplementedError()

    def _sel_rxns_for_diff_test(self):
        raise NotImplementedError()

    def _diff_test_iter(self,
                        r,
                        between,
                        parametric,
                        parametric_params,
                        method,
                        label_str_format,
                        data):
        from pipeGEM.analysis._stat import PairwiseTester, MultiGroupComparison

        data_to_be_analyzed = self._sel_data_for_diff_test(data=data,
                                                           rxn_id=r,
                                                           between=between)

        inferred_parametric = parametric
        gb_data = data_to_be_analyzed.groupby(between)
        mulcom_res, test_res = None, None

        if gb_data.count().query(f"{r} > 2").shape[0] >= 3:
            mulcom_res = MultiGroupComparison().test(data_to_be_analyzed,
                                                     dep_var=r,
                                                     between=between,
                                                     parametric=parametric,
                                                     parametric_params=parametric_params,
                                                     added_label=label_str_format.format(reaction=r))
            inferred_parametric = mulcom_res.inferred_parametric

        if gb_data.count().query(f"{r} > 2").shape[0] >= 2:
            test_res = PairwiseTester().test(data=data_to_be_analyzed,
                                             dep_var=r,
                                             between=between,
                                             parametric=inferred_parametric,
                                             method=method,
                                             added_label=label_str_format.format(reaction=r))

        return mulcom_res, test_res

    def diff_test(self,
                  between,
                  parametric="auto",
                  parametric_params=None,
                  method="mw",
                  label_str_format="{reaction}",
                  save_p_val=True,
                  do_parallel=False,
                  **kwargs) -> PairwiseTestResult:

        if between not in self._result["flux_df"].columns:
            raise KeyError(f"{between} is not in the categorical features. \n"
                           f"Possible features are {list(self.log['categorical'])}")
        assert parametric in ["auto", True, False]
        all_rxns = self._sel_rxns_for_diff_test()

        if do_parallel:
            client = Client(**kwargs)
            futures = []
            remote_data = client.scatter(self.result["flux_df"])
            for r in all_rxns:
                future = client.submit(self._diff_test_iter,
                                       r,
                                       between,
                                       parametric,
                                       parametric_params,
                                       method,
                                       label_str_format,
                                       remote_data)
                futures.append(future)
            gathered_results = client.gather(futures)
            progress(gathered_results)
        else:
            gathered_results = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for r in tqdm(all_rxns):
                    result = self._diff_test_iter(r=r,
                                                  between=between,
                                                  parametric=parametric,
                                                  parametric_params=parametric_params,
                                                  method=method,
                                                  label_str_format=label_str_format,
                                                  data=self.result["flux_df"])
                    gathered_results.append(result)

        all_res, all_mulcom_res = [dr for mr, dr in gathered_results if dr is not None], [mr for mr, dr in gathered_results if mr is not None]
        result = PairwiseTestResult.aggregate(results=all_res,
                                              log=dict(label_str_format=label_str_format,
                                                       between=between,
                                                       parametric=parametric, ))
        if len(all_mulcom_res) > 0:
            multcomp_result = MultiGroupComparisonTestResult.aggregate(results=all_mulcom_res)
            result.merge_multicomp_result(multcomp_result, on="label")

        if save_p_val:
            self._diff_test_result = result
        return result

    def plot(self, **kwargs):
        raise NotImplementedError()


class FBA_Analysis(FluxAnalysis):
    """
    FBA analysis result.

    Attributes
    ----------
    log
    """

    def __init__(self, log):
        super().__init__(log)

    def plot(self,
             dpi=150,
             prefix="FBA_",
             *args,
             **kwargs):
        pltr = FBAPlotter(dpi, prefix)

        if "group_annotation" in self._result:
            to_be_annot = [c for c in self.group_annotation.columns
                           if c not in self._result["flux_df"].columns]
            if len(to_be_annot) > 0:
                to_be_plot = self._result["flux_df"].merge(self.group_annotation[to_be_annot],
                                                           right_index=True,
                                                           left_on="model")
            else:
                to_be_plot = self._result["flux_df"]
        else:
            to_be_plot = self._result["flux_df"]

        pltr.plot(flux_df=to_be_plot,
                  *args,
                  **kwargs)

    def plot_heatmap(self,
                     rxn_group_by=None,
                     sample_group_by=None,
                     group_by_agg_method="mean",
                     dpi=150,
                     prefix="FBA_",
                     *args,
                     **kwargs
                     ):
        pltr = HeatmapPlotter(dpi=dpi, prefix=prefix)

        data = self._result["flux_df"].copy()

        if rxn_group_by is not None:
            if isinstance(rxn_group_by, list):
                for rg in rxn_group_by:
                    data[rg] = data["Reaction"].map(self._log["rxn_annotations"][rg]).astype("category")
            else:
                data[rxn_group_by] = data["Reaction"].map(self._log["rxn_annotations"][rxn_group_by]).astype("category")
        rxn_index_name = rxn_group_by if rxn_group_by is not None else "Reaction"
        sample_col_name = sample_group_by if sample_group_by is not None else "model"

        data = pd.pivot_table(data=data.drop(columns=["reduced_costs"]),
                              values="fluxes",
                              index=rxn_index_name,
                              columns=sample_col_name,
                              aggfunc=group_by_agg_method).fillna(0)
        pltr.plot(result=data,
                  *args,
                  **kwargs)

    def dim_reduction(self,
                      method="PCA",
                      **kwargs
                      ) -> Union[PCA_Analysis, EmbeddingAnalysis]:
        if "categorical" not in self.log:
            raise NotAggregatedError("This analysis result only contains fluxes of a model, "
                                     "please use Group.do_flux_analysis to get a proper result for dim reduction")
        flux_df = self._result["flux_df"].pivot_table(index="Reaction",
                                                      columns=self.log["group_by"],
                                                      values="fluxes",
                                                      aggfunc="mean").fillna(0).T
        dim_log = {**kwargs, **self.log} if self.log["group_by"] == "model" else \
            {"group": {self.log["group_by"]: flux_df.index},
             **kwargs, **{k: v for k, v in self.log.items() if k != "group"}}
        dim_log.update({"dr_method": method})
        if method == "PCA":
            final_df, exp_var_df, component_df = prepare_PCA_dfs(flux_df.T,
                                                                 **kwargs)
            result = PCA_Analysis(log=dim_log)
            result.add_result({"PC": final_df,
                               "exp_var": exp_var_df,
                               "components": component_df,
                               "group_annotation": self.result["group_annotation"]})
            return result
        else:
            emb_df = prepare_embedding_dfs(flux_df.T,
                                           reducer=method,
                                           **kwargs)
            result = EmbeddingAnalysis(log=dim_log)
            result.add_result({"embeddings": emb_df,
                               "group_annotation": self.result["group_annotation"]})
            return result

    def corr(self,
             rxn_corr=False,
             group_by="model",
             **kwargs):
        if group_by not in self._result["flux_df"].columns:
            raise KeyError(f"{group_by} is not in the categorical features. \n"
                           f"Possible features are {list(self.log['categorical'])}")

        if self._result["flux_df"][group_by].unique().shape[0] == 1:
            raise NotAggregatedError("This analysis result contains only 1 model's fluxes, "
                                     "please use Group.do_flux_analysis to get a proper result for dim reduction")

        flux_df = self._result["flux_df"].pivot_table(index="Reaction",
                                                      columns=group_by,
                                                      values="fluxes",
                                                      aggfunc="mean")
        if rxn_corr:
            flux_df = flux_df.T
        corr_result = flux_df.fillna(0).corr(**kwargs).fillna(0.)
        result = CorrelationAnalysis(log={"by": group_by,
                                          **kwargs})
        result.add_result(dict(correlation_result=corr_result))
        return result

    def _sel_data_for_diff_test(self, data, rxn_id, between):
        rxn_f = data.query(f"Reaction == '{rxn_id}'")
        rxn_f = rxn_f.rename(columns={"fluxes": rxn_id})
        rxn_f = rxn_f.loc[:, [rxn_id, between]]
        return rxn_f

    def _sel_rxns_for_diff_test(self):
        return self.result["flux_df"]["Reaction"].to_list()


class FVA_Analysis(FluxAnalysis):
    def __init__(self, log):
        super().__init__(log)

    @classmethod
    def aggregate(cls,
                  analyses: List["FVA_Analysis"],
                  method: str,
                  log: Optional[dict] = None,
                  **kwargs):
        """
        Returns an aggregated FVA_Analysis,
        if concat method is used, return a df with 'name' column representing the model name

        Parameters
        ----------
        analyses: list of FVA_Analysis
            FVA_Analysis objects to be aggregated
        method: str
            A string represents the aggregation method
            Possible choices are: concat, sum, mean, and median
        log: dict, optional
            A dict contains new analysis results' information
        kwargs: dict
            Additional keyword arguments
        Returns
        -------
        aggregated_flux_analysis: FluxAnalysis

        """
        new = cls(log=log)
        if method == "concat":
            dfs = []
            for a in analyses:
                one_df = a.result["flux_df"].reset_index().rename(columns={"index": "Reaction"})
                dfs.append(one_df)
            new_df = pd.concat(dfs, axis=0)
        else:
            min_dfs, max_dfs = [], []
            for a in analyses:
                min_df, max_df = a.result["flux_df"]["minimum"], a.result["flux_df"]["maximum"]
                min_df.name, max_df.name = f"minimum_{a.log['name']}", f"maximum_{a.log['name']}"
                min_dfs.append(min_df)
                max_dfs.append(max_df)
            new_min_df, new_max_df = pd.concat(min_dfs, axis=1), pd.concat(max_dfs, axis=1)
            new_min_df, new_max_df = getattr(new_min_df, method)(axis=1).to_frame(), \
                getattr(new_max_df, method)(axis=1).to_frame()
            new_df = {"minimum": new_min_df, "maximum": new_max_df}
        new.add_result({"flux_df": new_df})
        return new

    def plot(self,
             rxn_ids,
             dpi=150,
             prefix="FVA_",
             *args,
             **kwargs):
        pltr = FVAPlotter(dpi=dpi, prefix=prefix)
        pltr.plot(rxn_ids=rxn_ids,
                  flux_df=self._result["flux_df"],
                  *args,
                  **kwargs)


class SamplingAnalysis(FluxAnalysis):
    def __init__(self, log):
        super(SamplingAnalysis, self).__init__(log)
        self._result_loading_params["flux_df"] = {"index_col": 0}

    def rename(self, **kwargs) -> None:
        self._result["flux_df"]: pd.DataFrame = self._result["flux_df"].rename(**kwargs)

    def __getitem__(self, item) -> "SamplingAnalysis":
        new_analysis = self.__class__(log={k: v for k, v in self.log.items()})
        cat_cols = list(self.log["categorical"])
        if isinstance(item, str):
            item = [item]
        new_analysis.add_result({"flux_df": self.flux_df.loc[:, item+cat_cols]})
        return new_analysis

    def __add__(self, other):
        new_analysis = self.__class__(log={k: v for k, v in self.log.items()})
        cat_cols = list(self.log["categorical"])
        if isinstance(other, type(self)):
            if not self.flux_df[cat_cols].equals(other.flux_df[cat_cols]):
                raise ValueError("Adding two SamplingAnalysis with different categorical data."
                                 "Please check if the two sampling analysis are generated from the same model/group.")
            this_num = self.flux_df.drop(columns=cat_cols)
            other_num = other.flux_df.drop(columns=list(other.log["categorical"]))
            sum_num = this_num + other_num
        else:
            sum_num = self.flux_df.drop(columns=cat_cols) + other

        new_analysis.add_result({"flux_df": pd.concat([sum_num,
                                                      self.flux_df[cat_cols]], axis=1)})
        return new_analysis

    def __neg__(self):
        new_analysis = self.__class__(log={k: v for k, v in self.log.items()})
        cat_cols = list(self.log["categorical"])
        num_df = -self.flux_df.drop(columns=cat_cols)

        new_analysis.add_result({"flux_df": pd.concat([num_df, self.flux_df[cat_cols]], axis=1)})
        return new_analysis

    def __sub__(self, other):
        return self.__add__(-other)

    def operate(self, operation: str):
        operands = ["+", "-", "*", "/", "^"]
        pd_ops = {"+": "add", "-": "sub", "*": "mul", "/": "div", "^": "pow"}
        if any([op in c for c in self._result["flux_df"].columns for op in operands]):
            raise AttributeError("Please remove special characters: ['+', '-', '*', '/', '^'] in the columns of flux_df.",
                                 "Consider using result.rename(columns={'problematic_col': 'good_col'}) to solve this issue.")
        returned_series = pd.Series(data=0, index=self._result["flux_df"].index)
        new_col = None
        if "=" in operation:
            new_col, operation = operation.split("=")

        substrs = separate_operands(operation)
        if "-" == substrs[0]:
            substrs = ["0"] + substrs
        mem_op = None
        for i, substr in enumerate(substrs):
            if substr in operands:
                mem_op = substr
            elif substr in self._result["flux_df"]:
                if mem_op is None:
                    returned_series = self._result["flux_df"][substr]
                else:
                    returned_series = getattr(returned_series, pd_ops[mem_op])(self._result["flux_df"][substr])
            else:
                try:
                    num = float(substr)
                    returned_series = getattr(returned_series, pd_ops[mem_op])(num)
                except ValueError:
                    raise ValueError(f"substring {substr} is neither a column name in the flux df nor a number")
        if new_col is not None:
            self._result["flux_df"][new_col] = returned_series
            return

        return returned_series

    @classmethod
    def aggregate(cls,
                  analyses: List["FluxAnalysis"],
                  method: str = "concat",
                  log: Optional[dict] = None,
                  **kwargs):
        cat_log = set()
        log = log or {}

        na_fill_val = kwargs.get("na_value", 0)
        if method == "concat":
            for a in analyses:
                cat_log |= a.log["categorical"]
            new = cls(log={"categorical": cat_log, **log})
            new.add_result(dict(flux_df=pd.concat([i.flux_df for i in analyses],
                                                  axis=0).reset_index(drop=True).fillna(value=na_fill_val)))
        else:
            raise NotImplementedError()
        return new

    def plot(self,
             rxn_id,
             plot_significance=False,
             dpi=150,
             prefix="sampling_",
             *args,
             **kwargs):
        pltr = SamplingPlotter(dpi=dpi, prefix=prefix)
        if plot_significance:
            stat_analysis = self._diff_test_result
        else:
            stat_analysis = None

        pltr.plot(flux_df=self._result["flux_df"],
                  rxn_id=rxn_id,
                  stat_analysis=stat_analysis,
                  *args,
                  **kwargs)

    def _sel_data_for_diff_test(self, data, rxn_id, between):
        sel_col = [between, rxn_id] if isinstance(between, str) else between + [rxn_id]
        return data.loc[:, sel_col]

    def _sel_rxns_for_diff_test(self):
        return [c for c in self.result["flux_df"].columns if c not in self.log["categorical"]]


def combine(analyses, method, log, **kwargs):
    if len(analyses) < 2:
        raise ValueError("Analyses should be a container with more than 2 analysis objects")
    if isinstance(analyses[0], FluxAnalysis):
        return analyses[0].__class__.aggregate(analyses, method, log, **kwargs)

    raise ValueError("These analysis objects have no combining function.")