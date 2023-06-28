import pandas as pd

from ._base import *
from scipy import stats
from pipeGEM.analysis._utils import bh_adjust
from pipeGEM.analysis.pathway import hypergeometric_test


class NormalityTestResult(BaseAnalysis):
    def __init__(self, log):
        super(NormalityTestResult, self).__init__(log=log)

    def plot(self, method, **kwargs):
        pass


class VarHomogeneityTestResult(BaseAnalysis):
    def __init__(self, log):
        super(VarHomogeneityTestResult, self).__init__(log=log)

    def plot(self, method, **kwargs):
        pass


class HyperGeometricTestResult(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log=log)


class PairwiseTestResult(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log=log)

    @classmethod
    def aggregate(cls, results, log=None):
        new_log = {} if log is None else log
        p_value_col_name = results[0].p_value_col
        p_val_col = list(set([result.p_value_col for result in results]))
        assert len(p_val_col) == 1, "P-value column should be the same across the analyses, " \
                                    "please make sure concatenated analyses use the same method."
        result_df = pd.concat([result.result_df for result in results], axis=0)
        result_df["adjusted_p_value"] = bh_adjust(result_df[p_val_col[0]])
        result_df = result_df.reset_index(drop=True)
        new_obj = cls({**dict(p_value_col=p_value_col_name), **new_log})
        new_obj.add_result(dict(result_df=result_df))
        return new_obj

    def mark_sig(self,
                 use_default=True,
                 query=None,
                 key="sig"):
        if use_default:
            multicomp_p_val_col_name = self.log["multicomp_log"]["p_value_col"]
            pw_p_val_col_name = self.p_value_col
            query = f"{multicomp_p_val_col_name} < 0.05 and {pw_p_val_col_name} < 0.05"

        sel_index = self._result["result_df"].query(query).index
        new_ser = pd.Series(data=[False for _ in range(self._result["result_df"].shape[0])],
                            index=self._result["result_df"].index)
        new_ser[sel_index] = True
        self._result["result_df"][key] = new_ser
        self._log.update({"sig_key": key,
                          "sig_query": query})

    def annotate(self, key, annot_dic, map_from="label", map_index=False):
        if map_index:
            self._result["result_df"][key] = self._result["result_df"].index.to_series().map(annot_dic)
        else:
            self._result["result_df"][key] = self._result["result_df"][map_from].map(annot_dic)

    def do_hypergeom_test(self, draw_from_col):
        comp_pairs = self._result["result_df"].apply(lambda x: (x["A"], x["B"]), axis=1).unique()

        hg_result_dfs = []
        for ga, gb in comp_pairs:
            sel_data = self._result["result_df"]
            sel_data = sel_data[(sel_data["A"] == ga) & (sel_data["B"] == gb)]
            hg_result_df = hypergeometric_test(data=sel_data,
                                               pathway_col=draw_from_col,
                                               sig_col=self._log["sig_key"])
            hg_result_df["contrast_pair"] = f"{ga}_vs_{gb}"
            hg_result_dfs.append(hg_result_df)
        hg_result_dfs = pd.concat(hg_result_dfs, axis=0)
        result = HyperGeometricTestResult(log=dict(draw_from_col=draw_from_col,
                                                   sig_col=self._log["sig_key"]))
        result.add_result(dict(result_df=hg_result_dfs))
        return result

    def plot(self, method, **kwargs):
        pass

    def merge_multicomp_result(self, multicomp_result, on="label", suffixes=("_pairwise", "_multicomp")):
        self._log.update({"multicomp_log": multicomp_result.log,
                          "multicomp_merge_on": on,
                          "multicomp_pairwise_suffix": suffixes[0],
                          "multicomp_multicomp_suffix": suffixes[1]})

        new_result_df = pd.merge(self._result["result_df"], multicomp_result.result_df,
                                 on=on, suffixes=suffixes)
        self.add_result(dict(result_df=new_result_df))


class MultiGroupComparisonTestResult(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log=log)

    @classmethod
    def aggregate(cls, results, log=None):
        new_log = {} if log is None else log
        p_value_col_name = results[0].p_value_col
        p_val_col = list(set([result.p_value_col for result in results]))
        assert len(p_val_col) == 1, "P-value column should be the same across the analyses, " \
                                    "please make sure concatenated analyses use the same method."
        result_df = pd.concat([result.result_df for result in results], axis=0)
        result_df = result_df.reset_index(drop=True)
        new_obj = cls({**dict(p_value_col=p_value_col_name), **new_log})
        new_obj.add_result(dict(result_df=result_df))
        return new_obj
