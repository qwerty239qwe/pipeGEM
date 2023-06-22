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
        p_val_col = list(set([result.p_value_col for result in results]))
        assert p_val_col == 1, "P-value column should be the same across the analyses, " \
                               "please make sure concatenated analyses use the same method."
        result_df = pd.concat([result.result_df for result in results], axis=0)
        result_df["adjusted_p_value"] = bh_adjust(result_df[p_val_col[0]])
        new_obj = cls(new_log)
        new_obj.add_result(dict(result_df=result_df))
        return new_obj

    def mark_sig(self, query, key="sig"):
        sel_index = self._result["result_df"].query(query).index
        new_ser = pd.Series(data=[False for _ in range(self._result["result_df"].shape[0])],
                            index=self._result["result_df"].index)
        new_ser[sel_index] = True
        self._result["result_df"][key] = new_ser
        self._log.update({"sig_key": key,
                          "sig_query": query})

    def annotate(self, key, annot_dic):
        self._result["result_df"][key] = self._result["result_df"].index.to_series().map(annot_dic)

    def do_hypergeom_test(self, draw_from_col):
        hg_result_df = hypergeometric_test(data=self._result["result_df"],
                                           pathway_col=draw_from_col,
                                           sig_col=self._log["sig_key"])
        result = HyperGeometricTestResult(log=dict(draw_from_col=draw_from_col,
                                                   sig_col=self._log["sig_key"]))
        result.add_result(dict(result_df=hg_result_df))

    def plot(self, method, **kwargs):
        pass