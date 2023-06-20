import pandas as pd

from ._base import *
from scipy import stats


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


class PairwiseTestResult(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log=log)

    @classmethod
    def aggregate(cls, results, log=None):
        new_log = {} if log is None else log
        result_df = pd.concat([result.result_df for result in results], axis=0)
        new_obj = cls(new_log)
        new_obj.add_result(dict(result_df=result_df))
        return new_obj

    def plot(self, method, **kwargs):
        pass