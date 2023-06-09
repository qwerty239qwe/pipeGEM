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


