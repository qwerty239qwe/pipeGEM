from ._base import *


class DataAggregation(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)

    def add_result(self, result):
        self._result = result

    def corr(self):
        pass