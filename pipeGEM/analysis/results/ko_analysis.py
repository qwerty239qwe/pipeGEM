from ._base import *


class KO_Analysis(BaseAnalysis):
    def __init__(self, log):
        super().__init__(log)

    @property
    def result(self):
        return self._df

    def add_result(self, result):
        self._df = result


class Single_KO_Analysis(KO_Analysis):
    def __init__(self, log):
        super().__init__(log)