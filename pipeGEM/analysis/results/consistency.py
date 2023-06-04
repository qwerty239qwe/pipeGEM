from ._base import *


class ConsistencyAnalysis(BaseAnalysis):
    def __init__(self, log):
        super(ConsistencyAnalysis, self).__init__(log)


class FastCCAnalysis(ConsistencyAnalysis):
    def __init__(self, log):
        super(FastCCAnalysis, self).__init__(log)


class FVAConsistencyAnalysis(ConsistencyAnalysis):
    def __init__(self, log):
        super(FVAConsistencyAnalysis, self).__init__(log)