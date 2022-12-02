import numpy as np


class KO_Impact:
    def __init__(self,
                 affected_rxns: np.ndarray):
        self._affected_rxns = affected_rxns

    @property
    def affected_rxns(self):
        return self._affected_rxns

    def __eq__(self, other):
        return np.array_equal(self.affected_rxns, other.affected_rxns)


