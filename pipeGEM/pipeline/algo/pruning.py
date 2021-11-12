from pipeGEM.pipeline import Pipeline
from pipeGEM.integration.algo.fastcore import fastCore
from .consistent import FastCC


class FastCoreAlgo(Pipeline):
    def __init__(self):
        super().__init__()

    def run(self,
            model,
            C,
            nonP=None,
            epsilon=1e-6,
            return_rxn_ids=True,
            return_removed_rxn_ids=True,
            return_model=True):
        if nonP is None:
            nonP = set()

        self.output = fastCore(model=model,
                               C=C,
                               nonP=nonP,
                               epsilon=epsilon,
                               return_model=return_model,
                               return_rxn_ids=return_rxn_ids,
                               return_removed_rxn_ids=return_removed_rxn_ids)
        return self.output


class FastCore(Pipeline):
    def __init__(self):
        super().__init__()
        self.fastcc = FastCC()
        self.fastcore = FastCoreAlgo()

    def run(self,
            model,
            C,
            nonP = None,
            epsilon=1e-6,
            return_rxn_ids=True,
            return_removed_rxn_ids=True,
            return_model=True,
            **kwargs):
        consistent_mod = self.fastcc(model, **kwargs)["model"]
        self.output = self.fastcore(consistent_mod,
                                    C=C,
                                    nonP=nonP,
                                    epsilon=epsilon,
                                    return_model=return_model,
                                    return_rxn_ids=return_rxn_ids,
                                    return_removed_rxn_ids=return_removed_rxn_ids
                                    )
        return self.output


class rFastCormics(Pipeline):
    def __init__(self):
        super().__init__()
        self.fastcc = FastCC()

    def run(self, *args, **kwargs):
        pass


class CORDA(Pipeline):
    def __init__(self):
        super().__init__()


class mCARDRE(Pipeline):
    def __init__(self):
        super().__init__()


class SwiftCore(Pipeline):
    def __init__(self):
        super().__init__()

