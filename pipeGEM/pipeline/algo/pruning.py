from pipeGEM.pipeline import Pipeline
from pipeGEM.integration.algo.fastcore import fastCore
from pipeGEM.integration.algo.swiftcore import swiftCore
from pipeGEM.pipeline.algo import FastCC, SwiftCC
from pipeGEM.pipeline.threshold import BimodalThreshold
from pipeGEM.pipeline.preprocessing import GeneDataDiscretizer


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
    def __init__(self, consist_method = "fastcc"):
        super().__init__()
        if consist_method == "fastcc":
            self.consist_cc = FastCC()
        elif consist_method == "swiftcc":
            self.consist_cc = SwiftCC()
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
        consistent_mod = self.consist_cc(model,
                                         return_model=return_model,
                                         return_rxn_ids=return_rxn_ids,
                                         **kwargs)["model"]
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
    def __init__(self,
                 consist_method="swiftcc"):
        super().__init__()
        if consist_method == "fastcc":
            self.consist_cc = FastCC()
        elif consist_method == "swiftcc":
            self.consist_cc = SwiftCC()

        self.threshold = BimodalThreshold()
        self.disc = GeneDataDiscretizer()

    def run(self,
            model,
            data,
            *args,
            **kwargs):
        # get consistent model
        c_model = self.consist_cc(model,
                                  return_model=True,
                                  **kwargs)["model"]

        # get expression threshold for each samples
        expr_tol_dict, nexpr_tol_dict = {}, {}
        for sample in data.columns:
            expr_tol_dict[sample], nexpr_tol_dict[sample] = self.threshold(data=data[sample], sample_name=sample)

        discreted_df = self.disc(data=data, sample_names=data.columns,
                                 expr_threshold_dic=expr_tol_dict,
                                 non_expr_threshold_dic=nexpr_tol_dict)


class CORDA(Pipeline):
    def __init__(self):
        super().__init__()


class mCARDRE(Pipeline):
    def __init__(self):
        super().__init__()


class SwiftCore(Pipeline):
    def __init__(self):
        super().__init__()

