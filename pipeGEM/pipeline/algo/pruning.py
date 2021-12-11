from pipeGEM.pipeline import Pipeline
from pipeGEM.integration.algo.fastcore import fastCore
from pipeGEM.integration.algo.swiftcore import swiftCore
from pipeGEM.pipeline.algo import FastCC, SwiftCC
from pipeGEM.pipeline.threshold import BimodalThreshold
from pipeGEM.pipeline.preprocessing import GeneDataDiscretizer, GeneDataLinearScaler
from ..task import ReactionTester
from ..model import MediumConstraint
from pipeGEM.integration.mapping import Expression
from pipeGEM.utils.transform import log_xplus1
from pipeGEM.utils import get_rxns_in_subsystem


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
    def __init__(self,
                 consist_method,
                 task_file_path,
                 task_constr_name,
                 model_compartment_format):
        super().__init__()
        if consist_method == "fastcc":
            self.consist_cc = FastCC()
        elif consist_method == "swiftcc":
            self.consist_cc = SwiftCC()
        elif consist_method is None:
            self.consist_cc = None
        self.threshold = BimodalThreshold()
        self.medium_constr = MediumConstraint()
        self.rxn_tester = ReactionTester(task_file_path=task_file_path,
                                         model_compartment_format=model_compartment_format,
                                         constr_name=task_constr_name)
        self.weight_cal = GeneDataLinearScaler()

    def run(self,
            model,
            data,
            medium = None,
            protected_rxns = None,
            rxn_score_trans = log_xplus1,
            not_penalized_subsystem = None,
            not_penalized_weight = 0.1,
            *args,
            **kwargs):
        if self.consist_cc is not None:
            c_model = self.consist_cc(model,
                                      return_model=True,
                                      **kwargs)["model"]
        else:
            c_model = model
        # get expression threshold for each samples
        expr_tol_dict, nexpr_tol_dict = {}, {}
        if isinstance(medium, list):
            self.medium_constr.run(c_model, medium, protected_rxns)
        rxn_weight_dic = {}
        model_dic = {}
        non_penalized = get_rxns_in_subsystem(c_model, not_penalized_subsystem) \
            if not_penalized_subsystem is not None else []
        for sample in data.columns:
            expr_tol_dict[sample], nexpr_tol_dict[sample] = self.threshold(data=data[sample],
                                                                           sample_name=sample)
            if isinstance(medium, dict):
                self.medium_constr.run(c_model, medium[sample], protected_rxns)
            rxn_scores = Expression(c_model, data[sample], rxn_score_trans).rxn_scores
            core_rxns, _ = self.rxn_tester.run(expression_threshold = expr_tol_dict[sample],
                                               non_expression_threshold = nexpr_tol_dict[sample],
                                               rxn_scores = rxn_scores,
                                               ref_model = c_model)
            weights = self.weight_cal.run(data=rxn_scores,
                                          domain_lb=nexpr_tol_dict[sample],
                                          domain_ub=expr_tol_dict[sample],
                                          range_lb=1,
                                          range_ub=0,
                                          range_nan=not_penalized_weight)

            for c in core_rxns + protected_rxns:
                weights[c] = 0
            for r in non_penalized:
                weights[r] = min(not_penalized_weight, weights[r])

            rxn_weight_dic[sample] = weights
            model_dic[sample] = swiftCore(c_model, [], weights)

        return model_dic