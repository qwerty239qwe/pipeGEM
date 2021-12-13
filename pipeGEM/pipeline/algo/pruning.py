import numpy as np
import seaborn as sns

from pipeGEM.pipeline import Pipeline
from pipeGEM.integration.algo.fastcore import (fastCore, get_C_and_P_dic,
                                               supp_protected_rxns, get_unpenalized_rxn_ids,
                                               get_consistent_p_free_model,
                                               get_final_models)
from pipeGEM.integration.algo.swiftcore import swiftCore
from pipeGEM.pipeline.algo import FastCC, SwiftCC
from pipeGEM.pipeline.threshold import BimodalThreshold
from pipeGEM.pipeline.preprocessing import GeneDataDiscretizer, GeneDataLinearScaler
from ..task import ReactionTester
from ..model import MediumConstraint
from pipeGEM.integration.mapping import Expression
from pipeGEM.utils import get_rxns_in_subsystem


class ReactionCategorizer(Pipeline):
    def __init__(self,
                 method="get_CP"):
        super().__init__()
        if method == "get_CP":
            self.categorizer = get_C_and_P_dic

    def run(self, *args, **kwargs):
        self.output = self.categorizer(**kwargs)


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
                 consist_method,
                 task_file_path,
                 task_constr_name,
                 model_compartment_format,
                 core_threshold=1e-4,
                 cc_threshold=1e-4):
        super().__init__()
        if consist_method == "fastcc":
            self.consist_cc = FastCC()
        elif consist_method == "swiftcc":
            self.consist_cc = SwiftCC()
        self.core_threshold = core_threshold
        self.cc_threshold = cc_threshold
        self.threshold = BimodalThreshold()
        self.medium_constr = MediumConstraint()
        self.rxn_tester = ReactionTester(task_file_path=task_file_path,
                                         model_compartment_format=model_compartment_format,
                                         constr_name=task_constr_name)
        self.disc = GeneDataDiscretizer()
        self.rxn_categorizer = ReactionCategorizer()

    def run(self,
            model,
            data,
            medium=None,
            protected_rxns=None,
            rxn_score_trans=np.log2,
            not_penalized_subsystem=None,
            *args,
            **kwargs):
        # get consistent model
        c_model = self.consist_cc(model,
                                  return_model=True,
                                  **kwargs)["model"]

        # get expression threshold for each samples
        expr_tol_dict, nexpr_tol_dict = {}, {}

        if isinstance(medium, list):
            self.medium_constr.run(c_model, medium, protected_rxns)

        task_protected_rxn_dic = {}
        for sample in data.columns:
            expr_tol_dict[sample], nexpr_tol_dict[sample] = self.threshold(data=rxn_score_trans(data[sample]),
                                                                           sample_name=sample)
            if isinstance(medium, dict):
                self.medium_constr.run(c_model, medium[sample], protected_rxns)
            rxn_scores = Expression(c_model, data[sample], rxn_score_trans).rxn_scores
            core_rxns, _ = self.rxn_tester.run(expression_threshold=expr_tol_dict[sample],
                                               non_expression_threshold=nexpr_tol_dict[sample],
                                               rxn_scores=rxn_scores,
                                               ref_model=c_model,
                                               reset_tester=False)
            task_protected_rxn_dic[sample] = core_rxns + protected_rxns
        discreted_df = self.disc(data=data,
                                 sample_names=data.columns,
                                 expr_threshold_dic=expr_tol_dict,
                                 non_expr_threshold_dic=nexpr_tol_dict)
        C_P_dics = self.rxn_categorizer(expression_dic=discreted_df,
                                        sample_names=data.columns,
                                        consensus_proportion=0.9)
        self.supp_dic = supp_protected_rxns(c_model,
                                            data.columns,
                                            task_protected_rxn_dic,
                                            C_dic=C_P_dics["C_dic"],
                                            P_dic=C_P_dics["P_dic"],
                                            epsilon_for_fastcore=self.core_threshold)
        self.non_penalized_dic = get_unpenalized_rxn_ids(c_model,
                                                         C_dic=self.supp_dic["C_dic"],
                                                         P_dic=self.supp_dic["P_dic"],
                                                         sample_names=data.columns,
                                                         unpenalized_subsystem=not_penalized_subsystem)
        self.p_model_dic = get_consistent_p_free_model(c_model,
                                                       data.columns,
                                                       C_dic=self.supp_dic["C_dic"],
                                                       P_dic=self.non_penalized_dic["P_dic"],
                                                       epsilon_for_fastcc=self.cc_threshold)
        self.output = get_final_models(c_model,
                                       sample_names=data.columns,
                                       C_dic=self.p_model_dic["C_dic"],
                                       P_dic=self.non_penalized_dic["P_dic"],
                                       unpenalized_rxn_dic=self.non_penalized_dic["unpenalized_rxn_dic"],
                                       p_free_model_dic=self.p_model_dic["p_free_model_dic"],
                                       epsilon_for_fastcore=self.core_threshold)
        return self.output

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
                 model_compartment_format,
                 saved_dist_plot_format = None,):
        super().__init__()
        if consist_method == "fastcc":
            self.consist_cc = FastCC()
        elif consist_method == "swiftcc":
            self.consist_cc = SwiftCC()
        elif consist_method is None:
            self.consist_cc = None
        self.threshold = BimodalThreshold(naming_format=saved_dist_plot_format)
        self.medium_constr = MediumConstraint()
        self.rxn_tester = ReactionTester(task_file_path=task_file_path,
                                         model_compartment_format=model_compartment_format,
                                         constr_name=task_constr_name)
        self.weight_cal = GeneDataLinearScaler()

    def run(self,
            model,
            data,
            tissue_data = None,
            medium = None,
            protected_rxns = None,
            rxn_score_trans = np.log2,
            not_penalized_subsystem = None,
            not_penalized_weight = 0.1,
            tissue_u_weight = 0.2,
            tissue_l_weight = 0.8,
            *args,
            **kwargs):
        self.output = {}
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
        ts_score = Expression(c_model, tissue_data).rxn_scores \
            if tissue_data is not None else {}

        non_penalized = get_rxns_in_subsystem(c_model, not_penalized_subsystem) \
            if not_penalized_subsystem is not None else []
        for sample in data.columns:
            expr_tol_dict[sample], nexpr_tol_dict[sample] = self.threshold(data=rxn_score_trans(data[sample]),
                                                                           sample_name=sample)
            if isinstance(medium, dict):
                self.medium_constr.run(c_model, medium[sample], protected_rxns)
            rxn_scores = Expression(c_model, data[sample], rxn_score_trans).rxn_scores
            core_rxns, _ = self.rxn_tester.run(expression_threshold = expr_tol_dict[sample],
                                               non_expression_threshold = nexpr_tol_dict[sample],
                                               rxn_scores = rxn_scores,
                                               ref_model = c_model,
                                               reset_tester=False)
            weights = self.weight_cal.run(data=rxn_scores,
                                          domain_lb=nexpr_tol_dict[sample],
                                          domain_ub=expr_tol_dict[sample],
                                          range_lb=1,
                                          range_ub=0,
                                          range_nan=not_penalized_weight)

            print(f"Core rxns: {len(core_rxns)}; protected rxns: {len(protected_rxns)} (might be overlapped with core)")
            print(f"non_penalized rxns: {len(non_penalized)}")
            for c in core_rxns + protected_rxns:
                weights[c] = 0
            for r in non_penalized:
                weights[r] = min(not_penalized_weight, weights[r])
            for r, v in ts_score.items():
                if v == 2:
                    weights[r] = 0
                if v == 1:
                    weights[r] = min(weights[r], tissue_u_weight)
                if v == 0:
                    weights[r] = max(weights[r], tissue_l_weight)
                if v == -1:
                    weights[r] = 1
            sns.histplot(list(weights.values()))
            rxn_weight_dic[sample] = weights
            self.output[sample] = swiftCore(c_model, [], weights)

        return self.output