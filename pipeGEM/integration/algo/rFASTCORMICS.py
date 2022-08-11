from pipeGEM.analysis import rFastCormicsThreshold, rFastCormicAnalysis, timing, FastCCAnalysis, rFastCormicThresholdAnalysis
from pipeGEM.utils import get_rxns_in_subsystem
from pipeGEM.integration.algo import fastcore
from pipeGEM.integration.algo.fastcore import fastcc


@timing
def apply_rFASTCORMICS(model,
                       data,
                       protected_rxns,
                       predefined_threshold = None,
                       consistent_checking_method = "fastcc",
                       unpenalized_subsystem = "Transport.*",
                       use_heuristic_th: bool = False,
                       method: str = "onestep",
                       threshold: float = 1e-6):
    gene_data, rxn_scores = data.gene_data, data.rxn_scores

    if predefined_threshold is None:
        th_finder = rFastCormicsThreshold()
        th_result = th_finder.find_threshold(gene_data, use_first_guess=use_heuristic_th)
        non_exp_th = th_result.non_exp_th
        exp_th = th_result.exp_th
    else:
        th_result = predefined_threshold
        if isinstance(th_result, rFastCormicThresholdAnalysis):
            non_exp_th = th_result.non_exp_th
            exp_th = th_result.exp_th
        else:
            non_exp_th = th_result["non_exp_th"].exp_th if hasattr(th_result["non_exp_th"], "exp_th") else th_result["non_exp_th"]
            exp_th = th_result["exp_th"].exp_th if hasattr(th_result["exp_th"], "exp_th") else th_result["exp_th"]

    if consistent_checking_method == "fastcc":
        cons_obj = FastCCAnalysis(log={"threshold": threshold})
        fastcc_result = fastcc(model,
                               epsilon=threshold,
                               return_model=True,
                               return_rxn_ids=True,
                               return_removed_rxn_ids=True)
        cons_obj.add_result(result=fastcc_result)
        model = cons_obj.consist_model

    rxn_in_model = set([r.id for r in model.reactions])
    protected_rxns = protected_rxns if protected_rxns is not None else []
    non_core_rxns = (set([r for r, c in rxn_scores.items() if c < non_exp_th]) - set(protected_rxns)) & rxn_in_model
    unpenalized_rxns = set(get_rxns_in_subsystem(model, unpenalized_subsystem)) & rxn_in_model
    non_core_rxns = non_core_rxns - unpenalized_rxns
    core_rxns = (set([r for r, c in rxn_scores.items() if c > exp_th]) | set(protected_rxns)) & rxn_in_model
    unpenalized_rxns = unpenalized_rxns - core_rxns
    pr_result_obj = rFastCormicAnalysis(log={"name": model.name,
                                             "unpenalized_subsystem": unpenalized_subsystem,
                                             "use_heuristic_th": use_heuristic_th,
                                             "method": method,
                                             "threshold": threshold})
    if method == "onestep":
        pr_result = fastcore.fastCore(C=core_rxns,
                                      nonP=unpenalized_rxns,
                                      model=model,
                                      epsilon=threshold,
                                      return_model=True,
                                      return_rxn_ids=True,
                                      return_removed_rxn_ids=True)
        pr_result_obj.add_result(fastcore_result=pr_result,
                                 core_rxns=core_rxns,
                                 noncore_rxns=non_core_rxns,
                                 nonP_rxns=unpenalized_rxns,
                                 threshold_analysis=th_result)

    return pr_result_obj