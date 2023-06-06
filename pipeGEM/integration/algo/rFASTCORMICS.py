from typing import Literal
from pipeGEM.analysis import rFASTCORMICSAnalysis, timing, FastCCAnalysis, consistency_testers
from pipeGEM.utils import get_rxns_in_subsystem
from pipeGEM.integration.algo.FASTCORE import apply_FASTCORE
from pipeGEM.integration.utils import parse_predefined_threshold


@timing
def apply_rFASTCORMICS(model,
                       data,
                       protected_rxns,
                       predefined_threshold = None,
                       consistent_checking_method: Literal["FASTCC"] = "FASTCC",
                       unpenalized_subsystem = "Transport.*",
                       use_heuristic_th: bool = False,
                       method: str = "onestep",
                       threshold: float = 1e-6):
    gene_data, rxn_scores = data.gene_data, data.rxn_scores
    threshold_dic = parse_predefined_threshold(predefined_threshold,
                                               gene_data=gene_data,
                                               use_heuristic_th=use_heuristic_th)
    th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]

    if consistent_checking_method is not None:
        consistency_tester = consistency_testers[consistent_checking_method](model=model)
        consistency_tester.analyze(tol=threshold)
        model = consistency_tester.consistent_model

    rxn_in_model = set([r.id for r in model.reactions])
    protected_rxns = protected_rxns if protected_rxns is not None else []
    non_core_rxns = (set([r for r, c in rxn_scores.items() if c < non_exp_th]) - set(protected_rxns)) & rxn_in_model
    unpenalized_rxns = set(get_rxns_in_subsystem(model, unpenalized_subsystem)) & rxn_in_model
    non_core_rxns = non_core_rxns - unpenalized_rxns

    pr_result_obj = rFASTCORMICSAnalysis(log={"name": model.name,
                                             "unpenalized_subsystem": unpenalized_subsystem,
                                             "use_heuristic_th": use_heuristic_th,
                                             "method": method,
                                             "threshold": threshold})
    old_exchange_bounds = {r.id: r.bounds for r in model.exchanges}
    for r in model.exchanges:
        r.lower_bound = min(r.lower_bound, -1000) if r.lower_bound < 0 else r.lower_bound
        r.upper_bound = max(r.upper_bound, 1000) if r.upper_bound > 0 else r.upper_bound
    if method == "onestep":
        core_rxns = (set([r for r, c in rxn_scores.items() if c > exp_th]) | set(protected_rxns)) & rxn_in_model
        unpenalized_rxns = unpenalized_rxns - core_rxns
        pr_result = apply_FASTCORE(C=core_rxns,
                                   nonP=unpenalized_rxns,
                                   model=model,
                                   epsilon=threshold,
                                   return_model=True)
        for r in pr_result.result_model.exchanges:
            if r.id in old_exchange_bounds:
                pr_result.result_model.reactions.get_by_id(r.id).bounds = old_exchange_bounds[r.id]

        pr_result_obj.add_result(dict(fastcore_result=pr_result,
                                      core_rxns=core_rxns,
                                      noncore_rxns=non_core_rxns,
                                      nonP_rxns=unpenalized_rxns,
                                      threshold_analysis=th_result))
    elif method == "twostep":
        core_rxns = (set([r for r, c in rxn_scores.items() if c > exp_th]) - set(protected_rxns)) & rxn_in_model

        pr_result = apply_FASTCORE(C=set(protected_rxns),
                                   nonP=core_rxns,
                                   model=model,
                                   epsilon=threshold,
                                   return_model=False)

        core_rxns |= set(pr_result.rxn_ids)
        unpenalized_rxns = unpenalized_rxns - core_rxns

        for r in rxn_in_model - core_rxns - unpenalized_rxns:
            model.reactions.get_by_id(r).bounds = (0, 0)
        consistency_tester = consistency_testers[consistent_checking_method](model=model)
        consistency_tester.analyze(tol=threshold)
        model = consistency_tester.consistent_model
        pr_result = apply_FASTCORE(C=core_rxns,
                                   nonP=unpenalized_rxns,
                                   model=model,
                                   epsilon=threshold,
                                   return_model=False)
        pr_result_obj.add_result(dict(fastcore_result=pr_result,
                                      core_rxns=core_rxns,
                                      noncore_rxns=non_core_rxns,
                                      nonP_rxns=unpenalized_rxns,
                                      threshold_analysis=th_result))

    return pr_result_obj