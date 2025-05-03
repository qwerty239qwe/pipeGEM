from typing import Literal, List, Union, Optional

import cobra

from pipeGEM.analysis import rFASTCORMICSAnalysis, timing, consistency_testers
from pipeGEM.utils import get_rxns_in_subsystem
from pipeGEM.integration.algo.FASTCORE import apply_FASTCORE
from pipeGEM.integration.utils import parse_predefined_threshold, analysis_types


@timing
def apply_rFASTCORMICS(model: cobra.Model,
                       data,
                       protected_rxns: List[str] = None,
                       predefined_threshold: Optional[Union[dict, analysis_types]] = None,
                       threshold_kws: dict = None,
                       rxn_scaling_coefs: dict = None,
                       consistent_checking_method: Literal["FASTCC", "FVA"] = "FASTCC",
                       unpenalized_subsystem: Union[str, List[str]] = "Transport.*",
                       method: str = "onestep",
                       threshold: float = 1e-6,
                       FASTCORE_raise_error: bool = False,
                       calc_efficacy: bool = True) -> rFASTCORMICSAnalysis:
    """Apply the rFASTCORMICS algorithm to build a context-specific model.

    Leverages expression data to define core/non-core reaction sets and uses
    FASTCORE to extract a consistent subnetwork. Optionally includes model
    consistency checking and handling of protected reactions and unpenalized
    subsystems.

    Parameters
    ----------
    model : cobra.Model
        Input genome-scale metabolic model.
    data : object
        Object with gene expression data (`data.gene_data`) and reaction
        scores (`data.rxn_scores`).
    protected_rxns : list[str], optional
        Reaction IDs always included in the core set. Defaults to None.
    predefined_threshold : dict or analysis_types, optional
        Strategy or dictionary defining thresholds to classify reactions based
        on scores (e.g., 'percentile_90'). See
        `pipeGEM.integration.utils.parse_predefined_threshold`. Defaults to None.
    threshold_kws : dict, optional
        Additional keyword arguments for the thresholding function. Defaults to None.
    rxn_scaling_coefs : dict, optional
        Mapping of reaction IDs to scaling coefficients to adjust flux thresholds
        in FASTCORE. Defaults to None.
    consistent_checking_method : {'FASTCC', 'FVA'}, optional
        Method to ensure initial model consistency ('FASTCC' or 'FVA').
        Set to None to skip. Defaults to "FASTCC".
    unpenalized_subsystem : str or list[str], optional
        Subsystem name(s) (regex allowed) included in the non-penalty set (nonP)
        during FASTCORE. Defaults to "Transport.*".
    method : {'onestep', 'twostep'}, optional
        rFASTCORMICS variant:
        - 'onestep': Run FASTCORE once with core and non-penalty sets.
        - 'twostep': Run FASTCORE on protected reactions, refine, run again on
                     expanded core set. (May need validation).
        Defaults to "onestep".
    threshold : float, optional
        Flux threshold below which flux is considered zero. Defaults to 1e-6.
    FASTCORE_raise_error : bool, optional
        If True, FASTCORE raises error on inconsistency. If False, warns.
        Defaults to False.
    calc_efficacy : bool, optional
        If True, calculate efficacy metrics based on expression-defined sets.
        Defaults to True.

    Returns
    -------
    rFASTCORMICSAnalysis
        Object containing results: context-specific model (in nested FASTCORE result),
        core/non-core sets, thresholding analysis, efficacy metrics.

    Notes
    -----
    Based on the algorithm described in: Pacheco, M. P., Bintener, T., Ternes, D.,
    Kulik, M., Sauter, T., Sinkkonen, L., & Heinäniemi, M. (2019). rFASTCORMICS:
    A fast and effective reconstruction of context-specific metabolic models.
    PLoS computational biology, 15(10), e1007416.
    """
    gene_data, rxn_scores = data.gene_data, data.rxn_scores
    threshold_dic = parse_predefined_threshold(predefined_threshold,
                                               gene_data=gene_data,
                                               **threshold_kws)
    th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]

    if consistent_checking_method is not None:
        consistency_tester = consistency_testers[consistent_checking_method](model=model)
        cons_result = consistency_tester.analyze(tol=threshold, rxn_scaling_coefs=rxn_scaling_coefs)
        model = cons_result.consistent_model

    model = model.copy()
    rxn_in_model = set([r.id for r in model.reactions])
    protected_rxns = protected_rxns if protected_rxns is not None else []
    non_core_rxns = (set([r for r, c in rxn_scores.items() if c < non_exp_th]) - set(protected_rxns)) & rxn_in_model
    unpenalized_rxns = set(get_rxns_in_subsystem(model, unpenalized_subsystem)) & rxn_in_model
    non_core_rxns = non_core_rxns - unpenalized_rxns

    pr_result_obj = rFASTCORMICSAnalysis(log={"name": model.name,
                                              "unpenalized_subsystem": unpenalized_subsystem,
                                              "method": method,
                                              "threshold": threshold,
                                              **threshold_kws})
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
                                   return_model=True,
                                   copy_model=False,
                                   raise_err=FASTCORE_raise_error,
                                   rxn_scaling_coefs=rxn_scaling_coefs,
                                   calc_efficacy=calc_efficacy)

        for r in pr_result.result_model.exchanges:
            if r.id in old_exchange_bounds:
                pr_result.result_model.reactions.get_by_id(r.id).bounds = old_exchange_bounds[r.id]

        pr_result_obj.add_result(dict(fastcore_result=pr_result,
                                      core_rxns=core_rxns,
                                      noncore_rxns=non_core_rxns,
                                      nonP_rxns=unpenalized_rxns,
                                      threshold_analysis=th_result,
                                      algo_efficacy=pr_result.algo_efficacy))
    elif method == "twostep":
        core_rxns = (set([r for r, c in rxn_scores.items() if c > exp_th]) - set(protected_rxns)) & rxn_in_model

        pr_result = apply_FASTCORE(C=set(protected_rxns),
                                   nonP=core_rxns,
                                   model=model,
                                   epsilon=threshold,
                                   return_model=False,
                                   rxn_scaling_coefs=rxn_scaling_coefs,
                                   calc_efficacy=False)

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
                                   return_model=True,
                                   copy_model=False,
                                   rxn_scaling_coefs=rxn_scaling_coefs)
        pr_result_obj.add_result(dict(fastcore_result=pr_result,
                                      core_rxns=core_rxns,
                                      noncore_rxns=non_core_rxns,
                                      nonP_rxns=unpenalized_rxns,
                                      threshold_analysis=th_result,
                                      algo_efficacy=pr_result.algo_efficacy))

    return pr_result_obj
