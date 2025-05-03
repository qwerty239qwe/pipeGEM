import numpy as np
import pandas as pd
from pipeGEM.integration.utils import *
from pipeGEM.analysis import timing, iMAT_Analysis


def get_ind_var_for_rxns(model, core_rxn_ids, non_core_rxn_ids):
    core_f_ind_vars = build_vars_dict_from_rxns(model=model,
                                                rxn_ids=core_rxn_ids,
                                                prefix="cfi_",
                                                lbs=0, ubs=1, type_ab="b")

    core_b_ind_vars = build_vars_dict_from_rxns(model=model,
                                                rxn_ids=core_rxn_ids,
                                                prefix="cbi_",
                                                lbs=0, ubs=1, type_ab="b")

    non_core_ind_vars = build_vars_dict_from_rxns(model=model,
                                                  rxn_ids=non_core_rxn_ids,
                                                  prefix="nci_",
                                                  lbs=0, ubs=1, type_ab="b")
    return core_f_ind_vars, core_b_ind_vars, non_core_ind_vars


def add_iMAT_cons_to_model(model,
                           core_f_ind_vars,
                           core_b_ind_vars,
                           non_core_ind_vars,
                           core_rxn_ids,
                           core_rxn_lbs,
                           core_rxn_ubs,
                           non_core_rxn_ids,
                           non_core_rxn_lbs,
                           non_core_rxn_ubs,
                           eps=1,
                           use_gurobi=False):
    # add_cons_to_model(model, {f"cfi_{ri}": {core_f_ind_vars[f"cfi_{ri}"]: core_rxn_lbs[ri] - eps,
    #                                         model.reactions.get_by_id(ri).forward_variable: 1,
    #                                         #model.reactions.get_by_id(ri).reverse_variable: -1,
    #                                         } for ri in core_rxn_ids},
    #                   prefix="",
    #                   lbs=[core_rxn_lbs[ri] for ri in core_rxn_ids], ubs=np.inf)
    # add_cons_to_model(model, {f"cbi_{ri}": {core_b_ind_vars[f"cbi_{ri}"]: core_rxn_ubs[ri] + eps,
    #                                         #model.reactions.get_by_id(ri).forward_variable: 1,
    #                                         model.reactions.get_by_id(ri).reverse_variable: -1,
    #                                         } for ri in core_rxn_ids},
    #                   prefix="",
    #                   lbs=-np.inf, ubs=[core_rxn_ubs[ri] for ri in core_rxn_ids])

    # add_cons_to_model(model, {f"ncbi_{ri}": {non_core_ind_vars[f"nci_{ri}"]: non_core_rxn_lbs[ri],
    #                                          #model.reactions.get_by_id(ri).forward_variable: 1,
    #                                          model.reactions.get_by_id(ri).reverse_variable: -1,
    #                                          } for ri in non_core_rxn_ids},
    #                   prefix="",
    #                   lbs=[non_core_rxn_lbs[ri] for ri in non_core_rxn_ids], ubs=np.inf)
    # add_cons_to_model(model, {f"ncfi_{ri}": {non_core_ind_vars[f"nci_{ri}"]: non_core_rxn_ubs[ri],
    #                                          #model.reactions.get_by_id(ri).forward_variable: 1,
    #                                          #model.reactions.get_by_id(ri).reverse_variable: -1,
    #                                          } for ri in non_core_rxn_ids},
    #                   prefix="",
    #                   lbs=-np.inf, ubs=[non_core_rxn_ubs[ri] for ri in non_core_rxn_ids])

    success = add_cons_to_model(model, {f"cfi_{ri}": {model.reactions.get_by_id(ri).forward_variable: 1,
                                            model.reactions.get_by_id(ri).reverse_variable: -1,
                                            } for ri in core_rxn_ids},
                      prefix="",
                      lbs=eps, ubs=np.inf,
                      binary_vars=[core_f_ind_vars[f"cfi_{ri}"] for ri in core_rxn_ids],
                      bin_active_val=1,
                      use_gurobi=use_gurobi)

    if not success:
        return False

    add_cons_to_model(model, {f"cbi_{ri}": {model.reactions.get_by_id(ri).reverse_variable: 1,
                                            model.reactions.get_by_id(ri).forward_variable: -1,
                                            } for ri in core_rxn_ids},
                      prefix="",
                      lbs=eps, ubs=np.inf,
                      binary_vars=[core_b_ind_vars[f"cbi_{ri}"] for ri in core_rxn_ids],
                      bin_active_val=1,
                      use_gurobi=use_gurobi)

    add_cons_to_model(model, {f"nci_{ri}": {model.reactions.get_by_id(ri).forward_variable: 1,
                                            model.reactions.get_by_id(ri).reverse_variable: -1,
                                            } for ri in non_core_rxn_ids},
                      prefix="",
                      lbs=0, ubs=0,
                      binary_vars=[non_core_ind_vars[f"nci_{ri}"] for ri in non_core_rxn_ids],
                      bin_active_val=1,
                      use_gurobi=use_gurobi)

    model.solver.update()
    return True


@timing
def apply_iMAT(model,
               data,
               predefined_threshold,
               threshold_kws: dict,
               protected_rxns = None,
               rxn_scaling_coefs=None,
               eps = 1e-6,
               tol = 1e-6,
               use_gurobi=False) -> iMAT_Analysis:
    """Apply the iMAT algorithm to generate a context-specific metabolic model.

    iMAT (integrative Metabolic Analysis Tool) uses gene expression data to
    classify reactions into high-confidence (core) and low-confidence (non-core)
    sets. It then solves a mixed-integer linear programming (MILP) problem to
    find a flux distribution that maximizes activity through core reactions while
    minimizing activity through non-core reactions. Reactions with near-zero
    flux in the optimal solution are removed.

    Parameters
    ----------
    model : cobra.Model
        The input genome-scale metabolic model.
    data : object
        An object containing gene expression data (`data.gene_data`) and
        reaction scores (`data.rxn_scores`) derived from it.
    predefined_threshold : dict or analysis_types
        Strategy or dictionary defining thresholds (`exp_th`, `non_exp_th`) to
        classify reactions based on scores. See
        `pipeGEM.integration.utils.parse_predefined_threshold`.
    threshold_kws : dict
        Additional keyword arguments for the thresholding function.
    protected_rxns : list[str], optional
        A list of reaction IDs that should always be treated as high-confidence
        (core) and potentially weighted higher in the objective. Defaults to None.
    rxn_scaling_coefs : dict[str, float], optional
        Dictionary mapping reaction IDs to scaling coefficients. Currently unused
        in the main logic but potentially used for tolerance adjustment.
        Defaults to None.
    eps : float, optional
        Small flux value used in constraints to enforce activity through core
        reactions selected by the MILP. Defaults to 1e-6.
    tol : float, optional
        Flux tolerance threshold. Reactions with absolute flux below this value
        in the MILP solution are removed from the final model. Defaults to 1e-6.
    use_gurobi : bool, optional
        If True, use Gurobi-specific indicator constraints for potentially better
        performance. Requires Gurobi solver. Defaults to False.

    Returns
    -------
    iMAT_Analysis
        An object containing the results:
        - result_model (cobra.Model): The final context-specific model.
        - removed_rxn_ids (np.ndarray): IDs of removed reactions.
        - threshold_analysis (ThresholdAnalysis): Details of thresholding used.

    Notes
    -----
    Based on the algorithm described in: Shlomi, T., Cabili, M. N., Herrgård, M. J.,
    Palsson, B. Ø., & Ruppin, E. (2008). Network-based prediction of human
    tissue-specific metabolism. Nature biotechnology, 26(9), 1003-1010.
    The implementation uses binary indicator variables to control reaction activity.
    """
    gene_data, rxn_scores = data.gene_data, data.rxn_scores
    threshold_dic = parse_predefined_threshold(predefined_threshold,
                                               gene_data=gene_data,
                                               **threshold_kws)
    th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]
    model = model.copy()
    result_model = model.copy()
    core_rxn_ids = [r.id for r in model.reactions if rxn_scores[r.id] >= exp_th]
    non_core_rxn_ids = [r.id for r in model.reactions if rxn_scores[r.id] <= non_exp_th]
    core_rxn_ids = list(set(core_rxn_ids) | set(protected_rxns))
    non_core_rxn_ids = list(set(non_core_rxn_ids) - set(protected_rxns))

    core_rxn_lbs = {r: model.reactions.get_by_id(r).lower_bound for r in core_rxn_ids}
    core_rxn_ubs = {r: model.reactions.get_by_id(r).upper_bound for r in core_rxn_ids}

    non_core_rxn_lbs = {r: model.reactions.get_by_id(r).lower_bound for r in non_core_rxn_ids}
    non_core_rxn_ubs = {r: model.reactions.get_by_id(r).upper_bound for r in non_core_rxn_ids}

    model.objective.set_linear_coefficients({v: 0 for v in model.variables})
    new_objs = {}

    core_f_ind_vars, core_b_ind_vars, non_core_ind_vars = get_ind_var_for_rxns(model=model,
                                                                               core_rxn_ids=core_rxn_ids,
                                                                               non_core_rxn_ids=non_core_rxn_ids)

    for name, var in non_core_ind_vars.items():
        new_objs[var] = 0
    for name, var in core_f_ind_vars.items():
        new_objs[var] = 1
        if name[4:] in protected_rxns:
            print(name)
            new_objs[var] = 1000
    for name, var in core_b_ind_vars.items():
        new_objs[var] = 1
        if name[4:] in protected_rxns:
            print(name)
            new_objs[var] = 1000

    cons_added = add_iMAT_cons_to_model(model=model,
                           core_f_ind_vars=core_f_ind_vars,
                           core_b_ind_vars=core_b_ind_vars,
                           non_core_ind_vars=non_core_ind_vars,
                           core_rxn_ids=core_rxn_ids,
                           core_rxn_lbs=core_rxn_lbs,
                           core_rxn_ubs=core_rxn_ubs,
                           non_core_rxn_ids=non_core_rxn_ids,
                           non_core_rxn_lbs=non_core_rxn_lbs,
                           non_core_rxn_ubs=non_core_rxn_ubs,
                           eps=eps,
                           use_gurobi=use_gurobi)
    if not cons_added:
        # return an empty result
        result = iMAT_Analysis(log=dict(threshold_kws=threshold_kws,
                                        protected_rxns=protected_rxns,
                                        eps=eps,
                                        tol=tol))
        return result

    model.objective.set_linear_coefficients(new_objs)
    model.solver.update()
    # print(model.objective)
    sol = model.optimize("maximize")
    # print(sol.objective_value)
    # print(non_core_rxn_ubs)
    # print(non_core_rxn_lbs)
    # print(model.solver.primal_values)
    fluxes = abs(sol.to_frame()["fluxes"])
    if rxn_scaling_coefs is not None:
        tol_ = pd.Series({rxn_scaling_coefs[r] * tol for r in fluxes.index}, index=fluxes.index)
    else:
        tol_ = tol

    # print(fluxes)
    removed_rxn_ids = fluxes[fluxes < tol_].index.to_list()
    result_model.remove_reactions(removed_rxn_ids, remove_orphans=True)

    result = iMAT_Analysis(log=dict(threshold_kws=threshold_kws,
                                    protected_rxns=protected_rxns,
                                    eps=eps,
                                    tol=tol))
    result.add_result(dict(result_model=result_model,
                           removed_rxn_ids=np.array(removed_rxn_ids),
                           threshold_analysis=th_result))
    return result
