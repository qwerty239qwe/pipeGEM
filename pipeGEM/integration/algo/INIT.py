from typing import Literal

import numpy as np
import pandas as pd

from pipeGEM.integration.algo.iMAT import add_iMAT_cons_to_model, get_ind_var_for_rxns
from pipeGEM.integration.utils import *
from pipeGEM.analysis import timing, INIT_Analysis


def calc_INIT_weight(rxn_scores,
                     exp_th=None,
                     non_exp_th=None,
                     method: Literal["default", "threshold"] = "default"
                     ):
    if method == "default":
        return {r: 5 * np.log(v) for r, v in rxn_scores.items()}
    elif method == "threshold":
        z = np.polyfit([non_exp_th, exp_th], [0, 20], 1)
        p = np.poly1d(z)
        return {r: p(v) for r, v in rxn_scores.items()}


@timing
def apply_INIT(model,
               data,
               predefined_threshold,
               threshold_kws: dict,
               protected_rxns=None,
               eps=1.,
               tol=1e-8,
               weight_method: Literal["default", "threshold"] = "threshold",
               rxn_scaling_coefs: dict = None,) -> INIT_Analysis:
    gene_data, rxn_scores = data.gene_data, data.rxn_scores
    if weight_method == "threshold":
        threshold_dic = parse_predefined_threshold(predefined_threshold,
                                                   gene_data=gene_data,
                                                   **threshold_kws)
        th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]
    else:
        th_result = None
        non_exp_th = 0
        exp_th = 0

    weight_dic = calc_INIT_weight(rxn_scores=rxn_scores,
                                  exp_th=exp_th, non_exp_th=non_exp_th, method=weight_method)
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
        new_objs[var] = abs(weight_dic[name[4:]])
    for name, var in core_f_ind_vars.items():
        new_objs[var] = weight_dic[name[4:]]
    for name, var in core_b_ind_vars.items():
        new_objs[var] = weight_dic[name[4:]]

    add_iMAT_cons_to_model(model=model,
                           core_f_ind_vars=core_f_ind_vars,
                           core_b_ind_vars=core_b_ind_vars,
                           non_core_ind_vars=non_core_ind_vars,
                           core_rxn_ids=core_rxn_ids,
                           core_rxn_lbs=core_rxn_lbs,
                           core_rxn_ubs=core_rxn_ubs,
                           non_core_rxn_ids=non_core_rxn_ids,
                           non_core_rxn_lbs=non_core_rxn_lbs,
                           non_core_rxn_ubs=non_core_rxn_ubs,
                           eps=eps)

    sol = model.optimize()

    fluxes = abs(sol.to_frame()["fluxes"])
    if rxn_scaling_coefs is not None:
        tol_ = pd.Series({rxn_scaling_coefs[r] * tol for r in fluxes.index}, index=fluxes.index)
    else:
        tol_ = tol

    removed_rxn_ids = (fluxes > tol_).index.to_list()
    result_model.remove_reactions(removed_rxn_ids, remove_orphans=True)

    result = INIT_Analysis(log=dict(threshold_kws=threshold_kws,
                                    protected_rxns=protected_rxns,
                                    eps=eps,
                                    tol=tol,
                                    weight_method=weight_method))

    result.add_result(dict(model=result_model,
                           removed_rxns=np.array(removed_rxn_ids),
                           threshold_analysis=th_result if weight_method == "threshold" else None,
                           weight_dic=weight_dic))

    return result