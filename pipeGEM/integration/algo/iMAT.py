import numpy as np
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
                           eps=1):
    add_cons_to_model(model, {f"cfi_{ri}": {core_f_ind_vars[f"cfi_{ri}"]: core_rxn_lbs[ri] - eps,
                                            model.reactions.get_by_id(ri).forward_variable: 1,
                                            model.reactions.get_by_id(ri).reverse_variable: -1,
                                            } for ri in core_rxn_ids},
                      prefix="",
                      lbs=-np.inf, ubs=[core_rxn_ubs[ri] for ri in core_rxn_ids])

    add_cons_to_model(model, {f"cbi_{ri}": {core_b_ind_vars[f"cbi_{ri}"]: core_rxn_ubs[ri] + eps,
                                            model.reactions.get_by_id(ri).forward_variable: 1,
                                            model.reactions.get_by_id(ri).reverse_variable: -1,
                                            } for ri in core_rxn_ids},
                      prefix="",
                      lbs=[core_rxn_lbs[ri] for ri in core_rxn_ids], ubs=np.inf)

    add_cons_to_model(model, {f"ncfi_{ri}": {non_core_ind_vars[f"nci_{ri}"]: non_core_rxn_lbs[ri],
                                             model.reactions.get_by_id(ri).forward_variable: 1,
                                             model.reactions.get_by_id(ri).reverse_variable: -1,
                                             } for ri in non_core_rxn_ids},
                      prefix="",
                      lbs=-np.inf, ubs=[non_core_rxn_ubs[ri] for ri in non_core_rxn_ids])

    add_cons_to_model(model, {f"ncbi_{ri}": {non_core_ind_vars[f"nci_{ri}"]: non_core_rxn_ubs[ri],
                                             model.reactions.get_by_id(ri).forward_variable: 1,
                                             model.reactions.get_by_id(ri).reverse_variable: -1,
                                             } for ri in non_core_rxn_ids},
                      prefix="",
                      lbs=[non_core_rxn_lbs[ri] for ri in non_core_rxn_ids], ubs=np.inf)


@timing
def apply_iMAT(model,
               data,
               predefined_threshold,
               threshold_kws: dict,
               protected_rxns = None,
               rxn_scaling_coefs=None,
               eps = 1.,
               tol = 1e-8) -> iMAT_Analysis:
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
        new_objs[var] = 1
    for name, var in core_f_ind_vars.items():
        new_objs[var] = 1
    for name, var in core_b_ind_vars.items():
        new_objs[var] = 1

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

    result = iMAT_Analysis(log=dict(threshold_kws=threshold_kws,
                                    protected_rxns=protected_rxns,
                                    eps=eps,
                                    tol=tol))
    result.add_result(dict(result_model=result_model,
                           removed_rxn_ids=np.array(removed_rxn_ids),
                           threshold_analysis=th_result))
    return result