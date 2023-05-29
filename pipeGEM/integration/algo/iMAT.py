import numpy as np
from pipeGEM.integration.utils import *


def apply_iMAT(model,
               data,
               predefined_threshold,
               threshold_kws: dict,
               eps =1,
               tol=1e-8):
    gene_data, rxn_scores = data.gene_data, data.rxn_scores
    threshold_dic = parse_predefined_threshold(predefined_threshold,
                                               gene_data=gene_data,
                                               **threshold_kws)
    th_result, exp_th, non_exp_th = threshold_dic["th_result"], threshold_dic["exp_th"], threshold_dic["non_exp_th"]
    model = model.copy()
    result_model = model.copy()
    core_rxn_ids = [r.id for r in model.reactions if rxn_scores[r.id] >= exp_th]
    non_core_rxn_ids = [r.id for r in model.reactions if rxn_scores[r.id] <= non_exp_th]
    core_rxn_lbs = [model.reactions.get_by_id(r).lower_bound for r in core_rxn_ids]
    core_rxn_ubs = [model.reactions.get_by_id(r).upper_bound for r in core_rxn_ids]

    non_core_rxn_lbs = [model.reactions.get_by_id(r).lower_bound for r in non_core_rxn_ids]
    non_core_rxn_ubs = [model.reactions.get_by_id(r).upper_bound for r in non_core_rxn_ids]

    model.objective.set_linear_coefficients({v: 0 for v in model.variables})
    new_objs = {}

    core_f_indicator_vars = build_vars_dict_from_rxns(model=model,
                                                      rxn_ids=core_rxn_ids,
                                                      prefix="cfi_",
                                                      lbs=0, ubs=1, type_ab="b")

    core_b_indicator_vars = build_vars_dict_from_rxns(model=model,
                                                      rxn_ids=core_rxn_ids,
                                                      prefix="cbi_",
                                                      lbs=0, ubs=1, type_ab="b")

    non_core_ind_vars = build_vars_dict_from_rxns(model=model,
                                                  rxn_ids=non_core_rxn_ids,
                                                  prefix="nci_",
                                                  lbs=0, ubs=1, type_ab="b")

    for name, var in non_core_ind_vars:
        new_objs[var] = 1
    for name, var in core_f_indicator_vars:
        new_objs[var] = 1
    for name, var in core_b_indicator_vars:
        new_objs[var] = 1

    add_cons_to_model(model, {f"cfi_{ri}": {core_f_indicator_vars[ri]: core_rxn_lbs[ri] - eps,
                                            model.reactions.get_by_id(ri).forward_variable: 1,
                                            model.reactions.get_by_id(ri).reverse_variable: -1,
                                            } for ri in core_rxn_ids},
                      prefix="",
                      lbs=core_rxn_lbs, ubs=np.inf)

    add_cons_to_model(model, {f"cbi_{ri}": {core_b_indicator_vars[ri]: core_rxn_ubs[ri] + eps,
                                            model.reactions.get_by_id(ri).forward_variable: 1,
                                            model.reactions.get_by_id(ri).reverse_variable: -1,
                                            } for ri in core_rxn_ids},
                      prefix="",
                      lbs=-np.inf, ubs=core_rxn_ubs)

    add_cons_to_model(model, {f"ncfi_{ri}": {non_core_ind_vars[ri]: non_core_rxn_lbs[ri],
                                             model.reactions.get_by_id(ri).forward_variable: 1,
                                             model.reactions.get_by_id(ri).reverse_variable: -1,
                                             } for ri in non_core_rxn_ids},
                      prefix="",
                      lbs=non_core_rxn_lbs, ubs=np.inf)

    add_cons_to_model(model, {f"ncbi_{ri}": {non_core_ind_vars[ri]: non_core_rxn_ubs[ri],
                                             model.reactions.get_by_id(ri).forward_variable: 1,
                                             model.reactions.get_by_id(ri).reverse_variable: -1,
                                             } for ri in non_core_rxn_ids},
                      prefix="",
                      lbs=-np.inf, ubs=non_core_rxn_ubs)

    sol = model.optimize()
    removed_rxn_ids = sol.to_frame().query(f"abs(fluxes) < {tol}").index
    result_model.remove_reactions(removed_rxn_ids)