from typing import Dict

from optlang.symbolics import Zero
import cobra
import numpy as np
import pandas as pd

from pipeGEM.utils import make_irrev_rxn, merge_irrevs_in_df
from pipeGEM.integration.constraints import register


@register
def RIPTiDe(model: cobra.Model,
            rxn_expr_score: Dict[str, float],
            max_gw: float = None,
            obj_frac: float = 0.8,
            prune_rxns: bool = False,
            pruning_tol: float = 1e-4,
            return_pruning_coef: bool = False,
            get_details: bool = True):
    """
    RIPTiDe implementation
    Parameters
    ----------
    model: cobra.Model
        The analyzed model, should has set RMF
    rxn_expr_score: Dict[str, float]
        A dict with rxn_ids as keys and expression values as values
    max_gw: Optional, float

    obj_frac: float

    get_details: bool

    Returns
    -------
    None
    """
    forward_prefix, backward_prefix = "_F_", "_R_"
    rev_map_to_irrev = {r.id: [r.id] for r in model.reactions}
    max_gw = max_gw or max(rxn_expr_score.values())
    if max_gw < max(rxn_expr_score.values()):
        raise ValueError("max_gw must be greater than or equal to the max rxn score")
    min_gw = min(rxn_expr_score.values())
    for r in model.reactions:
        if r.reversibility:
            new_rxns = make_irrev_rxn(model,
                                      r.id,
                                      forward_prefix=forward_prefix,
                                      backward_prefix=backward_prefix)
            r.knock_out()
            new_rxns[0].objective_coefficient = r.objective_coefficient
            rev_map_to_irrev[r.id] = [nr.id for nr in new_rxns]
    sol_df = model.optimize().to_frame()
    for r in model.reactions:
        if r.objective_coefficient > 0:
            r.lower_bound = obj_frac * sol_df.loc[r.id, "fluxes"]
    minimized_rs = []

    not_related_rxns = set([r.id for r in model.reactions]) - set(rxn_expr_score.keys())
    if prune_rxns:
        obj_dict = {mapped_r_id: (max_gw + min_gw - r_exp) / max_gw
                    for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp)
                    for mapped_r_id in rev_map_to_irrev[r_id]}
        obj_dict.update({r.id: min_gw / max_gw
                        for r in model.reactions if r.id not in obj_dict})  # same as the smallest weight

        model.objective = model.problem.Objective(Zero, direction="min", sloppy=True)
        model.objective = {
            model.reactions.get_by_id(k): v for k, v in obj_dict.items() if v != 0
        }
        if return_pruning_coef:
            return
        min_sol_df = model.optimize(objective_sense="minimize").to_frame()
        rev_sol_df = merge_irrevs_in_df(min_sol_df, forward_prefix, backward_prefix)
        to_remove = rev_sol_df[rev_sol_df["fluxes"] < pruning_tol].index

        for r in to_remove:
            if r in rev_map_to_irrev:
                minimized_rs.extend(rev_map_to_irrev[r])
            else:
                minimized_rs.append(r)
        print(f"{len(minimized_rs)} reactions are non-core reactions (fluxes smaller than {pruning_tol})")
        # for r in minimized_rs:
        #     model.reactions.get_by_id(r).bounds = (0, 0)
        # model.remove_reactions(to_remove_rs, remove_orphans=True)
    current_rs = [r.id for r in model.reactions]

    obj_dict = {mapped_r_id: r_exp / max_gw if mapped_r_id not in minimized_rs else (-max_gw - min_gw + r_exp) / max_gw
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp)
                for mapped_r_id in rev_map_to_irrev[r_id]}
    obj_dict.update({r.id: 1
                     for r in model.reactions if r.id not in obj_dict})

    model.objective = model.problem.Objective(Zero, direction="max", sloppy=True)
    model.objective = {
        model.reactions.get_by_id(k): v for k, v in obj_dict.items() if v != 0 and k in current_rs
    }

    max_sol_df = model.optimize(objective_sense="maximize").to_frame()
    for i, row in max_sol_df.iterrows():
        model.reactions.get_by_id(i).upper_bound = row["fluxes"]

    if get_details:
        return obj_dict