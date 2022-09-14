from typing import Dict

from optlang.symbolics import Zero
import cobra
from cobra.util import fix_objective_as_constraint
import numpy as np
import pandas as pd

from pipeGEM.analysis import add_mod_pfba, add_norm_constraint, SPOTAnalysis


def apply_SPOT(model: cobra.Model,
               rxn_expr_score: Dict[str, float],
               protected_rxns = None,
               remove_zero_fluxes: bool = False,
               flux_threshold: float = 1e-6,
               return_fluxes: bool = True,
               keep_context: bool = False
               ):
    """
    GIMME implementation

    Parameters
    ----------
    model: cobra.Model
        A model with objective function
    rxn_expr_score: Dict[str, float]
        A dict with rxn_ids as keys and expression values as values
    Returns
    -------
    None
    """
    obj_dict = {r_id: r_exp
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp) and
                r_id not in protected_rxns}
    with model:
        add_norm_constraint(model)
        add_mod_pfba(model, weights=obj_dict, fraction_of_optimum=0, direction="max")
        sol = model.optimize("maximize")

    flux_df = sol.to_frame()
    new_model = None
    if remove_zero_fluxes:
        new_model = model.copy()
        to_remove = set(flux_df[abs(flux_df["fluxes"]) <= flux_threshold].index.to_list()) - set(protected_rxns)
        new_model.remove_reactions(list(to_remove), remove_orphans=True)

    if keep_context:
        rxns_in_model = [r.id for r in model.reactions]
        add_norm_constraint(model)
        add_mod_pfba(model, weights={k: v for k, v in obj_dict.items() if k in rxns_in_model},
                     fraction_of_optimum=0) # some are probably removed

    result = SPOTAnalysis(log={"name": model.name})
    result.add_result(rxn_expr_score,
                      fluxes=flux_df if return_fluxes else None,
                      model=new_model)
    return result
