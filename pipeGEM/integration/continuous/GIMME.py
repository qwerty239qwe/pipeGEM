from typing import Dict

from optlang.symbolics import Zero
import cobra
from cobra.util import fix_objective_as_constraint
import numpy as np
import pandas as pd

from pipeGEM.analysis import add_mod_pfba, GIMMEAnalysis, timing


@timing
def apply_GIMME(model: cobra.Model,
                rxn_expr_score: Dict[str, float],
                high_exp: float,
                protected_rxns = None,
                obj_frac: float = 0.8,
                remove_zero_fluxes: bool = False,
                flux_threshold: float = 1e-6,
                max_inconsistency_score = 1e3,
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
    high_exp: float
        Expression value higher than this value is treated as high_exp
    obj_frac: float

    Returns
    -------
    None
    """
    obj_dict = {r_id: (high_exp - r_exp)
                if high_exp - r_exp < max_inconsistency_score else max_inconsistency_score  # this is for preventing using -np.inf values
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp) and
                r_id not in protected_rxns and
                r_exp < high_exp}
    with model:
        add_mod_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac)
        sol = model.optimize("minimize")

    flux_df = sol.to_frame()
    new_model = None
    if remove_zero_fluxes:
        new_model = model.copy()
        to_remove = set(flux_df[abs(flux_df["fluxes"]) <= flux_threshold].index.to_list()) - set(protected_rxns)
        new_model.remove_reactions(list(to_remove), remove_orphans=True)

    if keep_context:
        rxns_in_model = [r.id for r in model.reactions]
        add_mod_pfba(model, weights={k: v for k, v in obj_dict.items() if k in rxns_in_model},
                     fraction_of_optimum=obj_frac) # some are probably removed

    result = GIMMEAnalysis(log={"name": model.name, "high_exp": high_exp, "obj_frac": obj_frac})
    result.add_result(obj_dict, rxn_expr_score,
                      fluxes=flux_df if return_fluxes else None,
                      model=new_model)
    return result
