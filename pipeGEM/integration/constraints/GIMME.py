from typing import Dict

from optlang.symbolics import Zero
import cobra
import numpy as np
import pandas as pd

from pipeGEM.utils import make_irrev_rxn, merge_irrevs_in_df
from pipeGEM.integration.constraints import register


@register
def GIMME(model: cobra.Model,
          rxn_expr_score: Dict[str, float],
          low_exp: float,
          high_exp: float,
          obj_frac: float = 0.8,
          get_details: bool = True):
    """
    GLF implementation

    Parameters
    ----------
    model: cobra.Model
        The analyzed model, should has set RMF
    rxn_expr_score: Dict[str, float]
        A dict with rxn_ids as keys and expression values as values
    low_exp: float
        Expression value lower than this value is treated as 0
    high_exp: float
        Expression value higher than this value is treated as high_exp
    obj_frac: float

    get_details: bool

    Returns
    -------
    None
    """
    forward_prefix, backward_prefix = "_F_", "_R_"
    rev_map_to_irrev = {r.id: [r.id] for r in model.reactions}
    exp_range = high_exp - low_exp
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

    obj_dict = {mapped_r_id: (r_exp - high_exp) / exp_range
                if low_exp < r_exp < high_exp else
                (-1 if r_exp <= low_exp else 0)
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp)
                for mapped_r_id in rev_map_to_irrev[r_id]}
    model.objective = model.problem.Objective(Zero, sloppy=True)
    model.objective = {
        model.reactions.get_by_id(k): v for k, v in obj_dict.items() if v != 0
    }
    if get_details:
        return obj_dict


def _GIMME_post_process(sol_df, forward_prefix="_F_", backward_prefix="_R_"):
    if isinstance(sol_df, cobra.Solution):
        obj_v = sol_df.objective_value
        status = sol_df.status
        flux_df = sol_df.to_frame()
        return cobra.Solution(objective_value=obj_v,
                              status=status,
                              fluxes=merge_irrevs_in_df(flux_df, forward_prefix, backward_prefix)["fluxes"])
    elif isinstance(sol_df, pd.DataFrame):
        return merge_irrevs_in_df(sol_df, forward_prefix, backward_prefix)


def _GIMME_follow_up(obj_dict, **kwargs):
    sol_df = kwargs.get("sol_df").copy()
    print("in fu")
    sol_df["obj_coef"] = sol_df.index.to_series().apply(
        lambda x: obj_dict[x] if x in obj_dict else None)
    sol_df["inconsistency score"] = sol_df["obj_coef"] * sol_df["fluxes"]
    return sol_df