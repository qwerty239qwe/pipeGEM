from typing import Dict

from optlang.symbolics import Zero
import cobra
from cobra.util import fix_objective_as_constraint
import numpy as np
import pandas as pd

from pipeGEM.utils import make_irrev_rxn, merge_irrevs_in_df
from pipeGEM.integration.continuous import register
from pipeGEM.analysis import add_mod_pfba, GIMMEAnalysis


@register
def apply_GIMME(model: cobra.Model,
                rxn_expr_score: Dict[str, float],
                low_exp: float,
                high_exp: float,
                protected_rxns = None,
                obj_frac: float = 0.8,
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
    exp_range = high_exp - low_exp
    obj_dict = {r_id: (high_exp - r_exp) / exp_range
                if low_exp < r_exp < high_exp else
                (1 if r_exp <= low_exp else 0)
                for r_id, r_exp in rxn_expr_score.items() if not np.isnan(r_exp) and r_id not in protected_rxns}
    with model:
        add_mod_pfba(model, weights=obj_dict, fraction_of_optimum=obj_frac)
        sol = model.optimize("minimize")

    flux_df = sol.to_frame()
    if remove_zero_fluxes:
        new_model = model.copy()
        to_remove = set(flux_df[abs(flux_df["fluxes"]) <= flux_threshold].index.to_list()) - set(protected_rxns)
        new_model.remove_reactions(list(to_remove), remove_orphans=True)

    if keep_context:
        rxns_in_model = [r.id for r in model.reactions]
        add_mod_pfba(model, weights={k: v for k, v in obj_dict.items() if k in rxns_in_model},
                     fraction_of_optimum=obj_frac) # some are probably removed

    result = GIMMEAnalysis(log={"name": model.name, "low_exp": low_exp, "high_exp": high_exp, "obj_frac": obj_frac})
    result.add_result(obj_dict, rxn_expr_score,
                      fluxes=flux_df if return_fluxes else None,
                      model=new_model if remove_zero_fluxes else None)
    return result


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
    sol_df["obj_coef"] = sol_df.index.to_series().apply(
        lambda x: obj_dict[x] if x in obj_dict else None)
    sol_df["inconsistency score"] = sol_df["obj_coef"] * sol_df["fluxes"]
    return sol_df